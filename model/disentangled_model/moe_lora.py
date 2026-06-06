"""
LoRA + MoE 的 ECAPA-TDNN 包裝。

這個模組提供：
1. `importance_loss`：MoE 重要性平衡損失
2. `LoRAMoECAPAModel`：三個 LoRA 專家 + 全局 router，可切換軟加權或 top-K 融合

使用方式：
    model = LoRAMoECAPAModel(
        C=1024,
        n_class=num_speakers,
        pretrained_path="pretrained_models/pretrain.model",
        expert_adapter_paths=[path1, path2, path3],
    )
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel

from loss.arcface import ArcMarginProduct
from model.feature_extractor.ecapa_tdnn import ECAPA_TDNN


def importance_loss(
    gate_weights: torch.Tensor,
    w_importance: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """計算 MoE 的重要性平衡損失（CV 比值最小化）。"""
    # 先對 batch 維度加總，得到每個專家的總重要性。
    importance = gate_weights.sum(dim=0)

    # 變異係數 CV = std / mean；加 eps 避免除以 0。
    mean = importance.mean()
    std = importance.std(unbiased=False)
    cv = std / (mean + eps)

    # CV 平方後乘上縮放係數，得到標量 loss。
    return w_importance * (cv * cv)


def entropy_minimization_loss(
    gate_weights: torch.Tensor,
    w_importance: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """計算 Router 權重的平均信息熵，作為要最小化的罰項。"""
    gw = gate_weights.clamp(min=eps)
    entropy_per_sample = -(gw * gw.log()).sum(dim=1)
    mean_entropy = entropy_per_sample.mean()
    return w_importance * mean_entropy


class ImportanceLoss(nn.Module):
    """重要性平衡損失的 Module 版本，方便直接掛進訓練流程。"""

    def __init__(self, w_importance: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.w_importance = w_importance
        self.eps = eps

    def forward(self, gate_weights: torch.Tensor) -> torch.Tensor:
        return importance_loss(gate_weights, self.w_importance, self.eps)


class EntropyMinimizationLoss(nn.Module):
    """熵最小化損失的 Module 版本。"""

    def __init__(self, w_importance: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.w_importance = w_importance
        self.eps = eps

    def forward(self, gate_weights: torch.Tensor) -> torch.Tensor:
        return entropy_minimization_loss(gate_weights, self.w_importance, self.eps)


def _extract_state_dict(checkpoint: object) -> dict:
    """從 checkpoint 中抽出 state dict，兼容 raw state_dict / checkpoint dict。"""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"不支援的 checkpoint 型別: {type(checkpoint)!r}")


def _load_flexible_weights(module: nn.Module, path: str) -> None:
    """彈性載入權重，支援 speaker_encoder 前綴與 module 前綴。"""
    checkpoint = torch.load(path, map_location="cpu")
    loaded_state = _extract_state_dict(checkpoint)
    current_state = module.state_dict()

    for name, param in loaded_state.items():
        candidate_names = [name]
        if name.startswith("module."):
            candidate_names.append(name.replace("module.", "", 1))
        if name.startswith("speaker_encoder."):
            candidate_names.append(name.replace("speaker_encoder.", "", 1))
        if name.startswith("module.speaker_encoder."):
            candidate_names.append(name.replace("module.speaker_encoder.", "", 1))

        matched_name = None
        for candidate in candidate_names:
            if candidate in current_state and current_state[candidate].shape == param.shape:
                matched_name = candidate
                break

        if matched_name is None:
            continue

        current_state[matched_name].copy_(param)


def _build_backbone(pretrained_path: str, C: int) -> ECAPA_TDNN:
    """建立並載入一個 ECAPA-TDNN backbone。"""
    backbone = ECAPA_TDNN(C=C)
    _load_flexible_weights(backbone, pretrained_path)
    return backbone


class LoRAMoECAPAModel(nn.Module):
    """
    三專家 LoRA MoE 架構。

    - `router_backbone`：提供全局 embedding，供 router 產生 gate weights
    - `experts`：三個已訓練完成的 LoRA 專家
    - `classifier`：原先的 ArcFace 分類損失
    - `importance_loss`：讓 router 不要過度偏向單一專家
    - `gate_strategy`：`soft` 或 `topk`
    """

    def __init__(
        self,
        C: int,
        n_class: int,
        pretrained_path: str,
        expert_adapter_paths: Sequence[str],
        m: float = 0.2,
        s: float = 64.0,
        router_hidden_dim: int = 128,
        w_importance: float = 0.1,
        eps: float = 1e-6,
        gate_strategy: str = "soft",
        top_k: int = 1,
    ):
        super().__init__()

        if len(expert_adapter_paths) != 3:
            raise ValueError("目前這個 MoE 架構預設只接受 3 個專家模型。")

        self.num_experts = 3
        self.w_importance = w_importance
        self.eps = eps
        self.gate_strategy = gate_strategy.lower().replace("-", "").replace("_", "")
        self.top_k = top_k

        if self.gate_strategy not in {"soft", "topk"}:
            raise ValueError("gate_strategy 只支援 'soft' 或 'topk'。")
        if self.gate_strategy == "topk" and not (1 <= self.top_k <= self.num_experts):
            raise ValueError(f"top_k 必須介於 1 和 {self.num_experts} 之間。")

        # 全局 router 用的共享 backbone，固定不訓練。
        self.router_backbone = _build_backbone(pretrained_path, C)
        self.router_backbone.requires_grad_(False)
        self.router_backbone.eval()

        # 三個 LoRA 專家，皆視為已訓練完成並凍結。
        experts = []
        for adapter_path in expert_adapter_paths:
            expert = _build_backbone(pretrained_path, C)
            expert = PeftModel.from_pretrained(expert, adapter_path)
            expert.requires_grad_(False)
            expert.eval()
            experts.append(expert)
        self.experts = nn.ModuleList(experts)

        # Router 用全局 embedding 產生三個 expert 的 soft gate weights。
        self.router = nn.Sequential(
            nn.Linear(192, router_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(router_hidden_dim, self.num_experts),
        )

        # 原先的分類頭仍保留。
        self.speaker_loss = ArcMarginProduct(
            in_features=192,
            out_features=n_class,
            m=m,
            s=s,
        )

        self.importance_loss_fn = ImportanceLoss(w_importance=w_importance, eps=eps)

    def _encode_backbone(self, backbone: nn.Module, waveform: torch.Tensor, aug: bool) -> torch.Tensor:
        """提取單一路徑的 speaker embedding。"""
        with torch.no_grad():
            return backbone(waveform, aug=aug)

    def extract_gate_weights(self, waveform: torch.Tensor, aug: bool = False) -> torch.Tensor:
        """先經由全局 router 算出每個樣本對三個專家的權重。"""
        global_embedding = self._encode_backbone(self.router_backbone, waveform, aug=aug)
        gate_logits = self.router(global_embedding)
        gate_weights = F.softmax(gate_logits, dim=1)

        if self.gate_strategy == "topk":
            _, topk_idx = gate_weights.topk(self.top_k, dim=1)
            topk_mask = torch.zeros_like(gate_weights).scatter_(1, topk_idx, 1.0)
            gate_weights = gate_weights * topk_mask
            gate_weights = gate_weights / gate_weights.sum(dim=1, keepdim=True).clamp_min(self.eps)

        return gate_weights

    def extract_fused_embedding(self, waveform: torch.Tensor, aug: bool = False):
        """回傳 MoE 融合後的 embedding，以及對應的 gate weights。"""
        gate_weights = self.extract_gate_weights(waveform, aug=aug)

        expert_embeddings = []
        for expert in self.experts:
            emb = self._encode_backbone(expert, waveform, aug=aug)
            emb = F.normalize(emb, p=2, dim=1)  # L2 normalize 後再融合
            expert_embeddings.append(emb)

        # exchange the last embedding with a random noise
        noise = torch.randn_like(expert_embeddings[-1])
        # expert_embeddings[1] = noise
        # expert_embeddings[2] = noise

        expert_stack = torch.stack(expert_embeddings, dim=1)  # [B, E, D]
        fused_embedding = torch.sum(gate_weights.unsqueeze(-1) * expert_stack, dim=1)
        return fused_embedding, gate_weights

    def forward(self, waveform: torch.Tensor, labels: torch.Tensor = None, aug: bool = False):
        """
        前向傳播。

        若有 labels，回傳總 loss、分類 loss、重要性 loss、acc、gate_weights；
        若沒有 labels，回傳 fused embedding 與 gate_weights。
        """
        fused_embedding, gate_weights = self.extract_fused_embedding(waveform, aug=aug)

        if labels is None:
            return fused_embedding, gate_weights

        cls_loss, acc = self.speaker_loss(fused_embedding, labels)
        imp_loss = self.importance_loss_fn(gate_weights)
        total_loss = cls_loss + imp_loss
        return total_loss, cls_loss, imp_loss, acc, gate_weights
