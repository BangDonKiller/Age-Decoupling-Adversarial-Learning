import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """
    ArcFace 的核心分類頭（ArcMarginProduct）。

    這個模組會把一般 softmax 的 logits:
        logits = W^T x
    改成帶角度邊界（angular margin）的形式，讓同一類的特徵更緊密、不同類更分離。

    參數說明:
        in_features: 輸入特徵維度（例如 mu_id 的維度）
        out_features: 類別數（例如 speaker 數）
        s: feature/logit 的縮放係數，常見值 30.0
        m: 角度邊界 margin，常見值 0.30~0.50
        easy_margin: 是否使用 easy margin 策略
    """

    def __init__(
        self,
        in_features: 192,
        out_features: int,
        s: float = 30.0,
        m: float = 0.35,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.easy_margin = easy_margin

        # ArcFace 可學習類別中心權重矩陣。
        # shape: [out_features, in_features]
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # 初始化 margin 及衍生常數。
        self.set_margin(m)
        
        self.ce = nn.CrossEntropyLoss()

    def set_margin(self, margin: float) -> None:
        """更新 ArcFace margin，並同步刷新 forward 會用到的常數。"""
        m = float(margin)
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        前向計算。

        兩種模式:
        1) 訓練模式（labels 不為 None）:
           對目標類別套用 cos(theta + m)，其餘類別保留 cos(theta)。
        2) 推論模式（labels 為 None）:
           回傳一般 cosine logits（乘上 s）。

        參數:
            features: [B, in_features]
            labels:   [B]

        回傳:
            logits: [B, out_features]
        """
        # 先做 L2 normalize，讓內積變成 cosine。
        normed_features = F.normalize(features, p=2, dim=1)
        normed_weight = F.normalize(self.weight, p=2, dim=1)

        # cosine = cos(theta)
        cosine = F.linear(normed_features, normed_weight)

        # 推論模式：不需要 margin，直接回傳縮放後 logits。
        if labels is None:
            return cosine * self.s

        # sine = sqrt(1 - cos^2(theta))，加 clamp 避免數值誤差變成負數。
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))

        # phi = cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # 兩種 margin 策略。
        if self.easy_margin:
            # easy margin: 若 cosine <= 0，就不強推 margin，避免早期訓練不穩。
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # 原始 ArcFace 常用策略。
            # 若 theta 太大（cosine 太小），避免 phi 過度彎折。
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # one-hot 指示目標類別。
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # 只對目標類別套 margin，其餘類別保持原 cosine。
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 最後乘上縮放係數，讓 softmax 梯度尺度更適合訓練。
        logits = logits * self.s
        loss = self.ce(logits, labels)
        
        acc = (logits.argmax(dim=1) == labels).float().mean().item() * 100.0
        
        return loss, acc
