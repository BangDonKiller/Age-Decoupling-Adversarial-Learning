import torch
import torch.nn as nn
from speechbrain.inference import EncoderClassifier
from pathlib import Path
import warnings

torch.manual_seed(42)

# 精準地忽略關於 speechbrain.pretrained 的 UserWarning
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message="Module 'speechbrain.pretrained' was deprecated"
)

class SpeakerEmbeddingExtractor(nn.Module):
    """
    一個用於提取說話人嵌入的類別，封裝了 SpeechBrain 的預訓練模型。

    這個類別在初始化時會載入指定的預訓練模型（例如 ECAPA-TDNN 或 ResNet）。
    它可以被呼叫來從音檔路徑或原始音訊張量中提取嵌入向量。

    Args:
        model_id (str): Hugging Face Hub 上的模型 ID。
        device (str, optional): 運行的設備 ('cpu' or 'cuda')。預設為 'cpu'。
        savedir (str, optional): 模型下載後儲存的本地路徑。預設為 'pretrained_models'。
    """
    def __init__(self, model_id: str, device: str = "cpu", savedir: str = "pretrained_models"):
        super().__init__()
        
        self.model_id = model_id
        self.device = device
        self.savedir = Path(savedir) / model_id.replace("/", "_")

        print(f"--- 正在從 {model_id} 載入模型到 {device} ---")
        
        self.classifier = EncoderClassifier.from_hparams(
            source=model_id, 
            run_opts={"device": device}
        )

        self.classifier.eval() # 確保模型處於評估模式

    def forward(self, audio_input) -> torch.Tensor:
        """
        從音訊輸入中提取嵌入向量。
        
        Args:
            audio_input (str or torch.Tensor): 
            已經載入的音訊波形張量 (shape: [batch, samples] or [samples])。
            張量應該是 16kHz 的取樣率。
        
        Returns:
            torch.Tensor: 提取出的嵌入向量，shape 為 [embedding_dim]。
                          如果是批次輸入，shape 為 [batch, embedding_dim]。
        """
        
        if isinstance(audio_input, torch.Tensor):
            # 如果輸入是張量，直接使用
            waveform = audio_input
        else:
            raise TypeError("輸入必須是檔案路徑 (str) 或音訊張量 (torch.Tensor)。")

        # 確保 waveform 是 2D (batch, samples) 且在正確的設備上
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.squeeze(1)  # 假設輸入是 (batch, 1, samples)
        
        waveform = waveform.to(self.device)

        # 提取嵌入 (在 no_grad 上下文中以節省記憶體和加速)
        with torch.no_grad():
            embeddings = self.classifier.encode_batch(waveform)
            # 輸出 shape 是 (batch, 1, embedding_dim)，我們將其壓縮
            embeddings = embeddings.squeeze(1)
        
        return embeddings


class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN + 年齡分類頭。
    
    支持兩種模式：
    - 'inference': 使用官方預訓練權重進行推論
    - 'training': 隨機初始化權重，從零開始訓練
    
    Args:
        model_id (str): Hugging Face Hub 上的模型 ID
        num_age_groups (int): 年齡分組數量
        mode (str): 'inference' 或 'training'
        embedding_dim (int): Speaker embedding 維度（通常 192）
        device (str): 設備 ('cpu' 或 'cuda')
    """
    def __init__(
        self, 
        model_id: str, 
        mode: str = "training",
        embedding_dim: int = 192,
        device: str = "cpu",
    ):
        super().__init__()
        
        if mode not in ["inference", "training"]:
            raise ValueError(f"mode 必須是 'inference' 或 'training'，得到 {mode}")
        
        self.mode = mode
        self.embedding_dim = embedding_dim
        
        print(f"構建 ECAPA-TDNN 模型...")
        print(f"  模式: {mode}")
        
        # 忽略關於 speechbrain.pretrained 的 UserWarning
        warnings.filterwarnings(
            'ignore',
            category=UserWarning,
            message="Module 'speechbrain.pretrained' was deprecated"
        )
        
        # 載入預訓練分類器以獲得架構
        classifier = EncoderClassifier.from_hparams(
            source=model_id, 
            run_opts={"device": device},
            savedir=str(Path("pretrained_models") / model_id.replace("/", "_"))
        )

        # 依照 SpeechBrain 官方 encode_batch 的資料流，保留前處理與嵌入模型。
        # 官方流程是：waveform -> compute_features -> mean_var_norm -> embedding_model。
        if not hasattr(classifier.mods, "compute_features"):
            raise RuntimeError("SpeechBrain 模型缺少 compute_features，無法使用官方前處理流程。")
        if not hasattr(classifier.mods, "mean_var_norm"):
            raise RuntimeError("SpeechBrain 模型缺少 mean_var_norm，無法使用官方前處理流程。")
        if not hasattr(classifier.mods, "embedding_model"):
            raise RuntimeError("SpeechBrain 模型缺少 embedding_model，無法提取 speaker embedding。")

        self.compute_features = classifier.mods.compute_features
        self.mean_var_norm = classifier.mods.mean_var_norm
        self.embedding_model = classifier.mods.embedding_model
        print("  ✓ 已載入 SpeechBrain 官方前處理流程：compute_features -> mean_var_norm -> embedding_model")
        
        # 根據 mode 決定是否重新初始化權重
        if mode == "training":
            self._reinitialize_weights(self.embedding_model)
            print("  ✓ 所有權重已重新初始化為隨機值（訓練模式）")
        else:
            print("  ✓ 使用官方預訓練權重（推論模式）")
        
        # 只讓嵌入網路決定是否可訓練；前處理模組維持官方設定。
        self.embedding_model.requires_grad_(mode == "training")
        self.compute_features.requires_grad_(False)
        self.mean_var_norm.requires_grad_(False)
        print(f"  → Embedding model {'可訓練' if mode == 'training' else '凍結'}（{mode} 模式）")
        
    def _reinitialize_weights(self, model: nn.Module) -> None:
        """
        遞迴地重新初始化模型中所有層的權重和偏置。
        
        使用標準的初始化策略：
        - Conv/Linear: Kaiming normal
        - BatchNorm: weight=1, bias=0
        - LSTM: 正交初始化
        """
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def train(self, mode: bool = True):
        super().train(mode)
        self.compute_features.eval()
        self.mean_var_norm.eval()
        self.embedding_model.train(mode and self.mode == "training")
        return self
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        從音訊波形中提取 embedding 並進行年齡分類。
        
        Args:
            waveform: 形狀 [batch, samples]，16kHz 單聲道音訊
            
        Returns:
            embedding: 形狀 [batch, embedding_dim] 的說話人嵌入向量
        """
        # 依照 SpeechBrain 官方 encode_batch 的方式處理原始波形。
        # 1) 接受 [T]、[B, T] 或 [B, 1, T]
        # 2) 補成 batch 形式
        # 3) 轉成 [B, T]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform.squeeze(1)

        model_device = next(self.embedding_model.parameters()).device
        waveform = waveform.to(model_device)
        waveform = waveform.float()

        wav_lens = torch.ones(waveform.shape[0], device=waveform.device)
        feats = self.compute_features(waveform)
        feats = self.mean_var_norm(feats, wav_lens)
        embedding = self.embedding_model(feats, wav_lens)

        # 輸出通常是 [batch, 1, embedding_dim]，這裡壓成 [batch, embedding_dim]
        if embedding.ndim == 3 and embedding.shape[1] == 1:
            embedding = embedding.squeeze(1)

        return embedding
