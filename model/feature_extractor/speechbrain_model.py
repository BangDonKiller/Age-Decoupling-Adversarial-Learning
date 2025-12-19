import torch
import torch.nn as nn
# from params import param
from speechbrain.lobes.models.ResNet import ResNet
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
    
    Example:
        >>> extractor = SpeakerEmbeddingExtractor(
        ...     model_id="speechbrain/spkrec-ecapa-tdnn-voxceleb",
        ...     device="cuda"
        ... )
        >>> embedding = extractor.get_embedding_from_file("path/to/audio.wav")
        # 或者直接呼叫
        >>> embedding = extractor("path/to/audio.wav")
    """
    def __init__(self, model_id: str, device: str = "cpu", savedir: str = "pretrained_models"):
        super().__init__()
        
        self.model_id = model_id
        self.device = device
        self.savedir = Path(savedir) / model_id.replace("/", "_")

        print(f"--- 正在從 {model_id} 載入模型到 {device} ---")
        
        self.classifier = EncoderClassifier.from_hparams(
            source=model_id, 
            savedir=self.savedir,
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
            embeddings = embeddings.squeeze()
        
        return embeddings

# # --- 使用範例 ---
# if __name__ == "__main__":
#     # 輔助函數：生成假音檔
#     def create_dummy_audio(file_path, sample_rate=16000, duration_s=3):
#         num_samples = int(sample_rate * duration_s)
#         waveform = torch.randn(1, num_samples)
#         torchaudio.save(file_path, waveform, sample_rate)
#         print(f"已生成假音檔: {file_path}")

#     # 選擇運行設備
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"使用設備: {DEVICE}")

#     # 1. 建立 ECAPA-TDNN 嵌入提取器物件
#     #    模型只會在這一行被下載和載入一次
#     ecapa_extractor = SpeakerEmbeddingExtractor(
#         model_id="speechbrain/spkrec-ecapa-voxceleb",
#         device=DEVICE
#     )
    
#     # 2. 準備音檔
#     audio_file_1 = "dummy_audio_speaker_A.wav"
#     audio_file_2 = "dummy_audio_speaker_B.wav"
#     create_dummy_audio(audio_file_1)
#     create_dummy_audio(audio_file_2)

#     # 3. 提取嵌入
#     print("\n--- 正在提取嵌入 ---")
#     embedding = ecapa_extractor(audio_file_1)

#     # 5. 你也可以輕鬆建立另一個模型的提取器
#     print("\n--- 建立 ResNet (x-vector) 提取器 ---")
#     xvector_extractor = SpeakerEmbeddingExtractor(
#         model_id="speechbrain/spkrec-xvect-voxceleb",
#         device=DEVICE
#     )
#     xvector_embedding = xvector_extractor(audio_file_1)
#     print(f"ResNet (x-vector) 嵌入維度: {xvector_embedding.shape}")