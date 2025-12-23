# 二分類 ECAPA-TDNN 微調腳本

import torch
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path
import speechbrain as sb
from speechbrain.lobes.features import Fbank
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN, Classifier
from speechbrain.processing.features import InputNormalization
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.utils.parameter_transfer import Pretrainer
from speechbrain.utils.metric_stats import EER
from data.vox1_loader import TrainDataset
import warnings 
import csv

warnings.filterwarnings("ignore")  # 忽略警告訊息
warnings.filterwarnings("ignore", category=FutureWarning)

# ====== 定義 SpeechBrain Brain 物件 ======
class ECAPA_Trainer(sb.Brain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化 best loss
        self.best_valid_loss = float("inf")
        self.best_eer = float('inf')
        
    def compute_forward(self, batch, stage):
        labels, wav1, wav2 = batch
        wav1 = wav1.to(self.device)
        wav2 = wav2.to(self.device)
        
        # 建立相對長度向量 (假設 batch 內音訊長度一致，設為全 1.0)
        # SpeechBrain 的 ECAPA 需要 wav_lens 作為輸入
        wav_lens = torch.ones(wav1.shape[0], device=self.device)

        # waveform 先做 mel‑fbanks 特徵
        feats1 = self.hparams.compute_features(wav1)
        feats1 = self.hparams.norm(feats1, wav_lens)
        
        feats2 = self.hparams.compute_features(wav2)
        feats2 = self.hparams.norm(feats2, wav_lens)

        # 經過 ECAPA‑TDNN 提取 Embedding
        embedding1 = self.modules.ecapa(feats1, wav_lens)
        embedding2 = self.modules.ecapa(feats2, wav_lens)

        # 先做標準化後做餘弦相似度
        embedding1 = nn.functional.normalize(embedding1, p=2, dim=2)
        embedding2 = nn.functional.normalize(embedding2, p=2, dim=2)
        
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        scores = cos(embedding1, embedding2)

        return scores, (embedding1, embedding2)

    def compute_objectives(self, predictions, batch, stage):
        scores, (embedding1, embedding2) = predictions
        labels, wav1, wav2 = batch
        labels = labels.to(self.device)

        # turn scores to float32
        scores = (scores.squeeze(1)).to(torch.float32)
        
        # 計算 Loss
        loss = self.hparams.loss_fn(scores, labels)
        
        # 如果是在驗證階段，可以記錄正確率
        if stage != sb.Stage.TRAIN:            
            self.val_scores.extend(scores.detach().cpu().tolist())
            self.val_labels.extend(labels.detach().cpu().tolist())
            
        return loss

    def on_stage_start(self, stage, epoch=None):
        # 每個 stage 開始時初始化正確率統計
        if stage != sb.Stage.TRAIN:
            self.error_metrics = sb.utils.metric_stats.ClassificationStats()
            # 為 EER 計算準備容器
            self.val_scores = []
            self.val_labels = []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.VALID:
            # 計算 valid loss/accuracy
            # stats = self.error_metrics.summarize()
            valid_loss = float(stage_loss)
            
            # ==================== EER 計算步驟 ====================
            # 從收集的標籤中分離出正樣本和負樣本的分數
            positive_scores = [self.val_scores[i] for i, label in enumerate(self.val_labels) if label == 1]
            negative_scores = [self.val_scores[i] for i, label in enumerate(self.val_labels) if label == 0]
            
            # 使用 SpeechBrain 內建的 EER 計算函數
            # 注意：函數需要 torch.tensor 作為輸入
            eer, threshold = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
            
            print(f"Epoch {epoch}: valid loss: {valid_loss:.4f}, EER: {eer:.2f}%")
            # =======================================================

            # ① 寫入同一份 log
            first_time = not LOG_PATH.exists()
            with open(LOG_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                if first_time:
                    writer.writerow(["epoch", "valid_loss", "EER"])
                writer.writerow([epoch, valid_loss, eer])

            # ② 若 EER 創新低 → 保存模型 (對於說話人識別，EER 通常比 loss 更重要)
            if eer < self.best_eer:
                self.best_eer = eer
                print(f"  📌 New best EER ({eer:.2f}%) → save model")

                # 模型 state dict + classifier state dict
                save_dict = {
                    "ecapa": self.modules.ecapa.state_dict(),
                    "classifier": self.modules.classifier.state_dict(),
                }
                # 儲存到同一檔案（覆蓋）
                torch.save(save_dict, f"{BEST_MODEL_PATH}/best_model.pt")
                print(f"  💾 Model saved to {BEST_MODEL_PATH}")

if __name__ == "__main__":
    
    # 定義 log 路徑 & best loss 初始值
    BEST_MODEL_PATH = Path("checkpoints/ecapa_train")
    LOG_PATH = Path("checkpoints/ecapa_train/train_log.csv")
    
    # ====== Dataset 設定 ======
    base_dir = Path("D:\\Dataset\\VoxCeleb1\\vox1_dev_wav\\wav")
    meta_dir = Path("D:\\Dataset\\VoxCeleb1\\vox1_meta.csv")
    
    dataset = TrainDataset(
        file_dir=base_dir,
        meta_dir=meta_dir
    )
        
    train_ds, valid_ds = dataset.split_train_valid(valid_ratio=0.1)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=0)

    # ====== SpeechBrain 的前處理 & model ======
    hparams = {
        "compute_features": Fbank(
            sample_rate=16000,
            n_mels=80,
        ),
       "norm": InputNormalization(norm_type="sentence", std_norm=True),

        # 說話者驗證(二分類)
        "classifier": nn.Linear(192, 1),

        "loss_fn": nn.BCEWithLogitsLoss(),

    }

    # 載入預訓練模型權重
    ecapa_model = ECAPA_TDNN(
        input_size=80,
        channels=[1024,1024,1024,1024,3072],
        lin_neurons=192,
    )

    # 用 Pretrainer 載入 pretrained embedding_model.ckpt
    pretrainer = Pretrainer(
        loadables={"model": ecapa_model},
        paths={"model": "official_pretrained_models/speechbrain_spkrec-ecapa-voxceleb/embedding_model.ckpt"}, # 或 HF repo 路徑
    )

    pretrainer.collect_files()
    pretrainer.load_collected()

    # 再把這個 pretrained backbone 放進你的 hparams
    hparams["ecapa"] = ecapa_model

    # ====== 建立 Trainer ======
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_folder = "checkpoints/ecapa_train"

    trainer = ECAPA_Trainer(
        modules={
            "ecapa": hparams["ecapa"],
            "classifier": hparams["classifier"],
        },
        opt_class=torch.optim.Adam,
        hparams=hparams,
        run_opts={"device": device},
        checkpointer=Checkpointer(save_folder),
    )

    trainer.checkpointer.add_recoverables({
        "model": trainer.modules,
    })

    # ====== 執行訓練 ======
    trainer.fit(
        epoch_counter=range(1, 6), # 訓練 5 個 Epoch
        train_set=train_loader,
        valid_set=valid_loader,
    )

    print("訓練完成 🎉")