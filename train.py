import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from pathlib import Path
import speechbrain as sb
from speechbrain.lobes.features import Fbank
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN, Classifier
# from speechbrain.nnet.losses import AdditiveAngularMarginLoss
from speechbrain.processing.features import InputNormalization
from speechbrain.utils.checkpoints import Checkpointer
from data.vox2_loader import TrainDataset
import warnings 
import csv

# 只忽略 AMP deprecated 的 FutureWarning
warnings.filterwarnings(
    "ignore",
    message=".*torch.cuda.amp.custom_fwd.*",
    category=FutureWarning
)

# 只忽略 Windows symlink 的 UserWarning
warnings.filterwarnings(
    "ignore",
    message=".*Requested Pretrainer collection using symlinks.*",
    category=UserWarning
)

# ====== 定義 SpeechBrain Brain 物件 ======
class ECAPA_Trainer(sb.Brain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化 best loss
        self.best_valid_loss = float("inf")
        
    def compute_forward(self, batch, stage):
        # 修正處 3: 配合 DataLoader 的返回內容 (只有 wavs 和 labels)
        wavs, labels = batch
        wavs = wavs.to(self.device)
        
        # 建立相對長度向量 (假設 batch 內音訊長度一致，設為全 1.0)
        # SpeechBrain 的 ECAPA 需要 wav_lens 作為輸入
        wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # waveform 先做 mel‑fbanks 特徵
        feats = self.hparams.compute_features(wavs)
        feats = self.hparams.norm(feats, wav_lens)

        # 經過 ECAPA‑TDNN 提取 Embedding
        embeddings = self.modules.ecapa(feats, wav_lens)

        # 經過分類器得到分數
        scores = self.modules.classifier(embeddings)

        return scores, embeddings

    def compute_objectives(self, predictions, batch, stage):
        scores, embeddings = predictions
        wavs, labels = batch
        labels = labels.to(self.device)

        # turn scores to float32
        scores = (scores.squeeze(1)).to(torch.float32)
        
        # 計算 Loss
        loss = self.hparams.loss_fn(scores, labels)
        
        # 如果是在驗證階段，可以記錄正確率
        if stage != sb.Stage.TRAIN:
            # step 1: logits → class index
            preds = torch.argmax(scores, dim=1)

            # step 2: 轉成 list of strings
            pred_list  = [int(x) for x in preds.detach().cpu().tolist()]
            label_list = [int(x) for x in labels.detach().cpu().tolist()]
            id_list    = [int(i) for i in range(len(pred_list))]

            # append 正確用法
            self.error_metrics.append(id_list, pred_list, label_list)
            
        return loss

    def on_stage_start(self, stage, epoch=None):
        # 每個 stage 開始時初始化正確率統計
        if stage != sb.Stage.TRAIN:
            self.error_metrics = sb.utils.metric_stats.ClassificationStats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.VALID:
            # 計算 valid loss/accuracy
            stats = self.error_metrics.summarize()
            acc_val = stats["accuracy"]
            valid_loss = float(stage_loss)

            print(f"Epoch {epoch}: valid loss: {valid_loss:.4f}, acc: {acc_val:.4f}")

            # ① 寫入同一份 log
            first_time = not LOG_PATH.exists()
            with open(LOG_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                if first_time:
                    writer.writerow(["epoch", "valid_loss", "accuracy"])
                writer.writerow([epoch, valid_loss, acc_val])

            # ② 若 valid loss 創新低 → 保存模型
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                print(f"  📌 New best valid loss ({valid_loss:.4f}) → save model")

                # 模型 state dict + classifier state dict
                save_dict = {
                    "ecapa": self.modules.ecapa.state_dict(),
                    "classifier": self.modules.classifier.state_dict(),
                }
                # 儲存到同一檔案（覆蓋）
                torch.save(save_dict, BEST_MODEL_PATH)


if __name__ == "__main__":
    
    # 定義 log 路徑 & best loss 初始值
    BEST_MODEL_PATH = Path("checkpoints/ecapa_train/best_model.pth")
    LOG_PATH = Path("checkpoints/ecapa_train/train_log.csv")
    
    # ====== Dataset 設定 ======
    base_dir = Path("D:\\Dataset\\VoxCeleb2\\vox2_dev_wav\\dev\\aac")
    
    # 1. 建立標籤映射
    all_data = []
    if not base_dir.exists():
        raise FileNotFoundError(f"找不到路徑: {base_dir}")

    speakers = sorted([d.name for d in base_dir.iterdir() if d.is_dir()])
    spk_to_idx = {spk: i for i, spk in enumerate(speakers)}
    num_speakers = len(speakers)
    
    print(f"找到 {num_speakers} 位講者。正在收集檔案清單...")

    for spk in speakers:
        spk_path = base_dir / spk
        for wav_path in spk_path.rglob("*.m4a"): # Vox2 通常是 .m4a
            all_data.append((str(wav_path), spk_to_idx[spk]))
        
        # 限制資料量測試用 (正式訓練請移除這兩行)
        if len(all_data) > 2000: 
            break 

    # 2. 切分資料集 (不重疊)
    labels_list = [item[1] for item in all_data]
    train_list, valid_list = train_test_split(
        all_data, 
        test_size=0.1, 
        random_state=42, 
        stratify=labels_list
    )

    train_ds = TrainDataset(base_dir, train_list)
    valid_ds = TrainDataset(base_dir, valid_list)

    # num_workers 在 Windows 建議先設為 0 避免 Multiprocessing 報錯
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=0)

    # ====== SpeechBrain 的前處理 & model ======
    hparams = {
        "compute_features": Fbank(
            sample_rate=16000,
            n_mels=80,
        ),
       "norm": InputNormalization(norm_type="sentence", std_norm=True),

        "ecapa": ECAPA_TDNN(
            input_size=80, 
            channels=[1024, 1024, 1024, 1024, 3072], 
            lin_neurons=192
        ),

        # 修正處 2: out_neurons 必須等於實際講者人數
        "classifier": Classifier(input_size=192, out_neurons=len(set(labels_list))),

        # "loss_fn": AdditiveAngularMarginLoss(margin=0.3, scale=30.0),
        "loss_fn": nn.CrossEntropyLoss(), 
    }

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
    # trainer.fit 需要 (epoch_counter, train_set, valid_set)
    trainer.fit(
        epoch_counter=range(1, 11), # 訓練 10 個 Epoch
        train_set=train_loader,
        valid_set=valid_loader,
    )

    print("訓練完成 🎉")