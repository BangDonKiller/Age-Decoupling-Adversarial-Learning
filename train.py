# train.py
import os
import torch
from torch.utils.data import DataLoader

import speechbrain as sb
from speechbrain.lobes.features import Fbank
from speechbrain.lobes.models import ECAPA_TDNN, Classifier
from speechbrain.nnet.losses import AdditiveAngularMarginLoss
from speechbrain.utils.checkpoints import Checkpointer
from data.vox2_loader import TrainDataset


# ====== 定義 SpeechBrain Brain 物件 ======
class ECAPA_Trainer(sb.Brain):
    def compute_forward(self, batch, stage):
        wavs, wav_lens, labels = batch

        # waveform 先做 mel‑fbanks 特徵
        feats = self.hparams.compute_features(wavs)
        feats = self.hparams.norm(feats, wav_lens)

        # 經過 ECAPA‑TDNN
        embeddings = self.modules.ecapa(feats, wav_lens)

        # classifier head
        scores = self.modules.classifier(embeddings)

        return scores, embeddings

    def compute_objectives(self, predictions, batch, stage):
        scores, embeddings = predictions
        _, _, labels = batch

        # AAM‑Softmax 或其他 loss
        loss = self.hparams.loss_fn(scores, labels)
        return loss

if __name__ == "__main__":
    # ====== Dataset 設定 ======
    train_dir = ""
    valid_dir = ""

    train_ds = TrainDataset(train_dir)
    valid_ds = TrainDataset(valid_dir)

    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=64, shuffle=False, num_workers=0
    )

    # ====== SpeechBrain 的前處理 & model ======
    hparams = {
        # mel feature
        "compute_features": Fbank(
            sample_rate=16000,
            n_mels=80,
            n_fft=512,
            hop_length=160,
            win_length=400,
        ),
        "norm": sb.lobes.normalization.MeanVarNorm(),

        # ECAPA‑TDNN model
        "ecapa": ECAPA_TDNN(input_size=80, lin_neurons=192),

        # classifier head
        "classifier": Classifier(input_size=192, out_neurons=5994),

        # loss function（例如 AAM‑Softmax）
        "loss_fn": AdditiveAngularMarginLoss(margin=0.3, scale=30.0),
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

    # ====== 執行訓練 ======
    trainer.fit(train_loader, valid_loader, valid_loader)

    print("訓練完成 🎉")
