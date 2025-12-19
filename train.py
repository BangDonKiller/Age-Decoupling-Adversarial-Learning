import torch
from model.feature_extractor.speechbrain_model import SpeakerEmbeddingExtractor
from data.vox2_pretrain_loader import create_dataloader
from tqdm import tqdm
import numpy as np
import warnings

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

# ========== 配置 ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_list = {
    "ECAPA-TDNN": "speechbrain/spkrec-ecapa-voxceleb",
    "ResNet34": "speechbrain/spkrec-resnet-voxceleb",
    "x-vector": "",
}
MODEL_ID = MODEL_list["ResNet34"]
AUDIO_DIR = "D:\\Dataset\\VoxCeleb2\\vox2_dev_wav\\dev\\aac"
AUDIO_LIST_DIR = "D:\\Dataset\\VoxCeleb2\\train_list.txt"
AUDIO_META_DIR = "D:\\Dataset\\VoxCeleb2\\vox2_meta.csv"
BATCH_SIZE = 64

# ========== 載入模型 ==========
print(f"使用設備: {DEVICE}")
speaker_extractor = SpeakerEmbeddingExtractor(
    model_id=MODEL_ID,
    device=DEVICE
)

# ========== Dataloader ==========
loader = create_dataloader(
    audio_dir=AUDIO_DIR,
    audio_list_dir=AUDIO_LIST_DIR,
    audio_meta_dir=AUDIO_META_DIR,
    batch_size=BATCH_SIZE
)

# ========== Batch 推論 ==========
all_embeddings = []
speaker_ids = []
genders = []

with torch.no_grad():
    for waveforms, speaker_id, gender in tqdm(loader):
        emb = speaker_extractor(waveforms)
        all_embeddings.append(emb.cpu())
        speaker_ids.extend(speaker_id)
        genders.extend(gender)

all_embeddings = torch.cat(all_embeddings, dim=0)

# ========== 儲存 embeddings ==========
MODEL = next((k for k, v in MODEL_list.items() if v == MODEL_ID), None)
torch.save({"embeddings": all_embeddings, "speaker_ids": speaker_ids, "genders": genders}, f"result/{MODEL}/{MODEL}_embeddings.pt")
print(f"推論完成，共 {len(all_embeddings)} 個向量")


