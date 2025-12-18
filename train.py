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
    "ResNet34": "",
    "x-vector": "",
}
MODEL_ID = MODEL_list["ECAPA-TDNN"]
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
    for waveforms, gender, paths in tqdm(loader):
        for waveform in waveforms:
            emb = speaker_extractor(waveform)
            all_embeddings.append(emb.cpu())
            speaker_ids.extend(paths)
            genders.extend(gender)

all_embeddings = torch.cat(all_embeddings, dim=0)

# ========== 儲存 embeddings ==========
torch.save({"embeddings": all_embeddings, "speaker_ids": speaker_ids, "genders": genders}, "ecapa_embeddings.pt")
np.save("ecapa_embeddings.npy", all_embeddings.numpy())
np.save("speaker_ids.npy", speaker_ids.numpy())
np.save("genders.npy", genders.numpy())

print(f"推論完成，共 {len(all_embeddings)} 個向量")


