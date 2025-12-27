import torch
from model.feature_extractor.speechbrain_model import SpeakerEmbeddingExtractor
# from data.vox1_loader import InferenceDataset
# from data.vox2_loader import InferenceDataset
# from data.librispeech_loader import InferenceDataset
from data.GLOBE_loader import InferenceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
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
MODEL_ID = MODEL_list["ECAPA-TDNN"]
DATASET_INFO = {
    "VoxCeleb1": {
        "AUDIO_DIR": "D:\\Dataset\\VoxCeleb1\\vox1_dev_wav\\wav",
        "AUDIO_LIST_DIR": None,
        "AUDIO_META_DIR": "D:\\Dataset\\VoxCeleb1\\vox1_meta.csv",
        "audio_suffix": ".wav",
    },
    "VoxCeleb2": {
        "AUDIO_DIR": "D:\\Dataset\\VoxCeleb2\\vox2_dev_wav\\dev\\aac",
        "AUDIO_META_DIR": "D:\\Dataset\\VoxCeleb2\\vox2_meta2.csv",
        "audio_suffix": ".m4a",
    },
    "LibriSpeech": {
        "AUDIO_DIR": "D:\\Dataset\\LibriSpeech",
        "AUDIO_META_DIR": "D:\\Dataset\\LibriSpeech\\SPEAKERS.txt",
        "audio_suffix": ".flac",
    },
    "GLOBE": {
        "AUDIO_DIR": list(Path("D:\\Dataset\\GLOBE\\data").rglob("train-*.parquet")),
        "audio_suffix": ".parquet", 
    },
}

DATASET = "GLOBE"
AUDIO_DIR = DATASET_INFO[DATASET]["AUDIO_DIR"]
# AUDIO_META_DIR = DATASET_INFO[DATASET]["AUDIO_META_DIR"]
AUDIO_SUFFIX = DATASET_INFO[DATASET]["audio_suffix"]

BATCH_SIZE = 64

# ========== 載入模型 ==========
print(f"使用設備: {DEVICE}")
speaker_extractor = SpeakerEmbeddingExtractor(
    model_id=MODEL_ID,
    device=DEVICE
)

# ========== Dataloader ==========
# dataset = InferenceDataset(AUDIO_DIR, AUDIO_META_DIR, suffix=AUDIO_SUFFIX)
dataset = InferenceDataset(AUDIO_DIR, suffix=AUDIO_SUFFIX)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

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
torch.save({"embeddings": all_embeddings, "speaker_ids": speaker_ids, "genders": genders}, f"result/{MODEL}/{DATASET}_{MODEL}_embeddings.pt")
print(f"推論完成，共 {len(all_embeddings)} 個向量")


