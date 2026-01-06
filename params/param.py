import warnings
import torch
from pathlib import Path


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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64

MODEL_list = {
    "ECAPA-TDNN": "speechbrain/spkrec-ecapa-voxceleb",
    "ResNet34": "speechbrain/spkrec-resnet-voxceleb",
    "x-vector": "",
}
MODEL_ID = MODEL_list["ECAPA-TDNN"]

DATASET_INFO = {
    "VoxCeleb1": {
        "AUDIO_DIR": ["D:\\Dataset\\VoxCeleb1\\vox1_dev_wav\\wav", "D:\\Dataset\\VoxCeleb1\\vox1_test_wav\\wav"],
        "AUDIO_META_DIR": "D:\\Dataset\\VoxCeleb1\\vox1_meta.csv",
        "audio_suffix": ".wav",
    },
    "VoxCeleb2": {
        "AUDIO_DIR": "D:\\Dataset\\VoxCeleb2\\vox2_dev_wav\\dev\\aac",
        "AUDIO_META_DIR": "D:\\Dataset\\VoxCeleb2\\vox2_meta.csv",
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
    "TIMIT": {
        "AUDIO_DIR": "D:\\Dataset\\TIMIT\\data\\TRAIN",
        "AUDIO_META_DIR": "D:\\Dataset\\TIMIT\\train_meta_data.csv",
        "audio_suffix": ".wav",
    },
}