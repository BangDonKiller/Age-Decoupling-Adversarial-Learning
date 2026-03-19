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
    # "ECAPA-TDNN": "speechbrain/spkrec-ecapa-voxceleb",
    "ECAPA-TDNN": "yangwang825/ecapa-tdnn-vox2",
    "ResNet34": "speechbrain/spkrec-resnet-voxceleb",
    "x-vector": "",
}
MODEL_ID = MODEL_list["ECAPA-TDNN"]

DATASET_INFO = {
    "VoxCeleb1": {
        "AUDIO_META_DIR": "D:\\Dataset\\VoxCeleb1\\vox1_meta.csv",
        "Vox-O": {
            "AUDIO_DIR": ["D:\\Dataset\\VoxCeleb1\\vox1_dev_wav\\wav", "D:\\Dataset\\VoxCeleb1\\vox1_test_wav\\wav"],
            "AUDIO_DATALIST": "D:\\Dataset\\VoxCeleb1\\Vox-O.txt",
            "audio_suffix": ".wav",
        },
        "Vox-E": {
            "AUDIO_DIR": ["D:\\Dataset\\VoxCeleb1\\vox1_dev_wav\\wav", "D:\\Dataset\\VoxCeleb1\\vox1_test_wav\\wav"],
            "AUDIO_DATALIST": "D:\\Dataset\\VoxCeleb1\\Vox-E.txt",
            "audio_suffix": ".wav",
        },
        "Vox-H": {
            "AUDIO_DIR": ["D:\\Dataset\\VoxCeleb1\\vox1_dev_wav\\wav", "D:\\Dataset\\VoxCeleb1\\vox1_test_wav\\wav"],
            "AUDIO_DATALIST": "D:\\Dataset\\VoxCeleb1\\Vox-H.txt",
            "audio_suffix": ".wav",
        },
        "Vox-CA5": {
            "AUDIO_DIR": ["D:\\Dataset\\VoxCeleb1\\vox1_dev_wav\\wav", "D:\\Dataset\\VoxCeleb1\\vox1_test_wav\\wav"],
            "AUDIO_DATALIST": "D:\\Dataset\\VoxCeleb1\\Vox-CA5.txt",
            "audio_suffix": ".wav",
        },
        "Vox-CA10": {
            "AUDIO_DIR": ["D:\\Dataset\\VoxCeleb1\\vox1_dev_wav\\wav", "D:\\Dataset\\VoxCeleb1\\vox1_test_wav\\wav"],
            "AUDIO_DATALIST": "D:\\Dataset\\VoxCeleb1\\Vox-CA10.txt",
            "audio_suffix": ".wav",
        },
        "Vox-CA15": {
            "AUDIO_DIR": ["D:\\Dataset\\VoxCeleb1\\vox1_dev_wav\\wav", "D:\\Dataset\\VoxCeleb1\\vox1_test_wav\\wav"],
            "AUDIO_DATALIST": "D:\\Dataset\\VoxCeleb1\\Vox-CA15.txt",
            "audio_suffix": ".wav",
        },
        "Vox-CA20": {
            "AUDIO_DIR": ["D:\\Dataset\\VoxCeleb1\\vox1_dev_wav\\wav", "D:\\Dataset\\VoxCeleb1\\vox1_test_wav\\wav"],
            "AUDIO_DATALIST": "D:\\Dataset\\VoxCeleb1\\Vox-CA20.txt",
            "audio_suffix": ".wav",
        },
    },
    "VoxCeleb2": {
        "AUDIO_DIR": "D:\\Dataset\\VoxCeleb2\\wav",
        "AUDIO_META_DIR": "D:\\Dataset\\VoxCeleb2\\vox2_meta.csv",
        "TRAIN_LIST": "D:\\Dataset\\VoxCeleb2\\train_list.txt",
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
    "MUSAN": {
        "AUDIO_DIR": "D:\\Dataset\\musan\\musan",
    },
    "RIR": {
        "AUDIO_DIR": "D:\\Dataset\\sim_rir_16k\\simulated_rirs_16k",
    },
}