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

BATCH_SIZE = 128
LEARNING_RATE = 1e-3  # 從零開始訓練，用更高的學習率
NUM_WORKERS = 0

MODEL_list = {
    # "ECAPA-TDNN": "speechbrain/spkrec-ecapa-voxceleb",
    "ECAPA-TDNN": "yangwang825/ecapa-tdnn-vox2",
    "ResNet34": "speechbrain/spkrec-resnet-voxceleb",
    "x-vector": "",
}
MODEL_ID = MODEL_list["ECAPA-TDNN"]

DATASET_INFO = {
    "VoxCeleb1": {
        "AUDIO_META_DIR": "/app/dataset/VoxCeleb1/vox1_meta.csv",
        "Vox-O": {
            "AUDIO_DIR": ["/app/dataset/VoxCeleb1/vox1_dev_wav/wav", "/app/dataset/VoxCeleb1/vox1_test_wav/wav"],
            "AUDIO_DATALIST": "/app/dataset/VoxCeleb1/Vox-O.txt",
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
            "AUDIO_DIR": ["/app/dataset/VoxCeleb1/vox1_dev_wav/wav", "/app/dataset/VoxCeleb1/vox1_test_wav/wav"],
            "AUDIO_DATALIST": "/app/dataset/VoxCeleb1/Vox-CA5.txt",
            "audio_suffix": ".wav",
        },
        "Vox-CA10": {
            "AUDIO_DIR": ["/app/dataset/VoxCeleb1/vox1_dev_wav/wav", "/app/dataset/VoxCeleb1/vox1_test_wav/wav"],
            "AUDIO_DATALIST": "/app/dataset/VoxCeleb1/Vox-CA10.txt",
            "audio_suffix": ".wav",
        },
        "Vox-CA15": {
            "AUDIO_DIR": ["/app/dataset/VoxCeleb1/vox1_dev_wav/wav", "/app/dataset/VoxCeleb1/vox1_test_wav/wav"],
            "AUDIO_DATALIST": "/app/dataset/VoxCeleb1/Vox-CA15.txt",
            "audio_suffix": ".wav",
        },
        "Vox-CA20": {
            "AUDIO_DIR": ["/app/dataset/VoxCeleb1/vox1_dev_wav/wav", "/app/dataset/VoxCeleb1/vox1_test_wav/wav"],
            "AUDIO_DATALIST": "/app/dataset/VoxCeleb1/Vox-CA20.txt",
            "audio_suffix": ".wav",
        },
        "Vox1-H.S": {
            "AUDIO_DIR": ["/app/dataset/VoxCeleb1/vox1_dev_wav/wav", "/app/dataset/VoxCeleb1/vox1_test_wav/wav"],
            "AUDIO_DATALIST": "/app/dataset/VoxCeleb1/Vox1_hard_sample.txt",
            "audio_suffix": ".wav",
        },
        "Vox1-S.S": {
            "AUDIO_DIR": ["/app/dataset/VoxCeleb1/vox1_dev_wav/wav", "/app/dataset/VoxCeleb1/vox1_test_wav/wav"],
            "AUDIO_DATALIST": "/app/dataset/VoxCeleb1/same_session.txt",
            "audio_suffix": ".wav",
        },
        "Vox-merge": {
            "AUDIO_DIR": ["/app/dataset/VoxCeleb1/vox1_dev_wav/wav", "/app/dataset/VoxCeleb1/vox1_test_wav/wav"],
            "AUDIO_DATALIST": "/app/dataset/VoxCeleb1/Vox1-merged.txt",
            "audio_suffix": ".wav",
        },
    },
    "VoxCeleb2": {
        "small": {
            "train": {
                "AUDIO_DIR": "/app/dataset/VoxCeleb2/vox2_small_gap",
                "AUDIO_META_DIR": "/app/dataset/VoxCeleb2/vox2_small_gap.csv",
            },
            "val": {
                "AUDIO_DIR": "/app/dataset/VoxCeleb2/wav",
                "AUDIO_META_DIR": "/app/dataset/VoxCeleb2/vox2_val_small_gap.csv",
            },
        },
        "medium": {
            "train": {
                "AUDIO_DIR": "/app/dataset/VoxCeleb2/vox2_medium_gap",
                "AUDIO_META_DIR": "/app/dataset/VoxCeleb2/vox2_medium_gap.csv",
            },
            "val": {
                "AUDIO_DIR": "/app/dataset/VoxCeleb2/wav",
                "AUDIO_META_DIR": "/app/dataset/VoxCeleb2/vox2_val_medium_gap.csv",
            },
        },
        "large": {
            "train": {
                "AUDIO_DIR": "/app/dataset/VoxCeleb2/vox2_large_gap",
                "AUDIO_META_DIR": "/app/dataset/VoxCeleb2/vox2_large_gap.csv",
            },
            "val": {
                "AUDIO_DIR": "/app/dataset/VoxCeleb2/wav",
                "AUDIO_META_DIR": "/app/dataset/VoxCeleb2/vox2_val_large_gap.csv",
            },
        },
        "LE5": {
            "AUDIO_DIR": "/app/dataset/VoxCeleb2/Vox2_delta_LE5",
            "AUDIO_META_DIR": "/app/dataset/VoxCeleb2/Vox2_delta_LE5.csv",
        },
        "Delta5": {
            "AUDIO_DIR": "/app/dataset/VoxCeleb2/Vox2_delta5",
            "AUDIO_META_DIR": "/app/dataset/VoxCeleb2/Vox2_delta5_utts.csv",
        },
        "Delta20": {
            "AUDIO_DIR": "/app/dataset/VoxCeleb2/Vox2_delta20",
            "AUDIO_META_DIR": "/app/dataset/VoxCeleb2/Vox2_delta20_utts.csv",
        },
        "Mixture": {
            "AUDIO_DIR": "/app/dataset/VoxCeleb2/Vox2_mixture",
            "AUDIO_META_DIR": "/app/dataset/VoxCeleb2/Vox2_mixture.csv",
        },
        "Mini": {
            "AUDIO_DIR": "/app/dataset/VoxCeleb2/Vox2_mini",
            "AUDIO_META_DIR": "/app/dataset/VoxCeleb2/Vox2_mini.csv",
        },
        "audio_suffix": ".wav",
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
        "AUDIO_DIR": "/app/dataset/musan/musan",
    },
    "RIR": {
        "AUDIO_DIR": "/app/dataset/sim_rir_16k/simulated_rirs_16k",
    },
}