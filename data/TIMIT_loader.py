import random
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

class InferenceDataset(Dataset):
    """只負責讀取音檔並 preprocess 到 16kHz 單聲道張量"""

    def __init__(self, audio_dir: str, audio_meta_dir: str, target_sample_rate: int = 16000, suffix: str = ".wav"):
        self.audio_dir = audio_dir
        self.audio_meta_dir = Path(audio_meta_dir)
        self.target_sr = target_sample_rate
        
        self.conv_age = {
            range(0, 11): 0,
            range(11, 21): 1,
            range(21, 31): 2,
            range(31, 41): 3,
            range(41, 51): 4,
            range(51, 61): 5,
            range(61, 200): 6,
        }
        
        self.datalist = self.read_meta_file(self.audio_meta_dir)

    def __len__(self):
        return len(self.datalist)
    
    def read_meta_file(self, meta_path: Path) -> pd.DataFrame:
        """讀取 meta 檔案"""
        datalist = []
        
        meta = pd.read_csv(meta_path, sep=",")
        # 去除重複的行，去除有nan的行
        meta = meta.dropna()
        meta = meta.drop_duplicates().reset_index(drop=True)
        print(f"Total valid entries in meta file: {len(meta)}")
        
        for idx, row in meta.iterrows():
            speaker_id = row["speaker_id"]
            env = row["dialect_region"]
            gender = row["gender"]
            age = row["age"]
            
            # 年齡分群
            age = int(age)
            for age_range, age_group in self.conv_age.items():
                if age in age_range:
                    age = age_group
                    break
            
            path = Path(self.audio_dir) / env / speaker_id
            audiolist = [p for p in path.rglob("*") if p.suffix == ".wav"]

            for audio_path in audiolist:
                datalist.append((audio_path, speaker_id, gender, age))
        
        return datalist
    
    def _load_and_preprocess_audio(self, file_path: str) -> torch.Tensor:
        """load + resample + mono"""
        signal, fs = torchaudio.load(file_path)
        
        # 切成三秒長的音訊
        if signal.shape[1] > fs * 3:
            signal = signal[:, : fs * 3]
        else:
            padding = fs * 3 - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, padding))

        # Resample
        if fs != self.target_sr:
            signal = T.Resample(orig_freq=fs, new_freq=self.target_sr)(signal)

        # Mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        return signal
    
    def __getitem__(self, index: int):
        path, speaker_id, gender, age = self.datalist[index]
        waveform = self._load_and_preprocess_audio(str(path))
        return waveform, speaker_id, gender, age
    
if __name__ == "__main__":
    dataset = InferenceDataset(
        audio_dir="D:\\Dataset\\TIMIT\\data\\TRAIN",
        audio_meta_dir="D:\\Dataset\\TIMIT\\train_meta_data.csv",
    )
    print(f"Dataset length: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # print the first two batches
    for i, (waveforms, speaker_ids, genders, ages) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"Waveforms shape: {waveforms.shape}")
        print(f"Speaker IDs: {speaker_ids}")
        print(f"Genders: {genders}")
        print(f"Ages: {ages}")
        if i == 1:
            break