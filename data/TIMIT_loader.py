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

class PairwiseDataset(Dataset):
    """讀取成對的音檔並 preprocess 到 16kHz 單聲道張量"""

    def __init__(self, audio_dir: str, audio_meta_dir: str, target_sample_rate: int = 16000, suffix: str = ".wav"):
        self.audio_dir = audio_dir
        self.audio_meta_dir = Path(audio_meta_dir)
        self.target_sr = target_sample_rate
        self.audio_list = self.read_meta_file(self.audio_meta_dir)
        self.datalist = self.create_pairwise_indices()
        
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
            
            path = Path(self.audio_dir) / env / speaker_id
            audiolist = [p for p in path.rglob("*") if p.suffix == ".wav"]

            for audio_path in audiolist:
                datalist.append((audio_path, speaker_id))
                
        return datalist
    

    def create_pairwise_indices(self, same_per_speaker=10, diff_per_speaker=10):
        """創建成對的音檔索引 (positive/negative pairs)"""
        pairs = []

        # Group audio files by speaker
        speaker_dict = {}
        for path, speaker_id in self.audio_list:
            speaker_dict.setdefault(speaker_id, []).append(path)

        speakers = list(speaker_dict.keys())

        # ---- Generate same speaker pairs ---- #
        for spk in speakers:
            files = speaker_dict[spk]
            # 如果該說話者檔案太少，沒有足夠配對，就跳過
            if len(files) < 2:
                continue

            # 取得可用的所有 pair combinations
            all_combos = [(a, b) for i, a in enumerate(files) for b in files[i+1:]]
            random.shuffle(all_combos)

            # 取指定數量
            count = min(same_per_speaker, len(all_combos))
            for a, b in all_combos[:count]:
                pairs.append((1, a, b))  # 1 表示同一個 speaker

        # ---- Generate different speaker pairs ---- #
        for spk in speakers:
            files = speaker_dict[spk]
            if len(files) == 0:
                continue

            # pick negative speaker ids
            other_speakers = [s for s in speakers if s != spk]
            for _ in range(diff_per_speaker):
                # pick one file from this speaker
                a = random.choice(files)
                # pick a different speaker
                other = random.choice(other_speakers)
                b = random.choice(speaker_dict[other])
                pairs.append((0, a, b))  # 0 表示不同 speaker
                
        # Shuffle all pairs
        random.shuffle(pairs)
        
        # 計算正對數量與負對數量
        pos_count = sum(1 for label, _, _ in pairs if label == 1)
        neg_count = sum(1 for label, _, _ in pairs if label == 0)
        print(f"總配對數量: {len(pairs)} (正對: {pos_count}, 負對: {neg_count})")
        
        return pairs
    
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
        is_same, spk1, spk2 = self.datalist[index]
        waveform1 = self._load_and_preprocess_audio(str(spk1))
        waveform2 = self._load_and_preprocess_audio(str(spk2))
        return is_same, waveform1, waveform2

class InferenceDataset(Dataset):
    """只負責讀取音檔並 preprocess 到 16kHz 單聲道張量"""

    def __init__(self, audio_dir: str, audio_meta_dir: str, target_sample_rate: int = 16000, suffix: str = ".wav"):
        self.audio_dir = audio_dir
        self.audio_meta_dir = Path(audio_meta_dir)
        self.target_sr = target_sample_rate
        
        self.conv_age = {
            range(0, 21): 0,
            range(21, 31): 1,
            range(31, 41): 2,
            range(41, 51): 3,
            range(51, 61): 4,
            range(61, 71): 5,
            range(71, 81): 6,
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
            converted_age = self.conv_age.get(next((r for r in self.conv_age if age in r), None), -1)
            if converted_age == -1:
                print(f"Unknown age group: {age} for speaker {speaker_id}")
                    
            path = Path(self.audio_dir) / env / speaker_id
            audiolist = [p for p in path.rglob("*") if p.suffix == ".wav"]

            for audio_path in audiolist:
                datalist.append((audio_path, speaker_id, gender, converted_age))
                
        # count the speaker, gender, age group distribution
        # speaker_set = set()
        # gender_count = {}
        # age_group_count = {}
        
        # for _, speaker_id, gender, age in datalist:
        #     speaker_set.add(speaker_id)
        #     gender_count[gender] = gender_count.get(gender, 0) + 1
        #     age_group_count[age] = age_group_count.get(age, 0) + 1
            
        # print(f"Total speakers: {len(speaker_set)}")
        # print(f"Gender distribution: {gender_count}")
        # print(f"Age group distribution: {age_group_count}")
        
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
    dataset = PairwiseDataset(
        audio_dir="D:\\Dataset\\TIMIT\\data\\TRAIN",
        audio_meta_dir="D:\\Dataset\\TIMIT\\train_meta_data.csv",
    )
    print(f"Dataset length: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # print the first two batches
    for i, (is_same, spk1, spk2) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"  is_same: {is_same}")
        print(f"  spk1 shape: {spk1.shape}")
        print(f"  spk2 shape: {spk2.shape}")
        if i == 1:
            break