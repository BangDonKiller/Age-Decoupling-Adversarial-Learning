import random
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import io

random.seed(42)

class PairwiseDataset(Dataset):
    """讀取成對的音檔並 preprocess 到 16kHz 單聲道張量"""

    def __init__(self, meta_dir: str, target_sample_rate: int = 16000, suffix: str = ".wav"):
        self.meta_dir = meta_dir
        self.target_sr = target_sample_rate
        self.audiolist = self.read_meta_file(meta_dir)
        self.datalist = self.create_pairwise_indices(self.audiolist)
        
    def __len__(self):
        return len(self.datalist)
    
    def read_meta_file(self, meta_path, speakers_limit=200):
        datalist = []

        seen_speakers = set()

        for file in meta_path:
            df = pd.read_parquet(file)
            for _, row in df.iterrows():
                audio = row["audio"]['bytes']
                speaker_id = row["speaker_id"]
                
                if len(seen_speakers) < speakers_limit or speaker_id in seen_speakers:
                    datalist.append((audio, speaker_id))
                    seen_speakers.add(speaker_id)
        
        # 計算set裡面的說話者語句最少的有幾句
        speaker_count = {}
        for _, speaker_id in datalist:
            if speaker_id not in speaker_count:
                speaker_count[speaker_id] = 0
            speaker_count[speaker_id] += 1
            
        min_count = min(speaker_count.values())
        print(f"總共讀取 {len(datalist)} 筆音訊資料，包含 {len(speaker_count)} 位說話者。")
        print(f"每位說話者最少有 {min_count} 筆語句資料。")
        
        print("所有檔案處理完畢。")
        return datalist

    def create_pairwise_indices(self, datalist, target_pos=5000, target_neg=5000):
        speaker_dict = {}
        for audio, spk in datalist:
            speaker_dict.setdefault(spk, []).append(audio)

        valid_spks = [s for s, files in speaker_dict.items() if len(files) >= 2]
        all_spks = list(speaker_dict.keys())

        pairs_set = set()
        pairs = []

        # ---- positive sampling ---- #
        while len([p for p in pairs if p[0]==1]) < target_pos:
            spk = random.choice(valid_spks)
            files = speaker_dict[spk]
            a, b = random.sample(files, 2)

            # 保持排序 (canonical) 來避免 (A,B) vs (B,A)
            pair_key = (min(a, b), max(a, b))
            if pair_key not in pairs_set:
                pairs_set.add(pair_key)
                pairs.append((1, pair_key[0], pair_key[1]))

        # ---- negative sampling ---- #
        while len([p for p in pairs if p[0]==0]) < target_neg:
            spk1, spk2 = random.sample(all_spks, 2)
            a = random.choice(speaker_dict[spk1])
            b = random.choice(speaker_dict[spk2])

            pair_key = (min(a, b), max(a, b))
            if pair_key not in pairs_set:
                pairs_set.add(pair_key)
                pairs.append((0, pair_key[0], pair_key[1]))

        random.shuffle(pairs)
        
        pos = sum(1 for label, _, _ in pairs if label == 1)
        neg = sum(1 for label, _, _ in pairs if label == 0)
        
        print(f"總配對數量: {len(pairs)} (正對: {pos}, 負對: {neg})")
        return pairs
    
    def _load_and_preprocess_audio_from_bytes(self, bytes_data: bytes) -> torch.Tensor:
        """
        從記憶體中的 bytes 物件載入音訊，並進行重採樣和單聲道處理。
        """
        # 1. 將 bytes 資料包裝成一個記憶體中的檔案串流
        bytes_stream = io.BytesIO(bytes_data)
        
        # 2. torchaudio.load() 現在從這個串流中讀取，而不是從檔案路徑
        signal, fs = torchaudio.load(bytes_stream)
        
        # 切成三秒長的音訊
        target_length = fs * 3
        if signal.shape[1] > target_length:
            signal = signal[:, :target_length]
        else:
            padding = target_length - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, padding))

        # Resample
        if fs != self.target_sr:
            # 確保 T.Resample 被正確初始化
            resampler = T.Resample(orig_freq=fs, new_freq=self.target_sr)
            signal = resampler(signal)

        # Mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        return signal

    def __getitem__(self, idx):
        is_same, spk1, spk2 = self.datalist[idx]
        waveform1 = self._load_and_preprocess_audio_from_bytes(spk1)
        waveform2 = self._load_and_preprocess_audio_from_bytes(spk2)
        return is_same, waveform1, waveform2
    
    

class InferenceDataset(Dataset):
    """只負責讀取音檔並 preprocess 到 16kHz 單聲道張量"""

    def __init__(self, meta_dir: str, target_sample_rate: int = 16000, suffix: str = ".wav"):
        self.meta_dir = meta_dir
        self.target_sr = target_sample_rate
        self.conv_age = {
            'teens': 0,
            'twenties': 1,
            'thirties': 2,
            'fourties': 3,
            'fifties': 4,
            'sixties': 5,
            'seventies': 6,
        }
        self.datalist = self.read_meta_file(meta_dir)
        

    def __len__(self):
        return len(self.datalist)
    
    def read_meta_file(self, meta_path, max_items=20000):
        datalist = []
        sexual = {"male": 0, "female": 0}

        seen_speakers = set()

        for file in meta_path:
            df = pd.read_parquet(file)
            for _, row in df.iterrows():
                audio = row["audio"]['bytes']
                speaker_id = row["speaker_id"]
                gender = row["gender"]
                age = row["age"]
                
                converted_age = self.conv_age[age] if age in self.conv_age else -1
                if converted_age == -1:
                    print(f"Unknown age group: {age} for speaker {speaker_id}")

                # # 如果說話者有重複，就跳過
                # if speaker_id not in seen_speakers:
                # 男性與女性資料各10000筆
                if gender in sexual and sexual[gender] < (max_items / 2):
                    datalist.append((audio, speaker_id, gender, converted_age))
                    sexual[gender] += 1
                    seen_speakers.add(speaker_id) # 將新的 speaker_id 加入 set

                # 檢查是否已滿，如果滿了就直接 return
                if len(datalist) >= max_items:
                    print(f"資料列表已滿 {max_items} 筆，提前結束。")
                    return datalist # <--- 直接返回結果，終止所有迴圈

        # count the age amount
        # age_count = {}
        # for item in datalist:
        #     age = item[3]
        #     if age not in age_count:
        #         age_count[age] = 0
        #     age_count[age] += 1
        # print("年齡分佈：", age_count)
        
        print("所有檔案處理完畢。")
        return datalist
        
    def _load_and_preprocess_audio_from_bytes(self, bytes_data: bytes) -> torch.Tensor:
        """
        從記憶體中的 bytes 物件載入音訊，並進行重採樣和單聲道處理。
        """
        # 1. 將 bytes 資料包裝成一個記憶體中的檔案串流
        bytes_stream = io.BytesIO(bytes_data)
        
        # 2. torchaudio.load() 現在從這個串流中讀取，而不是從檔案路徑
        signal, fs = torchaudio.load(bytes_stream)
        
        # 切成三秒長的音訊
        target_length = fs * 3
        if signal.shape[1] > target_length:
            signal = signal[:, :target_length]
        else:
            padding = target_length - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, padding))

        # Resample
        if fs != self.target_sr:
            # 確保 T.Resample 被正確初始化
            resampler = T.Resample(orig_freq=fs, new_freq=self.target_sr)
            signal = resampler(signal)

        # Mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        return signal

    def __getitem__(self, idx):
        path, speaker_id, gender, age = self.datalist[idx]
        waveform = self._load_and_preprocess_audio_from_bytes(path)
        return waveform, speaker_id, gender, age

if __name__ == "__main__":
    train_path = list(Path("D:\\Dataset\\GLOBE\\data").rglob("train-*.parquet"))
    val_path = list(Path("D:\\Dataset\\GLOBE\\data").rglob("val-*.parquet"))
    test_path = list(Path("D:\\Dataset\\GLOBE\\data").rglob("test-*.parquet"))

    dataset = PairwiseDataset(meta_dir=train_path)
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for i, (is_same, spk1, spk2) in enumerate(dataloader):
        print(f"Batch {i}:")
        print("is_same:", is_same)
        print("spk1:", spk1.shape)
        print("spk2:", spk2.shape)
        if i == 2:  # 只看前三個 batch
            break