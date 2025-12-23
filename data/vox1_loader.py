import random
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from collections import Counter
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

class TrainDataset(Dataset):
    def __init__(self, file_dir, meta_dir, pairs_per_speaker=5, seed=42):
        """
        file_dir: voxceleb1 根目錄
        meta_dir: voxceleb1_meta.csv 路徑
        依據 meta 建立說話者到語音檔案的映射(目前只取 dev 中的資料)
        """
        self.file_dir = file_dir
        self.meta = self.read_meta(meta_dir)
        self.spk2utts = self.speaker_utterance_dict()
        self.all_speakers = list(self.spk2utts.keys())
        
        # 在初始化時，為所有說話者創建一個完整的 datalist
        # split_train_valid 函數將會重新生成專用的 datalist
        self.datalist = self._generate_pairs_for_speakers(
            self.all_speakers, self.all_speakers, pairs_per_speaker, seed
        )
        print(f"Dataset 初始化完成，總共有 {len(self.datalist)} 筆資料。")

    def __len__(self):
        return len(self.datalist)
    
    def read_meta(self, meta_dir):
        df = pd.read_csv(meta_dir, encoding="latin1")
        df["splitted"] = df.iloc[:, 0].str.strip().str.split("\t").apply(lambda xs: [x.strip() for x in xs])
        df["col0"] = df["splitted"].apply(lambda x: x[0] if len(x) > 0 else None)
        df["col4"] = df["splitted"].apply(lambda x: x[4] if len(x) > 4 else None)
        df_valid = df.dropna(subset=["col0", "col4"])
        df_dev = df_valid[df_valid["col4"] == "dev"]
        return dict(zip(df_dev["col0"], df_dev["col4"]))
    
    def speaker_utterance_dict(self):
        spk2utts = {}
        for spk, _ in self.meta.items():
            if spk not in spk2utts:
                speaker_audio_path = self.file_dir / spk
                utts = [str(p) for p in speaker_audio_path.rglob("*.wav")]
                if len(utts) > 0: # 確保說話者有音檔
                    spk2utts[spk] = utts
        return spk2utts
    
    def _generate_pairs_for_speakers(self, target_speakers, pool_speakers, pairs_per_speaker, seed):
        """
        輔助函數：為指定的說話者列表生成音訊對。
        - target_speakers: 需要為其生成配對的主要說話者列表。
        - pool_speakers:   生成負樣本時，從這個池子裡挑選其他說話者。
        - pairs_per_speaker: 每個說話者生成的正/負樣本對數量。
        - seed:            隨機種子。
        """
        random.seed(seed)
        datalist = []
        
        for spk in target_speakers:
            utts = self.spk2utts.get(spk, [])

            if len(utts) < 2:
                continue

            # ===== 正樣本：同一說話者 =====
            for _ in range(pairs_per_speaker):
                utt1, utt2 = random.sample(utts, 2)
                datalist.append((1, utt1, utt2))

            # ===== 負樣本：不同說話者（從 pool_speakers 中挑選） =====
            for _ in range(pairs_per_speaker):
                other_spk = random.choice(pool_speakers)
                # 確保挑到的不是自己
                while other_spk == spk:
                    other_spk = random.choice(pool_speakers)
                
                other_utts = self.spk2utts.get(other_spk, [])
                if not other_utts:
                    continue

                utt1 = random.choice(utts)
                utt2 = random.choice(other_utts)
                datalist.append((0, utt1, utt2))
        
        random.shuffle(datalist)
        return datalist

    def split_train_valid(self, valid_ratio=0.1, pairs_per_speaker=5, seed=42):
        """
        【新版】以 Speaker-Disjoint 的方式分割數據集。
        """
        print("\n進行 speaker-disjoint 資料分割...")
        
        # 1. 分割說話者列表
        train_speakers, valid_speakers = train_test_split(
            self.all_speakers,
            test_size=valid_ratio,
            random_state=seed,
            shuffle=True
        )
        print(f"總說話者: {len(self.all_speakers)}")
        print(f"訓練集說話者數量: {len(train_speakers)}")
        print(f"驗證集說話者數量: {len(valid_speakers)}")

        # 2. 【驗證】確認說話者沒有重複
        train_spk_set = set(train_speakers)
        valid_spk_set = set(valid_speakers)
        intersection = train_spk_set.intersection(valid_spk_set)
        
        print(f"訓練集與驗證集的說話者交集數量: {len(intersection)}")
        assert len(intersection) == 0, "錯誤：訓練集和驗證集之間存在重疊的說話者！"
        print("✅ 說話者分割驗證成功，無重疊。")

        # 3. 為訓練集和驗證集分別生成數據對
        # 訓練集的負樣本池是訓練集本身
        train_datalist = self._generate_pairs_for_speakers(
            train_speakers, train_speakers, pairs_per_speaker, seed
        )
        # 驗證集的負樣本池是驗證集本身
        valid_datalist = self._generate_pairs_for_speakers(
            valid_speakers, valid_speakers, pairs_per_speaker, seed
        )
        
        # 為了能使用 Subset，我們將兩個 datalist 合併，並記住各自的索引範圍
        self.datalist = train_datalist + valid_datalist
        train_indices = list(range(len(train_datalist)))
        valid_indices = list(range(len(train_datalist), len(self.datalist)))
        
        print(f"生成訓練樣本數: {len(train_datalist)}")
        print(f"生成驗證樣本數: {len(valid_datalist)}")

        train_subset = Subset(self, train_indices)
        valid_subset = Subset(self, valid_indices)
        
        return train_subset, valid_subset
    
    def _audio_processing(self, wav_path):
        try:
            signal, sr = torchaudio.load(wav_path)
            target_length = 16000 * 3 
            if signal.shape[1] > target_length:
                signal = signal[:, :target_length]
            else:
                padding = target_length - signal.shape[1]
                signal = torch.nn.functional.pad(signal, (0, padding))
            if sr != 16000:
                resampler = T.Resample(orig_freq=sr, new_freq=16000)
                signal = resampler(signal)
            if signal.shape[0] > 1:
                signal = signal.mean(dim=0, keepdim=True)
            signal = signal.squeeze(0)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            return torch.zeros(48000)
        return signal

    def __getitem__(self, idx):
        is_same_speaker, spk1_audio_path, spk2_audio_path = self.datalist[idx]
        signal1 = self._audio_processing(spk1_audio_path)
        signal2 = self._audio_processing(spk2_audio_path)
        return torch.tensor(is_same_speaker, dtype=torch.float32), signal1, signal2



class InferenceDataset(Dataset):
    """只負責讀取音檔並 preprocess 到 16kHz 單聲道張量"""

    def __init__(self, audio_dir: str, audio_meta_dir: str, target_sample_rate: int = 16000, suffix: str = ".wav"):
        self.audio_dir = Path(audio_dir)
        self.audio_meta_dir = Path(audio_meta_dir)
        self.audio_list = []
        self.target_sr = target_sample_rate
        
        self.meta = self.read_meta_file(self.audio_meta_dir)
        self.datalist = self.get_audio_paths(self.meta)

    def __len__(self):
        return len(self.datalist)
    
    def read_meta_file(self, meta_path: str):
        df = pd.read_csv(meta_path, encoding="latin1")
        df["splitted"] = df.iloc[:, 0].str.strip().str.split("\t").apply(lambda xs: [x.strip() for x in xs])
        df["col0"] = df["splitted"].apply(lambda x: x[0] if len(x) > 0 else None)
        df["col2"] = df["splitted"].apply(lambda x: x[2] if len(x) > 2 else None)
        df["col4"] = df["splitted"].apply(lambda x: x[4] if len(x) > 4 else None)
        df_valid = df.dropna(subset=["col0", "col2", "col4"])
        df_dev = df_valid[df_valid["col4"] == "dev"]
        return dict(zip(df_dev["col0"], df_dev["col2"]))
    
    def get_audio_paths(self, audio_list):
        data_list = []
        for speaker_id, gender in audio_list.items():
            speaker_audio_path = self.audio_dir / speaker_id
            utts = [str(p) for p in speaker_audio_path.rglob("*.wav")]
            utt = random.choice(utts)
            data_list.append((Path(utt), speaker_id, gender))
        return data_list
        
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

    def __getitem__(self, idx):
        path, speaker_id, gender = self.datalist[idx]
        waveform = self._load_and_preprocess_audio(str(path))
        return waveform, speaker_id, gender

# if __name__ == "__main__":
#     # 測試 TrainDataset
#     from pathlib import Path

#     file_dir = Path("D:\\Dataset\\VoxCeleb1\\vox1_dev_wav\\wav")  # 替換為實際路徑
#     meta_dir = Path("D:\\Dataset\\VoxCeleb1\\vox1_meta.csv")  # 替換為實際路徑

#     inference_dataset = InferenceDataset(audio_dir=file_dir, audio_meta_dir=meta_dir)
#     print(f"InferenceDataset 長度: {len(inference_dataset)}")

#     from torch.utils.data import DataLoader
#     inference_dataloader = DataLoader(inference_dataset, batch_size=4, shuffle=True)

#     # 印出前三個
#     for i, (waveforms, speaker_ids, genders) in enumerate(inference_dataloader):
#         print(f"Batch {i+1}:")
#         print(f"Waveforms shape: {waveforms.shape}")
#         print(f"Speaker IDs: {speaker_ids}")
#         if i == 2:
#             break