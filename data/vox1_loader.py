import random
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, file_dir, meta_dir):
        """
        file_dir: voxceleb1 根目錄
        meta_dir: voxceleb1_meta.csv 路徑
        依據 meta 建立說話者到語音檔案的映射(目前只取 dev 中的資料)
        """
        self.file_dir = file_dir
        self.meta = self.read_meta(meta_dir)
        self.spk2utts = self.speaker_utterance_dict()
        self.datalist = self.create_datalist()
        print(f"Dataset 初始化完成，總共有 {len(self.datalist)} 筆資料。")

    def __len__(self):
        return len(self.datalist)
    
    def read_meta(self, meta_dir):
        df = pd.read_csv(meta_dir, encoding="latin1")

        df["splitted"] = (
            df.iloc[:, 0]
            .str.strip()
            .str.split("\t")
            .apply(lambda xs: [x.strip() for x in xs])
        )

        df["col0"] = df["splitted"].apply(lambda x: x[0] if len(x) > 0 else None)
        df["col4"] = df["splitted"].apply(lambda x: x[4] if len(x) > 4 else None)

        # 移除 None
        df_valid = df.dropna(subset=["col0", "col4"])

        # ⭐ 只保留 dev
        df_dev = df_valid[df_valid["col4"] == "dev"]

        meta_dict = dict(zip(df_dev["col0"], df_dev["col4"]))
        return meta_dict
    
    def speaker_utterance_dict(self):
        spk2utts = {}
        # 讀取 self.meta 建立說話者到語音檔案的映射
        for spk, _ in self.meta.items():
            if spk not in spk2utts:
                speaker_audio_path = self.file_dir / spk
                # 遍歷所有音訊檔名，建立列表，使用絕對路徑找出wav檔案
                utts = [
                    str(p) for p in speaker_audio_path.rglob("*.wav")
                ]
                spk2utts[spk] = utts
        return spk2utts
    

    def create_datalist(self, pairs_per_speaker=100, seed=42):
        random.seed(seed)

        speakers = list(self.spk2utts.keys())
        datalist = []

        for spk in speakers:
            utts = self.spk2utts[spk]

            # 同一說話者至少要有 2 個 utterance 才能做正樣本
            if len(utts) < 2:
                continue

            # ===== 正樣本：同一說話者 =====
            for _ in range(pairs_per_speaker):
                utt1, utt2 = random.sample(utts, 2)
                datalist.append((1, utt1, utt2))

            # ===== 負樣本：不同說話者（不包含自己） =====
            for _ in range(pairs_per_speaker):
                other_spk = random.choice(speakers)
                while other_spk == spk:
                    other_spk = random.choice(speakers)
                other_utts = self.spk2utts[other_spk]

                if len(other_utts) == 0:
                    continue

                utt1 = random.choice(utts)
                utt2 = random.choice(other_utts)

                label = 1 if other_spk == spk else 0
                datalist.append((label, utt1, utt2))

        random.shuffle(datalist)
        return datalist
    
    def _audio_processing(self, wav_path):
        # 讀取音檔
        try:
            signal, sr = torchaudio.load(wav_path)
            
            # 固定長度處理（例如 3 秒），這對 Batch 訓練非常重要
            # 如果不固定長度，DataLoader 在 collect 時會報錯 (因為 tensor size 不一)
            target_length = 16000 * 3 
            if signal.shape[1] > target_length:
                signal = signal[:, :target_length]
            else:
                padding = target_length - signal.shape[1]
                signal = torch.nn.functional.pad(signal, (0, padding))

            # 重採樣
            if sr != 16000:
                resampler = T.Resample(orig_freq=sr, new_freq=16000)
                signal = resampler(signal)
            
            # Mono
            if signal.shape[0] > 1:
                signal = signal.mean(dim=0, keepdim=True)

            signal = signal.squeeze(0) # (samples,)
        except Exception as e:
            # 預防損壞的檔案
            print(f"Error loading {wav_path}: {e}")
            return torch.zeros(48000)
        
        return signal

    def __getitem__(self, idx):
        is_same_speaker, spk1_audio_path, spk2_audio_path = self.datalist[idx]

        signal1 = self._audio_processing(spk1_audio_path)
        signal2 = self._audio_processing(spk2_audio_path)

        return torch.tensor(is_same_speaker, dtype=torch.float32), signal1, signal2