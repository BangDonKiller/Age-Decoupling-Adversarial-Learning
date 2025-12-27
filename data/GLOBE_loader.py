import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import io

class InferenceDataset(Dataset):
    """只負責讀取音檔並 preprocess 到 16kHz 單聲道張量"""

    def __init__(self, meta_dir: str, target_sample_rate: int = 16000, suffix: str = ".wav"):
        self.meta_dir = meta_dir
        self.target_sr = target_sample_rate
        self.datalist = self.read_meta_file(meta_dir)

    def __len__(self):
        return len(self.datalist)
    
    def read_meta_file(self, meta_path, max_items=1000):
        datalist = []
        sexual = {"male": 0, "female": 0}

        seen_speakers = set()

        for file in meta_path:
            df = pd.read_parquet(file)
            for _, row in df.iterrows():
                audio = row["audio"]['bytes']
                speaker_id = row["speaker_id"]
                gender = row["gender"]

                # 如果說話者有重複，就跳過
                if speaker_id not in seen_speakers:
                    # 男性與女性資料各500筆
                    if gender in sexual and sexual[gender] < (max_items / 2):
                        datalist.append((audio, speaker_id, gender))
                        sexual[gender] += 1
                        seen_speakers.add(speaker_id) # 將新的 speaker_id 加入 set

                # 檢查是否已滿，如果滿了就直接 return
                if len(datalist) >= max_items:
                    print(f"資料列表已滿 {max_items} 筆，提前結束。")
                    return datalist # <--- 直接返回結果，終止所有迴圈

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
        path, speaker_id, gender = self.datalist[idx]
        waveform = self._load_and_preprocess_audio_from_bytes(path)
        return waveform, speaker_id, gender

# if __name__ == "__main__":
#     # 測試 TrainDataset
#     from pathlib import Path

#     train_path = list(Path("D:\\Dataset\\GLOBE\\data").rglob("train-*.parquet"))
#     val_path = list(Path("D:\\Dataset\\GLOBE\\data").rglob("val-*.parquet"))
#     test_path = list(Path("D:\\Dataset\\GLOBE\\data").rglob("test-*.parquet"))

#     dataset = InferenceDataset(meta_dir=test_path)
    
#     dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
#     for i, (waveforms, speaker_ids, genders) in enumerate(dataloader):
#         print(f"Batch {i}:")
#         print(f"  Waveforms shape: {waveforms.shape}")
#         print(f"  Speaker IDs: {speaker_ids}")
#         print(f"  Genders: {genders}")
#         if i == 2:  # 只看前三個 batch
#             break