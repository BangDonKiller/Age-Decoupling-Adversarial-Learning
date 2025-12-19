import pandas as pd
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, file_dir, file_list):
        """
        file_dir: 根目錄 (其實現在 file_list 已經是完整路徑，這邊留著當備註)
        file_list: [(完整路徑, 講者編號), ...]
        """
        self.datalist = file_list

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        wav_path, spk_id = self.datalist[idx]

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
            return torch.zeros(48000), spk_id

        return signal, spk_id

class InferenceDataset(Dataset):
    """只負責讀取音檔並 preprocess 到 16kHz 單聲道張量"""

    def __init__(self, audio_dir: str, audio_list_dir: str, audio_meta_dir: str, target_sample_rate: int = 16000, suffix: str = ".wav"):
        self.audio_dir = Path(audio_dir)
        self.audio_list_dir = Path(audio_list_dir)
        self.audio_meta_dir = Path(audio_meta_dir)
        self.audio_list = []
        self.target_sr = target_sample_rate
        
        # # read the audio file list
        with open(self.audio_list_dir, "r") as f:
            for line in f:
                filepath = line.strip()
                if filepath.endswith(suffix):
                    self.audio_list.append(filepath)
                if len(self.audio_list) >= 10000:
                    break
        
        self.meta = self.read_meta_file(self.audio_meta_dir)
        self.datalist = self.get_audio_paths(self.audio_list)

    def __len__(self):
        return len(self.audio_list)
    
    def read_meta_file(self, meta_path: str):
        df = pd.read_csv(meta_path, encoding="latin1")

        # 先拆成 list
        df["splitted"] = (
            df.iloc[:, 0]
            .str.strip()
            .str.split("\t")
            .apply(lambda xs: [x.strip() for x in xs])
        )

        df["col1"] = df["splitted"].apply(lambda x: x[1] if len(x) > 1 else None)
        df["col3"] = df["splitted"].apply(lambda x: x[3] if len(x) > 3 else None)

        # 移除 key 或 value 是 None 的列（避免髒資料）
        df_valid = df.dropna(subset=["col1", "col3"])

        # 建立字典 col1 中的值當 key，col3 中的值當 value 
        meta_dict = dict(zip(df_valid["col1"], df_valid["col3"]))
        
        return meta_dict
    
    def get_audio_paths(self, audio_list):
        data_list = []
        for relative_path in audio_list:
            speaker_id = relative_path.split()[0]
            audio_path = relative_path.split()[1]
            full_path = self.audio_dir / audio_path
            gender = self.meta[speaker_id]            
            data_list.append((speaker_id, gender, full_path))
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
        speaker_id, gender, path = self.datalist[idx]
        waveform = self._load_and_preprocess_audio(str(path))
        return waveform, speaker_id, gender
