import pandas as pd
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import random

class InferenceDataset(Dataset):
    """只負責讀取音檔並 preprocess 到 16kHz 單聲道張量"""

    def __init__(self, audio_dir: str, audio_meta_dir: str, target_sample_rate: int = 16000, suffix: str = ".wav"):
        self.audio_dir = Path(audio_dir)
        self.audio_meta_dir = Path(audio_meta_dir)
        self.datalist = []
        self.target_sr = target_sample_rate
        
        self.datalist = self.read_meta_file(self.audio_meta_dir)
        
    def __len__(self):
        return len(self.datalist)
    
    def read_meta_file(self, meta_path: str):
        with open(meta_path, "r") as f:
            lines = f.readlines()
        
        audio_list = []
        
        for line in lines[12:]:  # 跳過標題行
            parts = line.strip().split("|")
            speaker_id = parts[0].strip()
            gender = parts[1].strip()
            subset = parts[2].strip()
            
            if subset in ["train-clean-100", "dev-clean", "test-clean", "train-other-500"]:       
                utts_dir = self.audio_dir / subset / "LibriSpeech" / subset / speaker_id
                utts = list(utts_dir.rglob(f"*{'.flac'}"))
                utt = random.sample(utts, min(10, len(utts)))  # 每個說話者最多取10個檔案
                
                for u in utt:
                    audio_list.append((str(u), speaker_id, gender))
                
        # 看性別的數量
        gender_counts = {"M": 0, "F": 0}
        for _, _, gender in audio_list:
            if gender in gender_counts:
                gender_counts[gender] += 1
        print(f"Gender counts: {gender_counts}")
                
        return audio_list
            
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
        audio_path, speaker_id, gender = self.datalist[idx]
        waveform = self._load_and_preprocess_audio(audio_path)
        return waveform, speaker_id, gender