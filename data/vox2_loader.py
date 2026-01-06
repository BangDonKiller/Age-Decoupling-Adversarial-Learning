import pandas as pd
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import random

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

    def __init__(self, audio_dir: str, audio_meta_dir: str, target_sample_rate: int = 16000, suffix: str = ".wav"):
        self.audio_dir = Path(audio_dir)
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
        
        self.meta = self.read_meta_file(self.audio_meta_dir)
        self.datalist = self.get_audio_paths()

    def __len__(self):
        return len(self.datalist)
    
    def read_meta_file(self, meta_path: str):
        """
        讀取 metadata 檔案，回傳 dict 結構如下：
        {
            "speaker_id_1": {
                "gender": "M",
                "utts": {   
                    "utterance_1": {"age": 2},
                    "utterance_2": {"age": 3},
                    ...
                }
            },
            "speaker_id_2": {
                "gender": "F",
                "utts": {
                    "utterance_1": {"age": 4},
                    "utterance_2": {"age": 5},
                    ...
                }
            }
        }
        """
        df = pd.read_csv(
            meta_path,
            sep=",",
            header=0,  # 表示第一列是 header，要跳過
            usecols=[0,1,2,3],  # 只抓這四欄，避免多餘欄位
            dtype={"age": str}   # 先讀成字串，後續再轉數字
        )

        meta_dict = {}

        for _, row in df.iterrows():
            speaker = row["speaker_id"] if "speaker_id" in df.columns else row.iloc[0]
            utt = row["utterance"] if "utterance" in df.columns else row.iloc[1]
            # turn age into age group
            age_str = row["age"] if "age" in df.columns else row.iloc[2]
            
            # 年齡分群
            age = int(age_str)
            converted_age = self.conv_age.get(next((r for r in self.conv_age if age in r), None), -1)
            if converted_age == -1:
                print(f"Unknown age group: {age} for speaker {speaker}")
            
            gender = row["gender"] if "gender" in df.columns else row.iloc[3]

            # speaker 第一次出現
            if speaker not in meta_dict:
                meta_dict[speaker] = {
                    "gender": gender,
                    "utts": {}
                }

            # 同一 speaker 底下加入不同 utterance
            meta_dict[speaker]["utts"][utt] = {
                "age": converted_age
            }

        return meta_dict
    
    def get_audio_paths(self, num_utts_per_speaker=10):
        data_list = []

        for speaker_id, info in self.meta.items():
            gender = info["gender"]
            utts = list(info["utts"].keys())

            # 該 speaker 的 utterance 不足 10 個 → 全拿
            sampled_utts = random.sample(
                utts,
                k=min(num_utts_per_speaker, len(utts))
            )

            for utt in sampled_utts:
                audio_folder = Path(self.audio_dir) / speaker_id / f"{utt}"
                audio_path = list(audio_folder.rglob(f"*.m4a"))
                random_select = random.sample(audio_path, k=1)[0]

                utt_info = info["utts"][utt]

                data_list.append((str(random_select),speaker_id,gender,utt_info["age"]))
                
        # count the speaker, gender, age group amount in datalist
        # speaker_count = len(set([item[1] for item in data_list]))
        # gender_count = {}
        # age_group_count = {}
        # for item in data_list:
        #     gender = item[2]
        #     age_group = item[3]
        #     gender_count[gender] = gender_count.get(gender, 0) + 1
        #     age_group_count[age_group] = age_group_count.get(age_group, 0) + 1
        # print(f"Total speakers: {speaker_count}")
        # print("Gender counts:", gender_count)
        # print("Age group counts:", age_group_count)

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
        path, speaker_id, gender, age = self.datalist[idx]
        waveform = self._load_and_preprocess_audio(str(path))
        return waveform, speaker_id, gender, age
    
if __name__ == "__main__":
    # 測試 Dataset
    audio_dir = "D:\\Dataset\\VoxCeleb2\\vox2_dev_wav\\dev\\aac"
    audio_meta_dir = "D:\\Dataset\\VoxCeleb2\\vox2_meta.csv"
    
    dataset = InferenceDataset(audio_dir, audio_meta_dir, suffix=".m4a")
    print(f"Dataset size: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        waveforms, speaker_ids, genders, ages = batch
        print(f"Waveforms shape: {waveforms.shape}")
        print(f"Speaker IDs: {speaker_ids}")
        print(f"Genders: {genders}")
        print(f"Ages: {ages}")
        
        