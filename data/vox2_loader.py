from numpy import random
import pandas as pd
from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class Vox2PairDataset(Dataset):
    """讀取音檔並 preprocess 到 16kHz 單聲道張量，並且每次回傳一對 speaker 的音檔與年齡差距"""
    def __init__(
        self,
        audio_dir: str,
        audio_meta_dir: str,
    ):
        self.audio_dir = Path(audio_dir)
        self.audio_meta_dir = Path(audio_meta_dir)
        self.datalist = []
        self.expert_mapping = {
            "small": 0,
            "medium": 1,
            "large": 2
        }

        self.datalist = self.read_meta_file(self.audio_meta_dir)
        
    def __len__(self):
        return len(self.datalist)
    
    def read_meta_file(self, meta_path: str):
        """
        column: 
            label: 0 or 1 (0: 同一說話者, 1: 不同說話者)
            speaker1_path: 說話者1的音檔路徑
            speaker2_path: 說話者2的音檔路徑
            age_diff: 說話者之間的年齡差距
            expert: 理想情況下，哪位專家應該處理這些數據？
        """
        df = pd.read_csv(
            meta_path,
            sep=",",
            header=0,  # 表示第一列是 header，要跳過
            dtype={"expert": str}   # 先讀成字串，後續再轉數字
        )
        
        for _, row in df.iterrows():
            label = int(row["label"])
            speaker1_path = Path(self.audio_dir) / Path(row["speaker1_path"])
            speaker2_path = Path(self.audio_dir) / Path(row["speaker2_path"])
            
            # 年齡差請四捨五入到第二位
            age_diff = round(float(row["age_diff"]), 2)
            expert = self.expert_mapping.get(row["expert"], -1)

            self.datalist.append((label, speaker1_path, speaker2_path, age_diff, expert))
        
        return self.datalist
    
    def _load_and_preprocess_audio(self, folder_path: str) -> torch.Tensor:
        # 從 folder_path 中隨機選一個 wav or m4a 讀取
        audio_files = list(Path(folder_path).glob("*.wav")) + list(Path(folder_path).glob("*.m4a"))
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in {folder_path}")
        
        file_path = str(random.choice(audio_files))
        signal, _ = torchaudio.load(file_path)

        # Mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        audio = signal.squeeze(0)  # [1, T] -> [T]
                
        return audio
    
    def __getitem__(self, idx):
        """on-the-fly 讀取兩個音檔並 preprocess，回傳兩個音檔的張量、年齡差距與對應專家標籤"""
        label, folder1, folder2, age_diff, expert = self.datalist[idx]
        waveform1 = self._load_and_preprocess_audio(str(folder1))
        waveform2 = self._load_and_preprocess_audio(str(folder2))

        return label, waveform1, waveform2, age_diff, expert
        
class Vox2Dataset(Dataset):
    """只負責讀取音檔並 preprocess 到 16kHz 單聲道張量"""

    def __init__(
        self,
        audio_dir: str,
        audio_meta_dir: str,
        musan_path=None,
        rir_path=None,
        augment=False,
        num_frames=200,
        min_utts_per_speaker: int = 10,
        suffix: str = ".wav",
        age_target_mode: str = "raw",
    ):
        self.audio_dir = Path(audio_dir)
        self.audio_meta_dir = Path(audio_meta_dir)
        self.musan_path = musan_path
        self.rir_path = rir_path
        self.min_utts_per_speaker = min_utts_per_speaker
        self.age_target_mode = age_target_mode

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
        # self._print_age_label_distribution()

        filtered_speakers = sorted({speaker_id for _, speaker_id, _, _ in self.datalist})
        
        self.speaker2idx = {
            speaker_id: idx
            for idx, speaker_id in enumerate(filtered_speakers)
        }
        
        self.idx2speaker = {
            idx: speaker_id
            for speaker_id, idx in self.speaker2idx.items()
        }
        
        self.num_age_classes = len(self.conv_age)

    def __len__(self):
        return len(self.datalist)

    def _print_age_label_distribution(self):
        """輸出每個年齡組（或年齡標籤）佔所有標籤的比例。"""
        if len(self.datalist) == 0:
            print("年齡標籤統計: datalist 為空")
            return

        age_counts = {}
        for _, _, _, age in self.datalist:
            age_counts[age] = age_counts.get(age, 0) + 1

        total = len(self.datalist)
        print("年齡標籤比例統計:")
        for age_label in sorted(age_counts.keys()):
            count = age_counts[age_label]
            ratio = count / total
            print(f"  年齡標籤 {age_label}: {count}/{total} ({ratio:.2%})")
    
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
            # turn age into age group / raw age
            age_str = row["age"] if "age" in df.columns else row.iloc[2]
            age = int(float(age_str))

            converted_age = self.conv_age.get(next((r for r in self.conv_age if age in r), None), -1)
            if converted_age == -1:
                print(f"Unknown age group: {age} for speaker {speaker}")

            if self.age_target_mode == "group":
                age_target = converted_age
            elif self.age_target_mode == "raw":
                age_target = age
            else:
                raise ValueError(f"Unsupported age_target_mode: {self.age_target_mode}")
            
            gender = row["gender"] if "gender" in df.columns else row.iloc[3]

            # speaker 第一次出現
            if speaker not in meta_dict:
                meta_dict[speaker] = {
                    "gender": gender,
                    "utts": {}
                }

            # 同一 speaker 底下加入不同 utterance
            meta_dict[speaker]["utts"][utt] = {
                "age": age_target
            }
            
        # print the speaker count
        print(f"訓練資料集的說話者數量: {len(meta_dict)}")

        return meta_dict
    
    def get_audio_paths(self):
        data_list = []
        total_speakers = len(self.meta)

        for speaker_id, info in self.meta.items():
            gender = info["gender"]
            utts = list(info["utts"].keys())

            # 保留每位 speaker 的全部 utterance，且把該 utterance 底下所有 wav 與 m4a 都加入
            for utt in utts:
                audio_folder = Path(self.audio_dir) / speaker_id / f"{utt}"
                audio_path = list(audio_folder.rglob(f"*.wav")) + list(audio_folder.rglob(f"*.m4a"))
                if not audio_path:
                    raise FileNotFoundError(f"找不到 speaker {speaker_id} / utt {utt} 的 wav 音檔")

                utt_info = info["utts"][utt]

                for wav_file in sorted(audio_path):
                    data_list.append((str(wav_file), speaker_id, gender, utt_info["age"]))

        print(f"Vox2 speaker filter: kept {total_speakers}/{total_speakers} speakers (all utterances included)")
                
        # count the speaker, gender, age group amount in datalist, and print the min and max utterance amount among speakers
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
        
        # utt_counts = {}
        # for item in data_list:
        #     speaker_id = item[1]
        #     utt_counts[speaker_id] = utt_counts.get(speaker_id, 0) + 1
        # print(f"Min utterances per speaker: {min(utt_counts.values())}")
        # print(f"Max utterances per speaker: {max(utt_counts.values())}")

        return data_list
        
    def _load_and_preprocess_audio(self, file_path: str) -> torch.Tensor:
        """load + mono"""
        signal, _ = torchaudio.load(file_path)

        # Mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        audio = signal.squeeze(0)  # [1, T] -> [T]
                
        return audio

    def __getitem__(self, idx):
        path, speaker_id, gender, age = self.datalist[idx]
        speaker_idx = self.speaker2idx[speaker_id]
        gender_idx = 1 if gender.lower() == 'm' else 0
        waveform = self._load_and_preprocess_audio(str(path))

        return waveform, speaker_idx, gender_idx, age
    
if __name__ == "__main__":
    # 測試 Dataset
    audio_dir = "/app/dataset/VoxCeleb2/wav"
    audio_meta_dir = "/app/dataset/VoxCeleb2/Vox2_delta_LE5.csv"

    dataset = Vox2Dataset(
        audio_dir, 
        audio_meta_dir,
        suffix=".m4a"
    )
    print(f"Dataset size: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        waveforms, speaker_ids, genders, ages = batch
        print(f"Waveforms shape: {waveforms.shape}")
        print(f"Speaker IDs: {speaker_ids}")
        print(f"Genders: {genders}")
        print(f"Ages: {ages}")