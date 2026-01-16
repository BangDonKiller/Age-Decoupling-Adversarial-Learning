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
    def __init__(self, audio_dir, audio_meta_dir):
        """
        file_dir: voxceleb1 根目錄
        meta_dir: voxceleb1_meta.csv 路徑
        依據 meta 建立說話者到語音檔案的映射(目前只取 dev 中的資料)
        """
        self.file_dir = audio_dir
        self.datalist = self.read_txt(audio_meta_dir)

        print(f"Dataset 初始化完成，總共有 {len(self.datalist)} 筆資料。")

    def __len__(self):
        return len(self.datalist)
    
    def read_txt(self, meta_dir):
        datalist = []
        
        with open(meta_dir, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.split(" ")
            is_same_speaker = int(line[0])
            spk1_path = self.find_audio_path(line[1])
            spk2_path = self.find_audio_path(line[2].strip())
            spk1_id = line[1].split("/")[0]
            spk2_id = line[2].split("/")[0]
            
            datalist.append((is_same_speaker, spk1_id, spk2_id, spk1_path, spk2_path))
            
        # 計算有多少組正對、有多少組負對
        pos_count = sum(1 for item in datalist if item[0] == 1)
        neg_count = sum(1 for item in datalist if item[0] == 0)
        print(f"正對數量: {pos_count}, 負對數量: {neg_count}")    
        
        return datalist
    
    def find_audio_path(self, relative_path):
        for audio_dir in self.file_dir:
            audio_path = Path(audio_dir) / relative_path
            if audio_path.exists():
                return str(audio_path)
        raise FileNotFoundError(f"Audio file {relative_path} not found in any of the provided directories.")
    
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
        is_same_speaker, spk1_id, spk2_id, spk1_audio_path, spk2_audio_path = self.datalist[idx]
        signal1 = self._audio_processing(spk1_audio_path)
        signal2 = self._audio_processing(spk2_audio_path)
        return torch.tensor(is_same_speaker, dtype=torch.float32), spk1_id, spk2_id, signal1, signal2



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
                    "utterance_2": {"age": 5},
                    ...
                }
            },
            "speaker_id_2": {
                "gender": "F",
                "utts": {
                    "utterance_1": {"age": 3},
                    "utterance_2": {"age": 4},
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
                for audio_dir in self.audio_dir:
                    audio_folder = Path(audio_dir) / speaker_id / f"{utt}"
                    if audio_folder.exists():
                        break
                audio_path = list(audio_folder.rglob(f"*.wav"))
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
    file_dir = [
        Path("D:\\Dataset\\VoxCeleb1\\vox1_dev_wav\\wav"), 
        Path("D:\\Dataset\\VoxCeleb1\\vox1_test_wav\\wav")
    ]
    meta_dir = Path("D:\\Dataset\\VoxCeleb1\\vox1_test.txt")

    dataset = PairwiseDataset(audio_dir=file_dir, audio_meta_dir=meta_dir)
    print(f"Dataset 長度: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # # 印出前三個
    for i, (is_same, spk1, spk2) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"  is_same: {is_same}")
        print(f"  spk1 shape: {spk1.shape}")
        print(f"  spk2 shape: {spk2.shape}")
        if i == 2:
            break