import pandas as pd
from pathlib import Path
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import glob
import os

class Vox2Dataset(Dataset):
    """只負責讀取音檔並 preprocess 到 16kHz 單聲道張量"""

    def __init__(self, audio_dir: str, audio_meta_dir: str, musan_path, rir_path, augment=False, num_frames=200, suffix: str = ".wav", age_target_mode: str = "raw"):
        self.audio_dir = Path(audio_dir)
        self.audio_meta_dir = Path(audio_meta_dir)
        self.augment = augment
        self.num_frames = num_frames
        self.age_target_mode = age_target_mode
        
        # 定義噪音類型與對應 SNR 範圍與數量
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}

        # 建立噪音資料清單
        self.noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path,'*/*/*.wav'))
        for file in augment_files:
            if file.split('\\')[-3] not in self.noiselist:
                self.noiselist[file.split('\\')[-3]] = []
            self.noiselist[file.split('\\')[-3]].append(file)
            
        # for noisetype, files in self.noiselist.items():
        #     print(f"  {noisetype}: {len(files)} 個檔案")

        # 讀取混響 RIR 檔案
        self.rir_files = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
        
        # print(f"  RIR 檔案數量: {len(self.rir_files)}")

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
        
        self.speaker2idx = {
            speaker_id: idx
            for idx, speaker_id in enumerate(sorted(self.meta.keys()))
        }
        
        self.idx2speaker = {
            idx: speaker_id
            for speaker_id, idx in self.speaker2idx.items()
        }
        
        self.num_age_classes = len(self.conv_age)

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
            # turn age into age group / raw age
            age_str = row["age"] if "age" in df.columns else row.iloc[2]
            age = int(age_str)

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
        """load + resample + mono"""
        signal, fs = torchaudio.load(file_path)

        if fs != 16000:
            signal = torchaudio.functional.resample(signal, fs, 16000)

        # Mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        audio = self._match_waveform_length(signal, self._target_num_samples())
        
        if self.augment:
            augtype = random.randint(0, 5)
            if augtype == 0:   # 原始資料
                audio = signal
            elif augtype == 1: # 混響
                audio = self.add_rev(signal)
            elif augtype == 2: # 語音型噪音（多人講話）
                audio = self.add_noise(signal, 'speech')
            elif augtype == 3: # 音樂噪音
                audio = self.add_noise(signal, 'music')
            elif augtype == 4: # 背景噪音
                audio = self.add_noise(signal, 'noise')
            elif augtype == 5: # 混合噪音（電視情境）
                audio = self.add_noise(signal, 'speech')
                audio = self.add_noise(audio, 'music')
            
        audio = audio.squeeze(0)  # [1, T] -> [T]
                
        return audio

    def _target_num_samples(self) -> int:
        return self.num_frames * 160 + 240

    def _match_waveform_length(self, waveform: torch.Tensor, target_length: int) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        current_length = waveform.shape[1]
        if current_length == 0:
            return torch.zeros((1, target_length), dtype=waveform.dtype, device=waveform.device)

        if current_length < target_length:
            shortage = target_length - current_length
            wrap_indices = torch.arange(shortage, device=waveform.device) % current_length
            waveform = torch.cat((waveform, waveform[:, wrap_indices]), dim=1)
            current_length = waveform.shape[1]

        start_frame = random.randint(0, current_length - target_length)
        waveform = waveform[:, start_frame:start_frame + target_length]

        return waveform

    def _load_augmentation_audio(self, file_path: str, target_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        waveform = self._match_waveform_length(waveform, target_length)
        return waveform.to(device=device, dtype=dtype)
    
    def add_rev(self, audio):
        """
        加入混響效果：
        - 從 RIR 檔案中選擇一個
        - 與原始語音做卷積模擬混響
        """
        target_length = self._target_num_samples()
        audio = self._match_waveform_length(audio, target_length)
        rir_file = random.choice(self.rir_files)
        rir = self._load_augmentation_audio(rir_file, target_length, audio.device, audio.dtype)
        rir = rir / torch.linalg.vector_norm(rir, ord=2).clamp_min(1e-6)
        reverberated = F.conv1d(
            audio.unsqueeze(0),
            rir.flip(-1).unsqueeze(0),
            padding=rir.shape[-1] - 1,
        ).squeeze(0)
        return reverberated[:, :target_length]

    def add_noise(self, audio, noisecat):
        """
        加入背景噪音：
        - 根據類型選擇 SNR、數量
        - 從噪音資料集中隨機取出
        - 根據 SNR 調整音量後加入語音
        """
        target_length = self._target_num_samples()
        audio = self._match_waveform_length(audio, target_length)
        
        # 計算乾淨語音的平均功率 (DB)
        clean_db = 10 * torch.log10(audio.pow(2).mean().clamp_min(1e-4))
        
        # 決定加入的噪音數量和選擇噪音檔案
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        
        
        noises = []
        for noise in noiselist:
            noiseaudio = self._load_augmentation_audio(noise, target_length, audio.device, audio.dtype)
            noise_db = 10 * torch.log10(noiseaudio.pow(2).mean().clamp_min(1e-4))
            
            # 確定目標的信噪比 (SNR)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            scale = torch.pow(
                torch.tensor(10.0, dtype=audio.dtype, device=audio.device),
                (clean_db - noise_db - noisesnr) / 20.0,
            )
            noises.append(scale * noiseaudio)
        noise = torch.cat(noises, dim=0).sum(dim=0, keepdim=True)
        return noise + audio

    def __getitem__(self, idx):
        path, speaker_id, gender, age = self.datalist[idx]
        speaker_idx = self.speaker2idx[speaker_id]
        gender_idx = 1 if gender.lower() == 'm' else 0
        waveform = self._load_and_preprocess_audio(str(path))
        return waveform, speaker_idx, gender_idx, age
    
if __name__ == "__main__":
    # 測試 Dataset
    audio_dir = "D:\\Dataset\\VoxCeleb2\\vox2_dev_wav\\dev\\aac"
    audio_meta_dir = "D:\\Dataset\\VoxCeleb2\\vox2_meta.csv"
    
    dataset = Vox2Dataset(
        audio_dir, 
        audio_meta_dir,
        musan_path="D:\\Dataset\\musan\\musan",
        rir_path="D:\\Dataset\\sim_rir_16k\\simulated_rirs_16k",    
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
        
        