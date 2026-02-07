# dataloader.py 完整代碼
import pandas as pd
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, Sampler
import random

class AgeGroupBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.age_indices = {0: [], 1: [], 2: []}
        
        # 建立索引映射
        for i, (_, _, _, age) in enumerate(dataset.datalist):
            if age in self.age_indices:
                self.age_indices[age].append(i)
        
        # 以壯年組 (1) 為基準，這確保了每一輪都能跑完所有核心數據
        self.num_batches = len(self.age_indices[1]) // batch_size
        
        print(f"[Sampler] 訓練 Step 數已對齊壯年組: {self.num_batches}")
        print(f"[Sampler] 幼年組將循環使用約 {self.num_batches * batch_size / len(self.age_indices[0]):.1f} 次")
        print(f"[Sampler] 老年組將循環使用約 {self.num_batches * batch_size / len(self.age_indices[2]):.1f} 次")

    def __iter__(self):
        # 每個 Epoch 開始前打亂
        for k in self.age_indices:
            random.shuffle(self.age_indices[k])
            
        # 用於記錄小組別目前抽到哪裡的指標
        pointers = {0: 0, 2: 0}
        
        for i in range(self.num_batches):
            # 1. 壯年組：正常順序抽取
            batch_1 = self.age_indices[1][i * self.batch_size : (i + 1) * self.batch_size]
            
            # 2. 幼年組與老年組：循環抽取邏輯
            def get_cyclic_batch(age):
                start = pointers[age]
                end = start + self.batch_size
                
                # 如果快抽完了，就重新打亂並回到頭部
                if end > len(self.age_indices[age]):
                    random.shuffle(self.age_indices[age])
                    pointers[age] = 0
                    start = 0
                    end = self.batch_size
                
                batch = self.age_indices[age][start:end]
                pointers[age] = end
                return batch

            batch_0 = get_cyclic_batch(0)
            batch_2 = get_cyclic_batch(2)
            
            # 依序回傳 0, 1, 2 組的索引拼接
            yield batch_0 + batch_1 + batch_2

    def __len__(self):
        return self.num_batches

class Vox2Dataset(Dataset):
    def __init__(self, audio_dir: str, audio_meta_dir: str, target_sample_rate: int = 16000, suffix: str = ".wav"):
        self.audio_dir = Path(audio_dir)
        self.audio_meta_dir = Path(audio_meta_dir)
        self.target_sr = target_sample_rate
        self.conv_age = {
            range(0, 21): 0, 
            range(21, 56): 1, 
            range(56, 80): 2
        }
        self.meta = self.read_meta_file(self.audio_meta_dir)
        self.datalist = self.get_audio_paths()
        self.speaker2idx = {s: i for i, s in enumerate(sorted(self.meta.keys()))}
        self.num_age_classes = len(self.conv_age)

    def read_meta_file(self, meta_path: str):
        df = pd.read_csv(meta_path, sep=",", header=0, usecols=[0,1,2,3], dtype={"age": str})
        meta_dict = {}
        for _, row in df.iterrows():
            speaker = row.iloc[0]; utt = row.iloc[1]; age = int(row.iloc[2]); gender = row.iloc[3]
            converted_age = self.conv_age.get(next((r for r in self.conv_age if age in r), None), -1)
            if speaker not in meta_dict: meta_dict[speaker] = {"gender": gender, "utts": {}}
            meta_dict[speaker]["utts"][utt] = {"age": converted_age}
        return meta_dict
    
    def get_audio_paths(self, num_utts_per_speaker=10):
        data_list = []
        for speaker_id, info in self.meta.items():
            utts = list(info["utts"].keys())
            sampled_utts = random.sample(utts, k=min(num_utts_per_speaker, len(utts)))
            for utt in sampled_utts:
                audio_folder = Path(self.audio_dir) / speaker_id / f"{utt}"
                audio_path = list(audio_folder.rglob(f"*.m4a"))
                if not audio_path: continue
                data_list.append((str(audio_path[0]), speaker_id, info["gender"], info["utts"][utt]["age"]))
        return data_list
        
    def _load_and_preprocess_audio(self, file_path: str) -> torch.Tensor:
        signal, fs = torchaudio.load(file_path)
        if signal.shape[1] > fs * 3: signal = signal[:, : fs * 3]
        else: signal = torch.nn.functional.pad(signal, (0, fs * 3 - signal.shape[1]))
        if fs != self.target_sr: signal = T.Resample(orig_freq=fs, new_freq=self.target_sr)(signal)
        if signal.shape[0] > 1: signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def __getitem__(self, idx):
        path, _, _, age = self.datalist[idx]
        return self._load_and_preprocess_audio(str(path)), self.speaker2idx[self.datalist[idx][1]], age