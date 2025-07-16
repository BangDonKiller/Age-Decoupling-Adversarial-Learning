import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import torchaudio.functional as F_audio
import numpy as np
import random
import os
import librosa
import warnings
import math
import glob
from tqdm import tqdm # 導入 tqdm 以顯示進度條

# 忽略 librosa 可能發出的警告
warnings.filterwarnings(
    "ignore",
    message="PySoundFile failed. Trying audioread instead."
)
warnings.simplefilter("ignore", category=FutureWarning)


class VoxCeleb_dataset(Dataset):
    def __init__(self, dataset_path, data_list_file, musan_path, rir_path, mode='train', augment=False):
        self.mode = mode
        self.augment = augment
        self.sample_rate = 16000  # 假設採樣率為 16000 Hz
        
        # MelSpectrogram 應保持在 CPU，因為輸入波形是 CPU Tensor
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,         # 25ms 幀長
            hop_length=160,    # 10ms hop_size
            n_mels=80          # 80 維 Mel-filterbank energies
        )
        
        # --- 數據增強相關的初始化和預載入 ---
        if self.augment:
            self.musan_noise_types = ['noise', 'speech', 'music']
            self.musan_path = musan_path
            self.rir_path = rir_path

            # Step 1: 載入所有噪音和 RIR 檔案的路徑
            self.noise_file_paths_by_type = {}
            # self._load_musan_noise_paths(self.musan_path)
            self.rir_file_paths = glob.glob(os.path.join(self.rir_path,'*','*','*.wav'))

            # Step 2: 將所有噪音和 RIR 檔案的波形預載入到記憶體中
            print("Preloading MUSAN noise files to memory...")
            self.loaded_musan_noise_waveforms = self._preload_audios_to_memory(self.noise_file_paths_by_type)
            print("Preloading RIR files to memory...")
            self.loaded_rir_waveforms = self._preload_audios_to_memory(self.rir_file_paths)
            print("Preloading complete.")
        
        # 加載數據列表，這個必須在增強文件預載入之後，因為 _load_data_list 中會檢查文件存在
        self.data_list = self._load_data_list(dataset_path, data_list_file)
            
    def _load_musan_noise_paths(self, musan_path):
        """
        遍歷 MUSAN 數據集目錄，載入所有噪音檔案的路徑並按類型分類。
        """
        augment_files = glob.glob(os.path.join(musan_path,'*','*','*.wav'))
        for file_path in augment_files:
            # 根據 MUSAN 的標準路徑結構，類型通常是倒數第三個資料夾名
            noise_type_key = file_path.split(os.sep)[-3] # 使用 os.sep 確保跨平台兼容性
            if noise_type_key not in self.noise_file_paths_by_type:
                self.noise_file_paths_by_type[noise_type_key] = []
            self.noise_file_paths_by_type[noise_type_key].append(file_path)

    def _preload_audios_to_memory(self, path_collection):
        """
        通用函數，用於將音頻檔案路徑列表或字典中的所有音頻檔案載入記憶體。
        返回一個包含 PyTorch Tensor 波形的字典或列表。
        """
        loaded_data = {} if isinstance(path_collection, dict) else []

        if isinstance(path_collection, dict):
            for noise_type, file_paths in tqdm(path_collection.items(), desc="Preloading noise types"):
                loaded_data[noise_type] = []
                for file_path in tqdm(file_paths, desc=f"  Loading {noise_type}", leave=False):
                    try:
                        # librosa.load 輸出為 NumPy 陣列，轉換為 PyTorch Tensor
                        waveform, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
                        loaded_data[noise_type].append(torch.from_numpy(waveform).float().unsqueeze(0)) # (1, samples)
                    except Exception as e:
                        warnings.warn(f"Failed to preload {file_path}: {e}. Skipping.")
        else: # path_collection is a list of RIR files
            for file_path in tqdm(path_collection, desc="Preloading RIR files"):
                try:
                    waveform, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
                    loaded_data.append(torch.from_numpy(waveform).float().unsqueeze(0)) # (1, samples)
                except Exception as e:
                    warnings.warn(f"Failed to preload {file_path}: {e}. Skipping.")
        return loaded_data

    def _load_data_list(self, dataset_path, data_list_path):
        """
        讀取包含 (audio_path, identity_id, age_group_id) 的列表。
        優化：在初始化時就確定每個樣本的具體音頻文件路徑。
        """
        data_list_raw = np.load(data_list_path, allow_pickle=True).item()
        
        # 根據 mode 選擇不同的數據量 (保留，但如果用於完整訓練，應移除此限制)
        if self.mode == 'train':    
            data_dicts = dict(list(data_list_raw.items())[:100])
        else:
            data_dicts = dict(list(data_list_raw.items())[:10])
        
        data = []
        speaker_id_map = {} # 用於將字串 ID 映射到整數 ID
        next_speaker_int_id = 0 # 下一個可用的整數 ID
        
        # 使用 tqdm 顯示進度條，因為這部分可能耗時
        for key, years_old in tqdm(data_dicts.items(), desc=f"Loading {self.mode} data list"):
            speaker_id = key[:7]
            utterance_id = key[8:] # 這是影片/語音段的 ID
            
            for folder in dataset_path:
                full_audio_dir = os.path.join(folder, speaker_id, utterance_id)
                if not os.path.exists(full_audio_dir):
                    continue
            
            # **關鍵優化點：在初始化時遍歷資料夾，找到所有 .m4a 檔案的路徑**
            m4a_files_in_dir = [
                os.path.join(full_audio_dir, f)
                for f in os.listdir(full_audio_dir) if f.endswith('.m4a') or f.endswith('.wav')
            ]

            if not m4a_files_in_dir:
                warnings.warn(f"No .m4a files found in {full_audio_dir}. Skipping this utterance.")
                continue

            # 確保 speaker_id 是唯一的整數 ID
            if speaker_id not in speaker_id_map:
                speaker_id_map[speaker_id] = next_speaker_int_id
                next_speaker_int_id += 1
            identity_id_int = speaker_id_map[speaker_id] # 獲取對應的整數 ID
            
            # years_olds => {0~20: 0, 21~30: 1, ..., 70~100: 6}
            bins = [20, 30, 40, 50, 60, 70]
            age_group_id = next((i for i, b in enumerate(bins) if years_old <= b), 6)

            # 將該 utterance_id 下所有找到的 .m4a 檔案作為單獨的樣本添加到數據列表中
            for audio_file_path in m4a_files_in_dir:
                 data.append((audio_file_path, identity_id_int, age_group_id))

        print(f"Loaded {len(data)} samples for {self.mode} mode.")
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 現在 self.data_list[idx] 已經是完整的音訊檔案路徑了
        audio_file_path, identity_id, age_group_id = self.data_list[idx]
        
        # **優化點：直接載入確定的音訊檔案，不再需要 os.listdir 或 random.choice**
        waveform, sr = librosa.load(audio_file_path, sr=self.sample_rate, mono=True)
        final_waveform = torch.from_numpy(waveform).float().unsqueeze(0)  # (1, samples)
        
        # --- 數據增強 ---
        if self.augment and self.mode == 'train':
            final_waveform = self._apply_augmentation(final_waveform)

        # 提取 Mel-filterbank energies
        # waveform 可能是 (channels, samples)，MelSpectrogram 期望 (channels, samples)
        mel_spec = self.mel_spectrogram(final_waveform)
        
        # 轉換為論文中期望的形狀 (features, frames)
        # 從 (channels, n_mels, n_frames) 轉為 (n_frames, n_mels)
        # mel_spec.squeeze(0) 將 (1, 80, n_frames) 變成 (80, n_frames)
        # .transpose(0, 1) 將 (80, n_frames) 變成 (n_frames, 80)
        mel_spec = mel_spec.squeeze(0).transpose(0, 1)
        
        # 對數 Mel-filterbank energies
        mel_spec = torch.log(mel_spec + 1e-6)

        # 通常會對特徵進行均值/方差歸一化 (CMVN)
        # 如果需要，可以在這裡實作 _apply_cmvn 方法
        # mel_spec = self._apply_cmvn(mel_spec)

        return mel_spec, identity_id, age_group_id
    
    def collate_fn(self, batch):
        """
        將批次數據填充到相同長度，並調整為模型期望的形狀。
        """
        # 過濾掉 __getitem__ 返回 None 的樣本 (如果有的話，雖然本次修改應該不會)
        # batch = [item for item in batch if item is not None] # 確保 batch 不包含 None
        
        mels, ident, age = zip(*batch)
        
        # 獲取每個 Mel 譜的幀數 (長度)
        lengths = [m.shape[0] for m in mels]
        maxlen = max(lengths) # 找出批次中最長的 Mel 譜幀數
        
        # 對所有 Mel 譜進行零填充，使其長度達到 maxlen
        # (0,0) 對應最後兩個維度 (n_mels)，(0, maxlen-m.shape[0]) 對應第一個維度 (frames)
        padded = [torch.nn.functional.pad(m, (0,0,0,maxlen-m.shape[0])) for m in mels]
        
        # 將填充後的 Mel 譜堆疊成一個批次的 Tensor
        # 原始：torch.stack(padded) 的形狀是 (batch_size, max_frames, n_mels=80)
        stacked_mels = torch.stack(padded) # (B, Max_Frames, 80)
        
        # 1. 轉置 Mel 譜，使 n_mels (80) 成為高度 (H)，max_frames 成為寬度 (W)
        #    從 (B, Max_Frames, 80) 變為 (B, 80, Max_Frames)
        permuted_mels = stacked_mels.permute(0, 2, 1) 
        
        # 2. 插入通道維度 (channels=1)，模型通常期望 (B, C, H, W) 格式
        #    從 (B, 80, Max_Frames) 變為 (B, 1, 80, Max_Frames)
        final_input_mels = permuted_mels.unsqueeze(1)
        
        # 將身份和年齡 ID 轉換為 Tensor
        return final_input_mels, torch.tensor(ident, dtype=torch.long), torch.tensor(age, dtype=torch.long)

    def _apply_augmentation(self, waveform):
        """
        資料強化方法:
            - aug_type = 0: 不進行增強
            - aug_type = 1: 添加噪音
            - aug_type = 2: 混響
            - aug_type = 3: 音量變化
            - aug_type = 4: 速度變化
        
        Args:
            waveform (Tensor): 音頻波形，形狀為 (channels, samples)
        Returns:
            Tensor: 增強後的音頻波形
        """
        aug_type = random.randint(0, 4)
        
        # 確保 waveform 在 CPU 上，因為 librosa 和 torchaudio 某些操作預設在 CPU
        # 並且在 worker 中進行，如果傳入 GPU Tensor，會在 worker 中造成額外複製到 CPU 的開銷
        waveform_on_cpu = waveform.cpu() 

        if aug_type == 0:
            return waveform_on_cpu
        elif aug_type == 1:
            # return self._add_noise(waveform_on_cpu)
            return waveform_on_cpu
        elif aug_type == 2:
            return self._apply_reverberation(waveform_on_cpu)
        elif aug_type == 3:
            return self._change_volume(waveform_on_cpu)
        elif aug_type == 4:
            return self._change_speed(waveform_on_cpu)
        
        return waveform_on_cpu # 確保返回的是 CPU Tensor
    
    def _add_noise(self, waveform):
        """
        添加噪音到音頻波形中。噪音從預載入的 MUSAN 數據集獲取。
        """
        if waveform.shape[1] == 0: # 處理空波形
            return waveform

        snr_db = random.uniform(0, 15) 

        # **優化點：直接從記憶體中獲取噪音波形**
        noise_type = random.choice(self.musan_noise_types)
        noise_pool = self.loaded_musan_noise_waveforms[noise_type]
        selected_noise_waveform = random.choice(noise_pool) # 這已經是 PyTorch Tensor

        # 確保噪音片段長度足夠，如果不足就重複
        if selected_noise_waveform.shape[1] < waveform.shape[1]:
            repeats = math.ceil(waveform.shape[1] / selected_noise_waveform.shape[1])
            selected_noise_waveform = selected_noise_waveform.repeat(1, repeats)
        
        # 裁剪到目標長度
        start_idx = random.randint(0, selected_noise_waveform.shape[1] - waveform.shape[1])
        noise_segment = selected_noise_waveform[:, start_idx : start_idx + waveform.shape[1]]
        
        # 計算 RMS 和縮放
        eps = 1e-6 
        speech_rms = torch.sqrt(torch.mean(waveform**2) + eps)
        noise_rms = torch.sqrt(torch.mean(noise_segment**2) + eps)

        snr_linear = 10**(snr_db / 10.0)
        noise_scaling_factor = (speech_rms / (noise_rms * torch.sqrt(torch.tensor(snr_linear, device=waveform.device))))
        
        scaled_noise = noise_segment * noise_scaling_factor
        noisy_waveform = waveform + scaled_noise
        
        return torch.clamp(noisy_waveform, -1.0, 1.0)
    
    def _apply_reverberation(self, waveform):
        """
        應用混響。RIRs 從預載入的數據集獲取。
        """
        if not self.loaded_rir_waveforms:
            warnings.warn("No RIR files preloaded for reverberation. Returning original waveform.")
            return waveform
        
        # **優化點：直接從記憶體中獲取 RIR 波形**
        rir_tensor = random.choice(self.loaded_rir_waveforms) # 這已經是 PyTorch Tensor

        # 將 RIR 正規化
        rir_tensor = rir_tensor / torch.norm(rir_tensor)

        # 進行卷積操作，確保輸入和 RIR 都是 (1, samples)
        reverb_waveform = F_audio.convolve(waveform, rir_tensor)

        return torch.clamp(reverb_waveform, -1.0, 1.0)
    
    def _change_volume(self, waveform):
        """
        隨機改變音量。
        """
        gain = random.uniform(0.5, 1.5)
        adjusted_waveform = waveform * gain
        return torch.clamp(adjusted_waveform, -1.0, 1.0)
    
    def _change_speed(self, waveform):
        """
        隨機改變音頻速度，同時保持音高不變。
        """
        speed_factor = random.uniform(0.7, 1.3) # 隨機選擇速度因子，例如 0.7 到 1.3 倍，避免極端速度
        if speed_factor == 1.0:
            return waveform # 不變速

        new_sample_rate = int(self.sample_rate * speed_factor)
        
        # torchaudio 的 Resample 預設在 CPU 運行，因為輸入 waveform 是 CPU Tensor
        resampler = T.Resample(orig_freq=self.sample_rate, new_freq=new_sample_rate)
        
        # 執行重新採樣
        speed_waveform = resampler(waveform)
        
        return speed_waveform
    
    # 僅用於測試
    def generate_song(self, origin_audio, adjusted_audio):
        """
        將波型圖還原回音訊，聽看看差別
        """
        import soundfile as sf
        # 將音訊轉換為 numpy 陣列
        origin_audio_np = origin_audio.squeeze().cpu().numpy()
        adjusted_audio_np = adjusted_audio.squeeze().cpu().numpy()

        # 使用 librosa 將音訊寫入檔案
        sf.write('origin_audio.wav', origin_audio_np, samplerate=self.sample_rate)
        sf.write('adjusted_audio.wav', adjusted_audio_np, samplerate=self.sample_rate)
        
        print("音訊已保存為 'origin_audio.wav' 和 'adjusted_audio.wav'. 請使用音訊播放器播放。")
    
# Testing
# if __name__ == "__main__":
#     # 請根據您的實際路徑修改這裡
#     dataset_path = ['D:/Dataset/VoxCeleb2/vox2_dev_wav/dev/aac']
#     data_list_file = 'D:/Dataset/Cross-Age_Speaker_Verification/vox2dev/segment2age.npy'
#     musan_path = 'D:/Dataset/musan/musan'
#     rir_path = 'D:/Dataset/sim_rir_16k/simulated_rirs_16k'
#     val_dataset_path = ['D:/Dataset/VoxCeleb1/vox1_dev_wav/wav', "D:/Dataset/VoxCeleb1/vox1_test_wav/wav"]
#     val_data_list_file = 'D:/Dataset/Cross-Age_Speaker_Verification/vox1/segment2age.npy'

#     print("Initializing training dataset (with augmentation)...")
#     train_dataset = VoxCeleb_dataset(
#         dataset_path=dataset_path,
#         data_list_file=data_list_file,
#         musan_path=musan_path,
#         rir_path=rir_path,
#         mode='train',
#         augment=False
#     )
#     print(f"Training Dataset size: {len(train_dataset)}")

#     print("\nInitializing validation dataset (no augmentation)...")
#     val_dataset = VoxCeleb_dataset(
#         dataset_path=val_dataset_path,
#         data_list_file=val_data_list_file,
#         musan_path=musan_path, # 即使不增強，也需要這些參數傳入
#         rir_path=rir_path,     # 為了保持 __init__ 參數一致
#         mode='val', # 或 'test'
#         augment=False
#     )
#     print(f"Validation Dataset size: {len(val_dataset)}")

#     from torch.utils.data import DataLoader
#     import time
#     import os

#     print("\nCreating DataLoader...")
#     # 建議 num_workers 設定為 CPU 核心數的 1/2 到 3/4，或者 os.cpu_count() - 1
#     # pin_memory=True 對於 GPU 訓練至關重要
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=32,
#         shuffle=True,
#         num_workers=0, # 根據你的 CPU 核心數調整
#         pin_memory=True,
#         collate_fn=train_dataset.collate_fn
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=32,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=True,
#         collate_fn=val_dataset.collate_fn
#     )

#     print(f"Testing DataLoader (first {min(5, len(train_loader))} batches for train_loader):")
#     start_time = time.time()
#     for i, (mels, ident, age) in enumerate(train_loader):
#         print(f"Batch {i+1}: Mel Spec Shape: {mels.shape}, Identity IDs: {ident.shape}, Age Group IDs: {age.shape}")
#         if i >= 4: # 只測試前5個批次
#             break
#     end_time = time.time()
#     print(f"Time to load 5 batches (train_loader): {end_time - start_time:.2f} seconds")

#     print(f"\nTesting DataLoader (first {min(5, len(val_loader))} batches for val_loader):")
#     start_time = time.time()
#     for i, (mels, ident, age) in enumerate(val_loader):
#         print(f"Batch {i+1}: Mel Spec Shape: {mels.shape}, Identity IDs: {ident.shape}, Age Group IDs: {age.shape}")
#         if i >= 4: # 只測試前5個批次
#             break
#     end_time = time.time()
#     print(f"Time to load 5 batches (val_loader): {end_time - start_time:.2f} seconds")

#     print("\nDataLoader test complete.")