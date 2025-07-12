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

warnings.filterwarnings(
    "ignore",
    message="PySoundFile failed. Trying audioread instead."
)
warnings.simplefilter("ignore", category=FutureWarning)



class VoxCeleb_loader(Dataset):
    def __init__(self, dataset_path, data_list_file, musan_path, rir_path, mode='train', augment=False):
        self.data_list = self._load_data_list(dataset_path, data_list_file)
        self.mode = mode
        self.augment = augment
        self.sample_rate = 16000  # 假設採樣率為 16000 Hz
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=16000, # 假設採樣率
            n_fft=400,         # 25ms 幀長
            hop_length=160,    # 10ms hop_size
            n_mels=80          # 80 維 Mel-filterbank energies
        )
        
        # --- 數據增強相關的初始化 ---
        if self.augment:
            self.musan_noise_types = ['noise', 'speech', 'music'] # 論文中提到 MUSAN
            self.musan_path = musan_path
            self.noiselist = {}
            self._load_musan_noise(self.musan_path)
            self.rir_files = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
        
            
    def _load_musan_noise(self, musan_path):
        augment_files = glob.glob(os.path.join(musan_path,'*/*/*.wav'))
        for file in augment_files:
            if file.split('\\')[-3] not in self.noiselist:
                self.noiselist[file.split('\\')[-3]] = []
            self.noiselist[file.split('\\')[-3]].append(file)

    def _load_data_list(self, dataset_path, data_list_path):
        # 讀取包含 (audio_path, identity_id, age_group_id) 的列表        
        data_list = np.load(data_list_path, allow_pickle=True)
        data_dicts = data_list.item()
        
        data = []
        
        for key, value in data_dicts.items():
            speaker_id = key[:7]
            utterance = key[8:]
            years_old = value
            
            # years_olds => {0~20: 0, 21~30: 1, ..., 70~100: 6}
            audio_folder_path = f"{dataset_path}/{speaker_id}/{utterance}/"
            identity_id = speaker_id
            bins = [20, 30, 40, 50, 60, 70]
            age_group_id = next((i for i, b in enumerate(bins) if years_old <= b), 6)

            
            data.append((audio_folder_path, identity_id, age_group_id))

        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        audio_folder_path, identity_id, age_group_id = self.data_list[idx]

        # 遍歷 audio_folder_path 中的所有音頻文件
        audio_files = [f for f in os.listdir(audio_folder_path) if f.endswith('.m4a')]
        
        concatenated_waveform = []

        for audio_filename in audio_files:
            audio_file = os.path.join(audio_folder_path, audio_filename)
            
            try:
                # 使用 librosa 讀取音頻。librosa 會自動處理 .m4a
                # sr=None 會保留原始採樣率，但為了 MelSpectrogram 一致性，指定為 self.sample_rate
                waveform_segment, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)
                # 將 NumPy 陣列轉換為 PyTorch 張量，並轉為 float
                concatenated_waveform.append(torch.from_numpy(waveform_segment).float())
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
                continue

        # 如果該資料夾下沒有有效的音檔，返回 None 或拋出錯誤
        if not concatenated_waveform:
            print(f"Warning: No valid audio segments loaded for {audio_folder_path}. Skipping this sample.")
            return None

        # 將所有音頻波形片段串接起來
        # 確保波形是 (samples,) 或 (1, samples) 格式，MelSpectrogram 期望 (channels, samples)
        final_waveform = torch.cat(concatenated_waveform, dim=-1)
        final_waveform = final_waveform.unsqueeze(0) # 轉為 (1, samples) 才能給 MelSpectrogram

        if self.augment and self.mode == 'train':
            final_waveform = self._apply_augmentation(final_waveform, sr)

        # 提取 Mel-filterbank energies
        # waveform 可能是 (channels, samples)
        mel_spec = self.mel_spectrogram(final_waveform)
        # 轉換為論文中期望的形狀 (features, frames)
        mel_spec = mel_spec.squeeze(0).transpose(0, 1) # 從 (channels, n_mels, n_frames) 轉為 (n_frames, n_mels)
        # 對數 Mel-filterbank energies
        mel_spec = torch.log(mel_spec + 1e-6)

        # 通常會對特徵進行均值/方差歸一化 (CMVN)
        # mel_spec = self._apply_cmvn(mel_spec)

        return mel_spec, identity_id, age_group_id

    def _apply_augmentation(self, waveform, sample_rate):
        """
        資料強化方法:
            - aug_type = 0: 不進行增強
            - aug_type = 1: 添加噪音
            - aug_type = 2: 混響
            - aug_type = 3: 音量變化
            - aug_type = 4: 速度變化
        
        Args:
            waveform (Tensor): 音頻波形，形狀為 (channels, samples)
            sample_rate (int): 音頻採樣率
        Returns:
            Tensor: 增強後的音頻波形
        """
        aug_type = random.randint(0, 4)
        if aug_type == 0:
            waveform = waveform
        elif aug_type == 1:
            waveform = self._add_noise(waveform)
        elif aug_type == 2:
            waveform = self._apply_reverberation(waveform)
        elif aug_type == 3:
            waveform = self._change_volume(waveform)
        elif aug_type == 4:
            waveform = self._change_speed(waveform)
        
        return waveform
    
    def _add_noise(self, waveform):
        """
        添加噪音到音頻波形中。
        噪音從 MUSAN 數據集模擬獲取。
        SNR（信噪比）隨機選擇在一個範圍內（例如 0 到 15 dB）。

        Args:
            waveform (Tensor): 原始音頻波形 (channels, samples)，通常是 (1, samples)。
            sample_rate (int): 音頻採樣率。

        Returns:
            Tensor: 加入噪音後的音頻波形。
        """
        if waveform.shape[1] == 0: # 處理空波形
            return waveform

        # 隨機選擇一個 SNR 值，例如從 0 dB 到 15 dB
        snr_db = random.uniform(0, 15) 

        # 1. 獲取一個與原始音頻長度相同的噪音片段
        # 這裡使用模擬函數，實際應從 MUSAN 載入
        noise_waveform = self._get_random_noise_segment(waveform.shape[1])
        
        # 確保 noise_waveform 和 waveform 都在相同的設備上
        noise_waveform = noise_waveform.to(waveform.device)

        # 2. 計算原始語音和噪音的 RMS（均方根）能量
        # 添加一個小的 epsilon 以防止除以零，尤其當波形是全零時
        eps = 1e-6 
        speech_rms = torch.sqrt(torch.mean(waveform**2) + eps)
        noise_rms = torch.sqrt(torch.mean(noise_waveform**2) + eps)

        # 3. 將 SNR（dB）轉換為線性比例
        snr_linear = 10**(snr_db / 10.0)

        # 4. 計算噪音需要縮放的因子，以達到目標 SNR
        # 目標噪音 RMS = 語音 RMS / sqrt(SNR_linear)
        # 噪音縮放因子 = 目標噪音 RMS / 實際噪音 RMS
        noise_scaling_factor = (speech_rms / (noise_rms * torch.sqrt(torch.tensor(snr_linear, device=waveform.device))))

        # 5. 縮放噪音並將其添加到原始語音中
        scaled_noise = noise_waveform * noise_scaling_factor
        noisy_waveform = waveform + scaled_noise

        # 6. 可選：將結果波形裁剪到 -1.0 到 1.0 的範圍，防止過載
        noisy_waveform = torch.clamp(noisy_waveform, -1.0, 1.0)
        
        # 轉回音訊聽看看
        # self.generate_song(waveform, noisy_waveform)

        return noisy_waveform
    
    def _apply_reverberation(self, waveform):
        if not self.rir_files:
            warnings.warn("No RIR files loaded for reverberation. Returning original waveform.")
            return waveform
        
        rir_file = random.choice(self.rir_files)

        try:
            rir_np, sr_rir = librosa.load(rir_file, sr=self.sample_rate, mono=True)
            rir_tensor = torch.from_numpy(rir_np).float()

            # 將 RIR 移到與輸入波形相同的設備
            rir_tensor = rir_tensor.to(waveform.device)
            
            # 正規化
            rir_tensor = rir_tensor / torch.norm(rir_tensor)

            # 進行卷積操作
            reverb_waveform = F_audio.convolve(waveform, rir_tensor.unsqueeze(0))

            # 將結果波形裁剪到 -1.0 到 1.0 的範圍，防止過載
            reverb_waveform = torch.clamp(reverb_waveform, -1.0, 1.0)
            
            # For testing
            # self.generate_song(waveform, reverb_waveform)

            return reverb_waveform

        except Exception as e:
            warnings.warn(f"Error applying reverberation from {rir_file}: {e}. Returning original waveform.")
            return waveform
    
    def _change_volume(self, waveform):
        """
        隨機改變音量。
        Args:
            waveform (Tensor): 音頻波形，形狀為 (channels, samples)。
        Returns:
            Tensor: 音量調整後的音頻波形。
        """
        gain = random.uniform(0.5, 1.5) # 隨機選擇增益因子，例如 0.5 到 1.5 倍
        adjusted_waveform = waveform * gain
        adjusted_waveform = torch.clamp(adjusted_waveform, -1.0, 1.0) # 防止裁剪
        
        # For testing
        # self.generate_song(waveform, adjusted_waveform)
        return adjusted_waveform
    
    def _change_speed(self, waveform):
        """
        隨機改變音頻速度，同時保持音高不變。
        這通常通過重新採樣實現。

        Args:
            waveform (Tensor): 音頻波形，形狀為 (channels, samples)。
        Returns:
            Tensor: 速度調整後的音頻波形。
        """
        
        speed_factor = random.uniform(0.5, 1.5) # 隨機選擇速度因子，例如 0.5 到 1.5 倍
        if speed_factor == 1.0:
            return waveform # 不變速

        # 計算新的採樣率
        new_sample_rate = int(self.sample_rate * speed_factor)
        
        # 使用 torchaudio 的 Resample 進行速度變化 (同時保持音高)
        resampler = T.Resample(orig_freq=self.sample_rate, new_freq=new_sample_rate).to(waveform.device)
        
        # 執行重新採樣
        speed_waveform = resampler(waveform)
        
        # For testing
        # self.generate_song(waveform, speed_waveform)
        
        return speed_waveform
    
    def _get_random_noise_segment(self, target_length_samples):
        """
        模擬從 MUSAN 數據集中獲取一個隨機噪音片段。
        在實際應用中，這裡會從預載入的噪音文件路徑中隨機選擇一個，
        並從該文件中載入一個長度足夠的片段。
        """
        noise_type = random.choice(self.musan_noise_types)
        noise_pool = self.noiselist[noise_type]
        
        selected_noise = random.choice(noise_pool)
        
        selected_audio, sr = librosa.load(selected_noise, sr=self.sample_rate, mono=True)
        
        selected_audio = torch.from_numpy(selected_audio).float().unsqueeze(0)  # 轉為 (1, samples)
        
        # 如果噪音片段比目標長度短，則重複填充
        if selected_audio.shape[1] < target_length_samples:
            repeats = math.ceil(target_length_samples / selected_audio.shape[1])
            selected_audio = selected_audio.repeat(1, repeats)
        
        # 裁剪到目標長度
        start_idx = random.randint(0, selected_audio.shape[1] - target_length_samples)
        noise_segment = selected_audio[:, start_idx : start_idx + target_length_samples]
        
        return noise_segment.float()
    
    # Only for testing
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
if __name__ == "__main__":
    # 測試數據集
    dataset = VoxCeleb_loader(dataset_path='D:/Dataset/VoxCeleb2/vox2_dev_wav/dev/aac', 
                               data_list_file='D:/Dataset/Cross-Age_Speaker_Verification/vox2dev/segment2age.npy',
                               musan_path='D:/Dataset/musan/musan',
                               rir_path='D:/Dataset/sim_rir_16k/simulated_rirs_16k', 
                               mode='train', 
                               augment=True)
    print(f"Dataset size: {len(dataset)}")
    
    for i in range(5):
        mel_spec, identity_id, age_group_id = dataset[i]
        print(f"Sample {i}: Mel Spec Shape: {mel_spec.shape}, Identity ID: {identity_id}, Age Group ID: {age_group_id}")