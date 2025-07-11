import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import numpy as np
import random
import os
import librosa
import warnings

warnings.filterwarnings(
    "ignore",
    message="PySoundFile failed. Trying audioread instead."
)
warnings.simplefilter("ignore", category=FutureWarning)



class VoxCeleb_dataset(Dataset):
    def __init__(self, dataset_path, data_list_file, mode='train', augment=False):
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
            # 這裡您可以根據需求處理，例如返回一個 None 或一個默認的靜音音頻
            # 對於訓練，通常會從 DataLoader 中篩選掉這種無效樣本
            print(f"Warning: No valid audio segments loaded for {audio_folder_path}. Skipping this sample.")
            return None # 或拋出異常，讓 DataLoader 的 collate_fn 處理

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
            waveform = self._add_noise(waveform, sample_rate)
        elif aug_type == 2:
            waveform = self._apply_reverberation(waveform, sample_rate)
        elif aug_type == 3:
            waveform = self._change_volume(waveform)
        elif aug_type == 4:
            waveform = self._change_speed(waveform)
        
        return waveform
    
    def _add_noise(self, waveform, sample_rate):
        return
    
    def _apply_reverberation(self, waveform, sample_rate):
        return
    
    def _change_volume(self, waveform):
        return 
    
    def _change_speed(self, waveform):
        return
    
# Testing
if __name__ == "__main__":
    # 測試數據集
    dataset = VoxCeleb_dataset(dataset_path='D:/Dataset/VoxCeleb2/vox2_dev_wav/dev/aac', 
                               data_list_file='D:/Dataset/Cross-Age_Speaker_Verification/vox2dev/segment2age.npy', 
                               mode='train', 
                               augment=False)
    print(f"Dataset size: {len(dataset)}")
    
    for i in range(5):
        mel_spec, identity_id, age_group_id = dataset[i]
        print(f"Sample {i}: Mel Spec Shape: {mel_spec.shape}, Identity ID: {identity_id}, Age Group ID: {age_group_id}")