import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import numpy as np
import random
import os
import librosa
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 忽略 librosa 可能發出的警告
warnings.filterwarnings(
    "ignore",
    message="PySoundFile failed. Trying audioread instead."
)
warnings.simplefilter("ignore", category=FutureWarning)

class eval_loader(Dataset):
    def __init__(self, dataset_path, data_list_file,frame_num):
        self.sample_rate = 16000
        self.frame_num = frame_num

        # MelSpectrogram 應保持在 CPU，因為輸入波形是 CPU Tensor
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,         # 25ms 幀長
            hop_length=160,    # 10ms hop_size
            n_mels=80          # 80 維 Mel-filterbank energies
        )

        # 加載數據列表，這個必須在增強文件預載入之後，因為 _load_data_list 中會檢查文件存在
        self.data_list = self._load_data_list(dataset_path, data_list_file)
        
    def __len__(self):
        return len(self.data_list)

    def _load_data_list(self, dataset_path, data_list_path):
        """
        讀取包含 (audio1_path, audio2_path, label) 的列表。
        """
        with open(data_list_path, 'r') as f:
            data_list_raw = f.readlines()

        data_list_raw = [line.strip().split() for line in data_list_raw]

        data_list_raw = random.sample(data_list_raw, 10000)  # 隨機選擇 10000 條數據

        data = []
        
        for line in data_list_raw:
            audio11_path = os.path.join(dataset_path[0], line[1])
            audio12_path = os.path.join(dataset_path[1], line[1])
            audio13_path = os.path.join(dataset_path[2], line[1])
            
            audio21_path = os.path.join(dataset_path[0], line[2])
            audio22_path = os.path.join(dataset_path[1], line[2])
            audio23_path = os.path.join(dataset_path[2], line[2])

            # 確保音訊檔案存在
            if os.path.exists(audio11_path):
                audio1_path = audio11_path
            elif os.path.exists(audio12_path):
                audio1_path = audio12_path
            else:
                audio1_path = audio13_path
            
            if os.path.exists(audio21_path):
                audio2_path = audio21_path
            elif os.path.exists(audio22_path):
                audio2_path = audio22_path
            else:
                audio2_path = audio23_path
    

            data.append((int(line[0]), audio1_path, audio2_path))  # (label, audio1, audio2, audio3)
        return data
    
    def spec_to_rgb(self, spec):
        """
        將單通道 spectrogram 視覺化成 RGB image。
        spec: 2D array (H, W)
        return: 3D uint8 RGB image: shape (H, W, 3)
        """
        return spec.repeat(1, 3, 1, 1)
    
    def min_max_normalize(self, spec):
        """
        將 PyTorch Tensor (B, C, H, W) 中的每個 (H, W) 頻譜圖獨立進行 Min-Max 正規化到 [0, 1]。
        
        Args:
            spec: PyTorch Tensor, shape (B, C, H, W).
                          通常 dtype 會是 float32 或 float64.
        
        Returns:
            PyTorch Tensor, shape (B, C, H, W), 正規化後的頻譜圖，值在 [0, 1] 範圍。
        """
        
        normalized_tensors = []
        
        # 遍歷批次中的每個樣本
        for b in range(spec.shape[0]): # 批次維度
            channel_tensors = []
            # 遍歷每個通道
            for c in range(spec.shape[1]): # 通道維度
                spec_2d = spec[b, c, :, :] # 獲取單個 (H, W) 頻譜圖

                min_val = spec_2d.min()
                max_val = spec_2d.max()

                # 處理所有值都相同的情況，避免除以零
                if max_val - min_val == 0:
                    # 如果所有值都相同，直接返回全零或全一，這裡返回原值（或全零如果想正規化到0）
                    normalized_spec = spec_2d 
                    # 或者如果您希望常量值正規化為0：
                    # normalized_spec = torch.zeros_like(spec_2d)
                else:
                    normalized_spec = (spec_2d - min_val) / (max_val - min_val)
                
                channel_tensors.append(normalized_spec)
            
            # 將所有通道的頻譜圖堆疊回 (C, H, W)
            normalized_tensors.append(torch.stack(channel_tensors, dim=0))
        
        # 將所有批次樣本堆疊回 (B, C, H, W)
        return torch.stack(normalized_tensors, dim=0)
    
    def __getitem__(self, idx):
        # 現在 self.data_list[idx] 已經是完整的音訊檔案路徑了
        label, audio1_path, audio2_path = self.data_list[idx]
        
        # 使用一個輔助函數來處理單個音頻，確保邏輯一致
        audio1_mel = self._process_audio(audio1_path)
        audio2_mel = self._process_audio(audio2_path)

        return audio1_mel, audio2_mel, label

    def _process_audio(self, audio_file_path):
        waveform, sr = librosa.load(audio_file_path, sr=self.sample_rate, mono=True)
        
        # 計算與訓練集完全相同的目標長度
        length = self.frame_num * 160 + 240
        
        if waveform.shape[0] <= length:
            shortage = length - waveform.shape[0]
            final_waveform = torch.nn.functional.pad(waveform, (0, shortage), 'constant', 0)
        else:
            # 隨機裁剪
            start_frame = random.randint(0, waveform.shape[0] - length)
            final_waveform = waveform[start_frame:start_frame + length]
        final_waveform = np.stack([final_waveform], axis=0)
            
        final_waveform = torch.from_numpy(final_waveform).float()

        mel_spec = self.mel_spectrogram(final_waveform)
        mel_spec = torch.log(mel_spec + 1e-6)
        
        return mel_spec
    
    def collate_fn(self, batch):
        # 【核心修改】collate_fn 現在變得和訓練集一樣簡單
        mel1_list, mel2_list, label_list = zip(*batch)
        
        # 直接堆疊，因為 __getitem__ 保證了大小一致
        mels1 = torch.stack(mel1_list)
        mels2 = torch.stack(mel2_list)

        # 應用與訓練集完全相同的處理流程
        def process_batch(mels):
            # resize_mels = F.interpolate(mels, size=(224, 224), mode="bilinear", align_corners=False)
            # norm_mels = self.min_max_normalize(resize_mels)
            # 複製成3通道
            return self.spec_to_rgb(mels)

        final_input_mels1 = process_batch(mels1)
        final_input_mels2 = process_batch(mels2)

        return final_input_mels1, final_input_mels2, torch.tensor(label_list, dtype=torch.long)