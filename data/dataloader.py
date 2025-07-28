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
from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms
import torch.nn.functional as F
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from params import param
import matplotlib.pyplot as plt


# 忽略 librosa 可能發出的警告
warnings.filterwarnings(
    "ignore",
    message="PySoundFile failed. Trying audioread instead."
)
warnings.simplefilter("ignore", category=FutureWarning)


class Voxceleb2_dataset(Dataset):
    def __init__(self, num_frames, dataset_path, data_list_file, musan_path, rir_path, augment=False):
        self.augment = augment
        self.sample_rate = 16000  # 假設採樣率為 16000 Hz
        self.frame_num = num_frames

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

    def _load_data_list(self, dataset_paths, data_list_path, num_people=500):
        """
        讀取包含 (audio_path, identity_id, age_group_id) 的列表。
        優化：在初始化時就確定每個樣本的具體音頻文件路徑，並確保說話者唯一性。
        """
        data_list_raw = np.load(data_list_path, allow_pickle=True).item()
        
        # --- 步驟 1: 將所有原始資料按說話者 ID 進行分組 ---
        # 鍵是 speaker_id (idXXXXX)，值是該說話者擁有的所有 utterance keys (idXXXXX/YYYYY)
        speaker_to_utterance_keys = defaultdict(list)
        for full_key in data_list_raw.keys():
            speaker_id = full_key[:7] # 提取說話者 ID
            speaker_to_utterance_keys[speaker_id].append(full_key)
        
        # 隨機打亂speaker_to_utterance_keys裡面的順序，讓每次都能使用不同說話者來做訓練
        
        all_unique_speaker_ids = list(speaker_to_utterance_keys.keys())
        
        # --- 步驟 2: 隨機選擇 num_people 個唯一的說話者 ID ---
        # 如果可用的獨立說話者數量不足，則選擇所有可用的說話者
        sampled_speaker_ids = random.sample(all_unique_speaker_ids, num_people)
        
        # --- 步驟 3: 選擇 N 段語句 ---
        selected_utterance_keys = {}
        # 複製一個dict，但只保留sampled_speaker_ids的說話者的 utterance keys
        for speaker_id in sampled_speaker_ids:
            if speaker_id in speaker_to_utterance_keys:
                selected_utterance_keys[speaker_id] = speaker_to_utterance_keys[speaker_id]
            
        utterance_num = 0
        for id in selected_utterance_keys:
            utterance_num += len(selected_utterance_keys[id])
        
        print(f"Selected {len(selected_utterance_keys)} unique speakers with a total of {utterance_num} utterances.")
        # 根據這些選定的 utterance key 構建 `data_dicts` 子集
        # 這樣 `data_dicts` 就包含了 N 筆資料，且來自 num_people 個不同說話者
        data_dicts = {}
        
        for speaker_id in selected_utterance_keys:
            for utterance_key in selected_utterance_keys[speaker_id]:
                data_dicts[utterance_key] = data_list_raw[utterance_key]

        data = []
        speaker_id_map = {} # 用於將字串 ID 映射到整數 ID
        next_speaker_int_id = 0 # 下一個可用的整數 ID
        
        # 使用 tqdm 顯示進度條，因為這部分可能耗時
        # 迭代的是確保唯一的說話者 ID 的資料子集
        for key, years_old in tqdm(data_dicts.items(), desc=f"Loading data for {len(data_dicts)} unique speakers"):
            speaker_id = key[:7]
            utterance_id = key[8:] # 這是影片/語音段的 ID
            
            # 遍歷可能的數據根路徑，找到實際的音頻資料夾
            full_audio_dir = None
            for folder in dataset_paths: # 注意這裡用 dataset_paths，因為它可能是個列表
                potential_dir = os.path.join(folder, speaker_id, utterance_id)
                if os.path.exists(potential_dir):
                    full_audio_dir = potential_dir
                    break
            
            if full_audio_dir is None:
                warnings.warn(f"Audio directory not found for {speaker_id}/{utterance_id}. Skipping this entry.")
                continue

            # **關鍵優化點：在初始化時遍歷資料夾，找到所有 .m4a/.wav 檔案的路徑**
            m4a_files_in_dir = [
                os.path.join(full_audio_dir, f)
                for f in os.listdir(full_audio_dir) if f.endswith('.m4a') or f.endswith('.wav')
            ]

            if not m4a_files_in_dir:
                warnings.warn(f"No .m4a/.wav files found in {full_audio_dir}. Skipping this utterance.")
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
            target_file = random.choice(m4a_files_in_dir)  # 隨機選擇一個音訊檔案
            data.append((target_file, identity_id_int, age_group_id))

        print(f"Loaded {len(data)} samples from {len(speaker_id_map)} unique speakers for training mode.")
        self.speaker_id_map = speaker_id_map

        return data    
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 現在 self.data_list[idx] 已經是完整的音訊檔案路徑了
        audio_file_path, identity_id, age_group_id = self.data_list[idx]
        
        # 載入音訊檔案
        waveform, sr = librosa.load(audio_file_path, sr=self.sample_rate, mono=True)
        length = self.frame_num * 160 + 240
        
        if waveform.shape[0] <= length:
            shortage = length - waveform.shape[0]
            waveform = np.pad(waveform, (0, shortage), 'wrap')
        start_frame = np.int64(random.random()*(waveform.shape[0]-length))
        waveform = waveform[start_frame:start_frame + length]
        final_waveform = np.stack([waveform], axis=0)
        
        # --- 數據增強 ---
        if self.augment:
            final_waveform = self._apply_augmentation(final_waveform)

        # 將 waveform 轉換為 PyTorch Tensor
        final_waveform = torch.from_numpy(final_waveform).float()

        # 提取 Mel-filterbank energies (channels, samples)
        mel_spec = self.mel_spectrogram(final_waveform)

        # 轉換為論文中期望的形狀 (channels, n_mels, n_frames)
        # mel_spec = mel_spec.squeeze(0).transpose(0, 1)
        # mel_spec = mel_spec.squeeze(0).permute(1, 0)
        
        # 對數 Mel-filterbank energies
        mel_spec = torch.log(mel_spec + 1e-6)

        # # 通常會對特徵進行均值/方差歸一化 (CMVN)
        # # mel_spec = self._apply_cmvn(mel_spec)

        return mel_spec, identity_id, age_group_id
    
    def spec_to_rgb(self, spec):
        """
        將單通道 spectrogram 視覺化成 RGB image。
        spec: 2D array (H, W)
        return: 3D uint8 RGB image: shape (H, W, 3)
        """
        return spec.repeat(1, 3, 1, 1)
    
    def collate_fn(self, batch):
        """
        將批次數據填充到相同長度，並調整為模型期望的形狀。
        """
        # 過濾掉 __getitem__ 返回 None 的樣本 (如果有的話，雖然本次修改應該不會)
        # batch = [item for item in batch if item is not None] # 確保 batch 不包含 None
        
        mels, ident, age = zip(*batch)
        
        # # 獲取每個 Mel 譜的幀數 (長度)
        # lengths = [m.shape[0] for m in mels]
        # maxlen = max(lengths) # 找出批次中最長的 Mel 譜幀數
        
        # # 對所有 Mel 譜進行零填充，使其長度達到 maxlen
        # # (0,0) 對應最後兩個維度 (n_mels)，(0, maxlen-m.shape[0]) 對應第一個維度 (frames)
        # padded = [torch.nn.functional.pad(m, (0,0,0,maxlen-m.shape[0])) for m in mels]
        
        # # 將填充後的 Mel 譜堆疊成一個批次的 Tensor
        # # 原始：torch.stack(padded) 的形狀是 (batch_size, max_frames, n_mels=80)
        # stacked_mels = torch.stack(padded) # (B, Max_Frames, 80)
        
        # # 1. 轉置 Mel 譜，使 n_mels (80) 成為高度 (H)，max_frames 成為寬度 (W)
        # #    從 (B, Max_Frames, 80) 變為 (B, 80, Max_Frames)
        # permuted_mels = stacked_mels.permute(0, 2, 1) 
        
        # # 2. 插入通道維度 (channels=1)，模型通常期望 (B, C, H, W) 格式
        # #    從 (B, 80, Max_Frames) 變為 (B, 1, 80, Max_Frames)
        # final_input_mels = permuted_mels.unsqueeze(1)
        
        mels = torch.stack(mels)  # 將 Mel 譜堆疊成一個批次的 Tensor (B, C, H, W)
        resize_mels = F.interpolate(mels, size=(224, 224), mode="bilinear", align_corners=False)  # 將 Mel 譜轉換為 Tensor 並調整大小
        norm_mels = self.min_max_normalize(resize_mels)  # 對 Mel 譜進行 Min-Max 正規化

        final_input_mels = norm_mels.numpy()  # 如果需要轉換為 NumPy 陣列
        # 將 RGB 圖像轉換為 Tensor
        final_input_mels = torch.tensor(final_input_mels, dtype=torch.float32)
        final_input_mels = self.spec_to_rgb(final_input_mels)  # 將 Mel 譜轉換為 RGB 圖像

        # 將身份和年齡 ID 轉換為 Tensor
        return final_input_mels, torch.tensor(ident, dtype=torch.long), torch.tensor(age, dtype=torch.long)

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
    

class Voxceleb1_dataset(Dataset):
    def __init__(self, dataset_path, data_list_file):
        self.sample_rate = 16000

        # MelSpectrogram 應保持在 CPU，因為輸入波形是 CPU Tensor
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,         # 25ms 幀長
            hop_length=160,    # 10ms hop_size
            n_mels=80          # 80 維 Mel-filterbank energies
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),     # 調整為 224×224，順序為 (height, width)
            transforms.ToTensor(),             # 轉為 Tensor
            # transforms.Normalize(...)        # 如果需要 normalize，也可以放這裡
        ])

        # 加載數據列表，這個必須在增強文件預載入之後，因為 _load_data_list 中會檢查文件存在
        self.data_list = self._load_data_list(dataset_path, data_list_file)
        
    def __len__(self):
        return len(self.data_list)
        
    def _load_data_list(self, dataset_path, data_list_path):
        """
        讀取包含 (audio1_path, audio2_path, label) 的列表。
        """
        with open (data_list_path, 'r') as f:
            # read txt
            data_list_raw = f.readlines()
        data_list_raw = [line.strip().split() for line in data_list_raw]
        data_list_raw = random.sample(data_list_raw, 1000)  # 隨機選擇 1000 條數據
        
        data = []
        
        for line in data_list_raw:
            audio11_path = os.path.join(dataset_path[0], line[1])
            audio12_path = os.path.join(dataset_path[1], line[1])
            
            audio21_path = os.path.join(dataset_path[0], line[2])
            audio22_path = os.path.join(dataset_path[1], line[2])
            
            # 確保音訊檔案存在
            if os.path.exists(audio11_path):
                audio1_path = audio11_path
            else:
                audio1_path = audio12_path
            if os.path.exists(audio21_path):
                audio2_path = audio21_path
            else:
                audio2_path = audio22_path

            data.append((int(line[0]), audio1_path, audio2_path))  # (label, audio1, audio2)
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
        
        audio1_mel = self.turn_to_mel(audio1_path)
        audio2_mel = self.turn_to_mel(audio2_path)

        return audio1_mel, audio2_mel, label

    def turn_to_mel(self, audio_file_path):
        """
        將音訊檔案轉換為 Mel-filterbank energies。
        """
        # **優化點：直接載入確定的音訊檔案，不再需要 os.listdir 或 random.choice**
        waveform, sr = librosa.load(audio_file_path, sr=self.sample_rate, mono=True)
        final_waveform = torch.from_numpy(waveform).float().unsqueeze(0)  # (1, samples)

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
        # mel_spec = self._apply_cmvn(mel_spec)
        return mel_spec
    
    def collate_fn(self, batch):
        """
        將批次數據填充到相同長度，並調整為模型期望的形狀。
        """
        # 過濾掉 __getitem__ 返回 None 的樣本 (如果有的話，雖然本次修改應該不會)
        # batch = [item for item in batch if item is not None] # 確保 batch 不包含 None
        
        mel1, mel2, label = zip(*batch)
        
        mel1 = self.padding_mel(mel1)  # 對 mel1 進行填充
        mel2 = self.padding_mel(mel2)  # 對 mel2 進行填充
        
        final_input_mels1 = F.interpolate(mel1, size=(224, 224), mode="bilinear", align_corners=False)  # 將 Mel 譜轉換為 Tensor 並調整大小
        final_input_mels2 = F.interpolate(mel2, size=(224, 224), mode="bilinear", align_corners=False)

        final_input_mels1 = self.min_max_normalize(final_input_mels1)  # 對 Mel 譜進行 Min-Max 正規化
        final_input_mels2 = self.min_max_normalize(final_input_mels2)

        final_input_mels1 = final_input_mels1.numpy()  # 如果需要轉換為 NumPy 陣列
        final_input_mels2 = final_input_mels2.numpy()

        # 將 RGB 圖像轉換為 Tensor
        final_input_mels1 = torch.tensor(final_input_mels1, dtype=torch.float32)
        final_input_mels1 = self.spec_to_rgb(final_input_mels1)  # 將 Mel 譜轉換為 RGB 圖像

        final_input_mels2 = torch.tensor(final_input_mels2, dtype=torch.float32)
        final_input_mels2 = self.spec_to_rgb(final_input_mels2)

        return final_input_mels1, final_input_mels2, torch.tensor(label, dtype=torch.long)
    
    def padding_mel(self, mel):
        # 獲取每個 Mel 譜的幀數 (長度)
        lengths = [m.shape[0] for m in mel]
        maxlen = max(lengths) # 找出批次中最長的 Mel 譜幀數
        
        # 對所有 Mel 譜進行零填充，使其長度達到 maxlen
        # (0,0) 對應最後兩個維度 (n_mels)，(0, maxlen-m.shape[0]) 對應第一個維度 (frames)
        padded = [torch.nn.functional.pad(m, (0,0,0,maxlen-m.shape[0])) for m in mel]
        
        # 將填充後的 Mel 譜堆疊成一個批次的 Tensor
        # 原始：torch.stack(padded) 的形狀是 (batch_size, max_frames, n_mels=80)
        stacked_mels = torch.stack(padded) # (B, Max_Frames, 80)
        
        # 1. 轉置 Mel 譜，使 n_mels (80) 成為高度 (H)，max_frames 成為寬度 (W)
        #    從 (B, Max_Frames, 80) 變為 (B, 80, Max_Frames)
        permuted_mels = stacked_mels.permute(0, 2, 1) 
        
        # 2. 插入通道維度 (channels=1)，模型通常期望 (B, C, H, W) 格式
        #    從 (B, 80, Max_Frames) 變為 (B, 1, 80, Max_Frames)
        final_input_mels = permuted_mels.unsqueeze(1)
        
        return final_input_mels
        
# Testing
if __name__ == "__main__":
    # 請根據您的實際路徑修改這裡
    dataset_path = ['D:/Dataset/VoxCeleb2/vox2_dev_wav/dev/aac']
    data_list_file = 'D:/Dataset/Cross-Age_Speaker_Verification/vox2dev/segment2age.npy'
    musan_path = 'D:/Dataset/musan/musan'
    rir_path = 'D:/Dataset/sim_rir_16k/simulated_rirs_16k'
    val_dataset_path = ['D:/Dataset/VoxCeleb1/vox1_dev_wav/wav', "D:/Dataset/VoxCeleb1/vox1_test_wav/wav"]
    val_data_list_file = 'D:/Dataset/Cross-Age_Speaker_Verification/trials/Vox-CA20/test.txt'

    print("Initializing training dataset (with augmentation)...")
    train_dataset = Voxceleb2_dataset(
        num_frames=param.NUM_FRAMES,
        dataset_path=dataset_path,
        data_list_file=data_list_file,
        musan_path=musan_path,
        rir_path=rir_path,
        augment=False
    )
    print(f"Training Dataset size: {len(train_dataset)}")

    # print("\nInitializing validation dataset (no augmentation)...")
    # val_dataset = Voxceleb1_dataset(
    #     dataset_path=val_dataset_path,
    #     data_list_file=val_data_list_file,
    # )
    # print(f"Validation Dataset size: {len(val_dataset)}")

    from torch.utils.data import DataLoader
    import time
    import os

    print("\nCreating DataLoader...")
    # pin_memory=True 對於 GPU 訓練至關重要
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0, # 根據你的 CPU 核心數調整
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )

    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=32,
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=True,
    #     collate_fn=val_dataset.collate_fn
    # )

    print(f"Testing DataLoader (first {min(5, len(train_loader))} batches for train_loader):")
    start_time = time.time()
    for i, (mels, ident, age) in enumerate(train_loader):
        print(f"Batch {i+1}: Mel Spec Shape: {mels.shape}, Identity IDs: {ident.shape}, Age Group IDs: {age.shape}")
        if i >= 4: # 只測試前5個批次
            break
    end_time = time.time()
    print(f"Time to load 5 batches (train_loader): {end_time - start_time:.2f} seconds")

    # print(f"\nTesting DataLoader (first {min(5, len(val_loader))} batches for val_loader):")
    # start_time = time.time()
    # for i, (mel1, mel2, label) in enumerate(val_loader):
    #     print(f"Batch {i+1}: Mel Spec Shape: {mel1.shape}, {mel2.shape}, Label: {label.shape}")
    #     if i >= 4: # 只測試前5個批次
    #         break
    # end_time = time.time()
    # print(f"Time to load 5 batches (val_loader): {end_time - start_time:.2f} seconds")

    # print("\nDataLoader test complete.")