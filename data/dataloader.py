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
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from params import param
import soundfile
from scipy import signal


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
        
        # 定義噪音類型與對應 SNR 範圍與數量
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
        
        # --- 數據增強相關的初始化和預載入 ---
        if self.augment:
            self.musan_path = musan_path
            self.rir_path = rir_path

            # Step 1: 載入所有噪音和 RIR 檔案的路徑
            self.noise_file_paths_by_type = {}
            self._load_musan_noise_paths(self.musan_path)
            self.rir_file_paths = glob.glob(os.path.join(self.rir_path,'*','*','*.wav'))
        
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
        
        with open("speaker_probability.txt", "w") as f:
            f.write("Speaker, Probability\n")
        with open("age_group_probability.txt", "w") as f:
            f.write("Age Group, Probability\n")
        
        # 計算原資料集中每個說話者的出現次數和年齡組的分佈
        speaker_utterance_count = defaultdict(int)
        for _, identity_id, _ in data:
            speaker_utterance_count[identity_id] += 1
        for speaker_id, count in speaker_utterance_count.items():
            with open("speaker_probability.txt", "a") as f:
                f.write(f"{speaker_id}, {count / len(data):.4f}\n")

        speaker_age_count = defaultdict(int)
        for _, _, age_group_id in data:
            speaker_age_count[age_group_id] += 1
        for age_group_id, count in speaker_age_count.items():
            with open("age_group_probability.txt", "a") as f:
                f.write(f"{age_group_id}, {count / len(data):.4f}\n")
            

        return data    
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        audio_file_path, identity_id, age_group_id = self.data_list[idx]
        
        # 1. 載入音訊檔案，得到 NumPy 陣列
        waveform, sr = librosa.load(audio_file_path, sr=self.sample_rate, mono=True)
        
        # ==================== 核心修正：對調順序 ====================
        # 3. 【後】對增強後的音訊進行固定長度處理
        length = self.frame_num * 160 + 240
        # current_length = final_waveform.shape[1]
        
        if waveform.shape[0] <= length:
            shortage = length - waveform.shape[0]
            final_waveform = torch.nn.functional.pad(waveform, (0, shortage), 'constant', 0)
        else:
            # 隨機裁剪
            start_frame = random.randint(0, waveform.shape[0] - length)
            final_waveform = waveform[start_frame:start_frame + length]
            # final_waveform = final_waveform[:, start_frame:start_frame + length]
        final_waveform = np.stack([final_waveform], axis=0)
            
        # 2. 【先】進行資料增強。這一步驟可能會改變 final_waveform 的長度
        if self.augment:
            final_waveform = self._apply_augmentation(final_waveform)
        # ============================================================
        final_waveform = torch.from_numpy(final_waveform).float() # Shape: (1, num_samples)

        # 4. 提取 Mel-filterbank energies
        mel_spec = self.mel_spectrogram(final_waveform)
        
        # 5. 對數 Mel-filterbank energies
        mel_spec = torch.log(mel_spec + 1e-6)

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
        
        mels, ident, age = zip(*batch)
        
        mels = torch.stack(mels)

        # # 將 RGB 圖像轉換為 Tensor
        final_input_mels = self.spec_to_rgb(mels)  # 將 Mel 譜轉換為 RGB 圖像

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
        aug_type = random.randint(0, 3)
        waveform_on_cpu = waveform

        if aug_type == 0:
            return waveform_on_cpu
        elif aug_type == 1:
            return self._add_noise(waveform_on_cpu, random.choice(self.noisetypes))
        elif aug_type == 2:
            return self._apply_reverberation(waveform_on_cpu)
        elif aug_type == 3:
            return self._change_volume(waveform_on_cpu)
        elif aug_type == 4:
            return self._change_speed(waveform_on_cpu)
        
        return waveform_on_cpu # 確保返回的是 CPU Tensor
    
    
    def _add_noise(self, audio, noisecat):
        """
        加入背景噪音：
        - 根據類型選擇 SNR、數量
        - 從噪音資料集中隨機取出
        - 根據 SNR 調整音量後加入語音
        """
        
        # 計算乾淨語音的平均功率 (DB)
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4) 
        
        # 決定加入的噪音數量和選擇噪音檔案
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noise_file_paths_by_type[noisecat], random.randint(numnoise[0], numnoise[1]))
        
        
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)

            # 定義模型輸入長度
            # 每10ms一幀，一幀的樣本數 = sampling rate (16000Hz) * 0.01s = 160
            # 加上前後各240個樣本的緩衝區，避免邊緣效應(猜測)
            length = self.frame_num * 160 + 240
            
            # 如果噪音長度不足，則重複填充 (不一定每段噪音都有足夠長度可供使用)
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
            # 隨機選擇噪音片段
            start_frame = np.int64(random.random()*(noiseaudio.shape[0]-length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = np.stack([noiseaudio], axis=0)
            noise_db = 10 * np.log10(np.mean(noiseaudio ** 2) + 1e-4)
            
            # 確定目標的信噪比 (SNR)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio
      
    def _apply_reverberation(self, waveform):
        """
        應用混響。RIRs 從預載入的數據集獲取。
        """
        rir_file = random.choice(self.rir_file_paths)
        rir, sr = soundfile.read(rir_file)
        rir = np.expand_dims(rir.astype(float), 0)
        rir = rir / np.sqrt(np.sum(rir**2))  # 正規化        
        return signal.convolve(waveform, rir, mode='full')[:, :self.frame_num * 160 + 240]
    
    def _change_volume(self, waveform):
        """
        隨機改變音量，使用對數 dB 模式。
        """
        # 隨機增益範圍：-6 到 +6 分貝
        gain_db = random.uniform(-6.0, 6.0)
        gain = 10 ** (gain_db / 20)  # dB -> 線性比例

        adjusted = waveform * gain
        # 避免削型：如果溢出就正規化到 -1~1 範圍
        max_val = np.max(np.abs(adjusted))
        if max_val > 1.0:
            adjusted = adjusted / max_val

        return adjusted
    
    def _change_speed(self, waveform):
        """
        隨機改變音頻速度，同時保持音高不變。
        """
        speed_factor = random.uniform(0.7, 1.3) # 隨機選擇速度因子，例如 0.7 到 1.3 倍，避免極端速度
        if speed_factor == 1.0:
            return waveform # 不變速
        
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()

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
    def __init__(self, dataset_path, data_list_file,frame_num, augment, musan_path, rir_path, train):
        self.sample_rate = 16000
        self.frame_num = frame_num
        self.augment = augment
        self.musan_path = musan_path
        self.rir_path = rir_path
        self.train = train

        # MelSpectrogram 應保持在 CPU，因為輸入波形是 CPU Tensor
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,         # 25ms 幀長
            hop_length=160,    # 10ms hop_size
            n_mels=80          # 80 維 Mel-filterbank energies
        )
        
        # 定義噪音類型與對應 SNR 範圍與數量
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}

        # --- 數據增強相關的初始化和預載入 ---
        if self.augment:
            self.musan_path = musan_path
            self.rir_path = rir_path

            # Step 1: 載入所有噪音和 RIR 檔案的路徑
            self.noise_file_paths_by_type = {}
            self._load_musan_noise_paths(self.musan_path)
            self.rir_file_paths = glob.glob(os.path.join(self.rir_path,'*','*','*.wav'))

        # 加載數據列表，這個必須在增強文件預載入之後，因為 _load_data_list 中會檢查文件存在
        self.data_list = self._load_data_list(dataset_path, data_list_file)
        
    def __len__(self):
        return len(self.data_list)
    
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

    def _load_data_list(self, dataset_path, data_list_path):
        """
        讀取包含 (audio1_path, audio2_path, label) 的列表。
        """
        with open(data_list_path, 'r') as f:
            data_list_raw = f.readlines()

        data_list_raw = [line.strip().split() for line in data_list_raw]

        # 如果是測試模式，隨機選擇 10000 條數據來加快速度
        if not self.train:
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
        # 【核心修改】這個函數現在完全模仿訓練集的 __getitem__ 邏輯
        waveform, sr = librosa.load(audio_file_path, sr=self.sample_rate, mono=True)

        # waveform = torch.from_numpy(waveform).float().unsqueeze(0) # Shape: (1, num_samples)
        
        # 計算與訓練集完全相同的目標長度
        length = self.frame_num * 160 + 240
        # current_length = waveform.shape[1]
        
        # if current_length <= length:
        #     shortage = length - current_length
        #     # 使用 PyTorch 的 pad 函式來填充 Tensor
        #     final_waveform = torch.nn.functional.pad(waveform, (0, shortage), 'constant', 0)
        # else:
        #     # 隨機裁剪
        #     start_frame = random.randint(0, current_length - length)
        #     final_waveform = waveform[:, start_frame:start_frame + length]
        
        length = self.frame_num * 160 + 240
        # current_length = final_waveform.shape[1]
        
        if waveform.shape[0] <= length:
            shortage = length - waveform.shape[0]
            final_waveform = torch.nn.functional.pad(waveform, (0, shortage), 'constant', 0)
        else:
            # 隨機裁剪
            start_frame = random.randint(0, waveform.shape[0] - length)
            final_waveform = waveform[start_frame:start_frame + length]
            # final_waveform = final_waveform[:, start_frame:start_frame + length]
        final_waveform = np.stack([final_waveform], axis=0)
            
        if self.augment:
            final_waveform = self._apply_augmentation(final_waveform)
            final_waveform = torch.from_numpy(final_waveform).float()
        else:
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
        aug_type = random.randint(0, 3)
        waveform_on_cpu = waveform

        if aug_type == 0:
            return waveform_on_cpu
        elif aug_type == 1:
            return self._add_noise(waveform_on_cpu, random.choice(self.noisetypes))
        elif aug_type == 2:
            return self._apply_reverberation(waveform_on_cpu)
        elif aug_type == 3:
            return self._change_volume(waveform_on_cpu)
        # elif aug_type == 4:
        #     return self._change_speed(waveform_on_cpu)
        
        return waveform_on_cpu # 確保返回的是 CPU Tensor
    
    
    def _add_noise(self, audio, noisecat):
        """
        加入背景噪音：
        - 根據類型選擇 SNR、數量
        - 從噪音資料集中隨機取出
        - 根據 SNR 調整音量後加入語音
        """
        
        # 計算乾淨語音的平均功率 (DB)
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4) 
        
        # 決定加入的噪音數量和選擇噪音檔案
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noise_file_paths_by_type[noisecat], random.randint(numnoise[0], numnoise[1]))
        
        
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)

            # 定義模型輸入長度
            # 每10ms一幀，一幀的樣本數 = sampling rate (16000Hz) * 0.01s = 160
            # 加上前後各240個樣本的緩衝區，避免邊緣效應(猜測)
            length = self.frame_num * 160 + 240
            
            # 如果噪音長度不足，則重複填充 (不一定每段噪音都有足夠長度可供使用)
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
            # 隨機選擇噪音片段
            start_frame = np.int64(random.random()*(noiseaudio.shape[0]-length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = np.stack([noiseaudio], axis=0)
            noise_db = 10 * np.log10(np.mean(noiseaudio ** 2) + 1e-4)
            
            # 確定目標的信噪比 (SNR)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio
      
    def _apply_reverberation(self, waveform):
        """
        應用混響。RIRs 從預載入的數據集獲取。
        """
        rir_file = random.choice(self.rir_file_paths)
        rir, sr = soundfile.read(rir_file)
        rir = np.expand_dims(rir.astype(float), 0)
        rir = rir / np.sqrt(np.sum(rir**2))  # 正規化        
        return signal.convolve(waveform, rir, mode='full')[:, :self.frame_num * 160 + 240]
    
    def _change_volume(self, waveform):
        """
        隨機改變音量，使用對數 dB 模式。
        """
        # 隨機增益範圍：-6 到 +6 分貝
        gain_db = random.uniform(-6.0, 6.0)
        gain = 10 ** (gain_db / 20)  # dB -> 線性比例

        adjusted = waveform * gain
        # 避免削型：如果溢出就正規化到 -1~1 範圍
        max_val = np.max(np.abs(adjusted))
        if max_val > 1.0:
            adjusted = adjusted / max_val

        return adjusted
        
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