import os
import random
import glob
import warnings
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import torchaudio
import soundfile
from scipy import signal
from tqdm import tqdm
import torch.nn.functional as F

# 忽略 librosa 可能發出的警告
warnings.filterwarnings(
    "ignore",
    message="PySoundFile failed. Trying audioread instead."
)
warnings.simplefilter("ignore", category=FutureWarning)


class Train_loader(Dataset):
    def __init__(self, num_frames, dataset_path, data_list_file, musan_path, rir_path, augment=False, num_people=None):
        self.augment = augment
        self.sample_rate = 16000  # 假設採樣率為 16000 Hz
        self.frame_num = num_frames
        self.num_people = num_people
        
        self.target_length = num_frames * 160 + 240

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
        
        if self.augment:
            # 建立噪音資料清單
            self.noiselist = {}
            augment_files = glob.glob(os.path.join(musan_path,'*/*/*.wav'))
            for file in augment_files:
                if file.split('\\')[-3] not in self.noiselist:
                    self.noiselist[file.split('\\')[-3]] = []
                self.noiselist[file.split('\\')[-3]].append(file)
            
            self.rir_file_paths = glob.glob(os.path.join(rir_path,'*','*','*.wav'))

        self.data_list = self._load_data_list(dataset_path, data_list_file, self.num_people)

    def _load_data_list(self, dataset_paths, data_list_path, num_people):
        """
        讀取包含 (audio_path, identity_id, age_group_id) 的列表。
        優化：在初始化時就確定每個樣本的具體音頻文件路徑，並確保說話者唯一性。
        """
        data_list_raw = np.load(data_list_path, allow_pickle=True).item()
        
        # --- 步驟 1: 將所有原始資料按說話者 ID 進行分組 ---
        speaker_to_utterance_keys = defaultdict(list)
        for full_key in data_list_raw.keys():
            speaker_id = full_key[:7] # 提取說話者 ID
            speaker_to_utterance_keys[speaker_id].append(full_key)
        
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

        # data_dicts 就包含了 N 筆資料，且來自 num_people 個不同說話者
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
        try:
            # 使用 torchaudio 讀取音訊，直接得到 PyTorch 張量
            # torchaudio 可以直接處理 .m4a, .wav, .mp3 等多種格式 (需安裝 ffmpeg 後端)
            waveform, sr = torchaudio.load(audio_file_path)
        except Exception:
            # 如果檔案損壞或無法讀取，返回一個靜音的張量作為替代
            return torch.zeros(1, self.target_length), identity_id, age_group_id
        
        # 轉為單聲道 (Mono)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 4. 裁剪或填充至固定長度 (Padding/Cropping)
        current_length = waveform.shape[1]
        if current_length < self.target_length:
            # 使用 torch.nn.functional.pad 進行填充
            waveform = F.pad(waveform, (0, self.target_length - current_length), 'constant', 0)
        else:
            # 使用 torch.randint 進行隨機裁剪
            start_frame = torch.randint(0, current_length - self.target_length + 1, (1,)).item()
            waveform = waveform[:, start_frame:start_frame + self.target_length]
        
        # 5. 應用資料增強 (所有增強函式都已改為處理 PyTorch 張量)
        if self.augment:
            waveform = self._apply_augmentation(waveform)

        return waveform, identity_id, age_group_id

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

        if aug_type == 0:
            return waveform
        elif aug_type == 1:
            return self._add_noise(waveform, random.choice(self.noisetypes))
        elif aug_type == 2:
            return self._apply_reverberation(waveform)
        elif aug_type == 3:
            return self._change_volume(waveform)
        elif aug_type == 4:
            return self._change_speed(waveform)
        
        return waveform
    
    def _add_noise(self, audio, noisecat):
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4) 
        numnoise_val = self.numnoise[noisecat]
        noiselist = random.sample(self.noise_file_paths_by_type[noisecat], random.randint(numnoise_val[0], numnoise_val[1]))
        noises = []
        for noise_path in noiselist:
            noiseaudio, sr = soundfile.read(noise_path)
            length = self.frame_num * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = np.int64(random.random()*(noiseaudio.shape[0]-length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = np.stack([noiseaudio], axis=0)
            noise_db = 10 * np.log10(np.mean(noiseaudio ** 2) + 1e-4)
            noisesnr_val = self.noisesnr[noisecat]
            noisesnr_db = random.uniform(noisesnr_val[0], noisesnr_val[1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr_db) / 10)) * noiseaudio)
        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio
    
    def _apply_reverberation(self, waveform):
        rir_file = random.choice(self.rir_file_paths)
        rir, sr = soundfile.read(rir_file)
        rir = np.expand_dims(rir.astype(float), 0)
        rir = rir / np.sqrt(np.sum(rir**2))
        return signal.convolve(waveform, rir, mode='full')[:, :self.frame_num * 160 + 240]
    
    def _change_volume(self, waveform):
        gain_db = random.uniform(-6.0, 6.0)
        gain = 10 ** (gain_db / 20)
        adjusted = waveform * gain
        max_val = np.max(np.abs(adjusted))
        if max_val > 1.0:
            adjusted = adjusted / max_val
        return adjusted
    
    def _change_speed(self, waveform):
        speed_factor = random.uniform(0.7, 1.3)
        if speed_factor == 1.0: return waveform
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        new_sample_rate = int(self.sample_rate * speed_factor)
        resampler = T.Resample(orig_freq=self.sample_rate, new_freq=new_sample_rate)
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