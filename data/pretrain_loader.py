import os
import random
import glob
import warnings
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import librosa
import soundfile
from scipy import signal
from tqdm import tqdm

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
        waveform, sr = librosa.load(audio_file_path, sr=self.sample_rate, mono=True)

        # 對增強後的音訊進行固定長度處理
        length = self.frame_num * 160 + 240
        
        if waveform.shape[0] <= length:
            shortage = length - waveform.shape[0]
            final_waveform = torch.nn.functional.pad(waveform, (0, shortage), 'constant', 0)
        else:
            # 隨機裁剪
            start_frame = random.randint(0, waveform.shape[0] - length)
            final_waveform = waveform[start_frame:start_frame + length]
        final_waveform = np.stack([final_waveform], axis=0)

        if self.augment:
            final_waveform = self._apply_augmentation(final_waveform)

        final_waveform = torch.from_numpy(final_waveform).float() # Shape: (1, num_samples)

        return final_waveform, identity_id, age_group_id

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