import os
import random
import glob
import warnings
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import librosa
import soundfile  # <--- 確保 soundfile 已匯入
from scipy import signal
from tqdm import tqdm
import time
import torch.nn.functional as F
import torchaudio


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
            self.noise_file_paths_by_type = defaultdict(list)
            augment_files = glob.glob(os.path.join(musan_path,'*/*/*.wav'))
            for file in augment_files:
                # 修正了噪音類型提取的邏輯
                noise_type = os.path.basename(os.path.dirname(os.path.dirname(file)))
                self.noise_file_paths_by_type[noise_type].append(file)
            
            self.rir_file_paths = glob.glob(os.path.join(rir_path,'*','*','*.wav'))

        self.data_list = self._load_data_list(dataset_path, data_list_file, self.num_people)

    def _load_data_list(self, dataset_paths, data_list_path, num_people):
        """
        讀取包含 (audio_path, identity_id, age_group_id) 的列表。
        優化：在初始化時就確定每個樣本的具體音頻文件路徑，並確保說話者唯一性。
        """
        data_list_raw = np.load(data_list_path, allow_pickle=True).item()
        
        speaker_to_utterance_keys = defaultdict(list)
        for full_key in data_list_raw.keys():
            speaker_id = full_key[:7]
            speaker_to_utterance_keys[speaker_id].append(full_key)
        
        all_unique_speaker_ids = list(speaker_to_utterance_keys.keys())
        
        sampled_speaker_ids = random.sample(all_unique_speaker_ids, num_people)
        
        selected_utterance_keys = {}
        for speaker_id in sampled_speaker_ids:
            if speaker_id in speaker_to_utterance_keys:
                selected_utterance_keys[speaker_id] = speaker_to_utterance_keys[speaker_id]
            
        utterance_num = sum(len(v) for v in selected_utterance_keys.values())
        print(f"Selected {len(selected_utterance_keys)} unique speakers with a total of {utterance_num} utterances.")

        data_dicts = {}
        for speaker_id in selected_utterance_keys:
            for utterance_key in selected_utterance_keys[speaker_id]:
                data_dicts[utterance_key] = data_list_raw[utterance_key]

        data = []
        speaker_id_map = {}
        next_speaker_int_id = 0
        
        for key, years_old in tqdm(data_dicts.items(), desc=f"Loading data for {len(data_dicts)} unique speakers"):
            speaker_id = key[:7]
            utterance_id = key[8:]
            
            full_audio_dir = None
            for folder in dataset_paths:
                potential_dir = os.path.join(folder, speaker_id, utterance_id)
                if os.path.exists(potential_dir):
                    full_audio_dir = potential_dir
                    break
            
            if full_audio_dir is None:
                continue

            audio_files_in_dir = [
                os.path.join(full_audio_dir, f)
                for f in os.listdir(full_audio_dir) if f.endswith('.m4a') or f.endswith('.wav')
            ]

            if not audio_files_in_dir:
                continue

            if speaker_id not in speaker_id_map:
                speaker_id_map[speaker_id] = next_speaker_int_id
                next_speaker_int_id += 1
            identity_id_int = speaker_id_map[speaker_id]
            
            bins = [20, 30, 40, 50, 60, 70]
            age_group_id = next((i for i, b in enumerate(bins) if years_old <= b), 6)

            target_file = random.choice(audio_files_in_dir)
            data.append((target_file, identity_id_int, age_group_id))

        print(f"Loaded {len(data)} samples from {len(speaker_id_map)} unique speakers for training mode.")
        self.speaker_id_map = speaker_id_map
        
        # ... (寫入檔案的程式碼保持不變)
            
        return data    
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        audio_file_path, identity_id, age_group_id = self.data_list[idx]

        try:
            # 1. 使用 torchaudio 讀取音訊，直接得到 PyTorch 張量
            #    torchaudio 可以直接處理 .m4a, .wav, .mp3 等多種格式 (需安裝 ffmpeg 後端)
            waveform, sr = torchaudio.load(audio_file_path)
        except Exception:
            # 如果檔案損壞或無法讀取，返回一個靜音的張量作為替代
            return torch.zeros(1, self.target_length), identity_id, age_group_id

        # 2. 處理重採樣 (Resampling)
        if sr != self.sample_rate:
            if sr not in self.resamplers:
                self.resamplers[sr] = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = self.resamplers[sr](waveform)

        # 3. 轉為單聲道 (Mono)
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
            waveform = self._apply_augmentation_torch(waveform)

        return waveform, identity_id, age_group_id

    # ... (後續的 _apply_augmentation 和其他輔助方法保持不變) ...
    def _apply_augmentation(self, waveform):
        aug_type = random.randint(0, 3)
        if aug_type == 0: return waveform
        elif aug_type == 1: return self._add_noise(waveform, random.choice(self.noisetypes))
        elif aug_type == 2: return self._apply_reverberation(waveform)
        elif aug_type == 3: return self._change_volume(waveform)
        elif aug_type == 4: return self._change_speed(waveform)
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
    
       
if __name__ == "__main__":
    dataset_path = ['D:/Dataset/VoxCeleb2/vox2_dev_wav/dev/aac']
    data_list_file = 'D:/Dataset/Cross-Age_Speaker_Verification/vox2dev/segment2age.npy'
    musan_path = 'D:/Dataset/musan/musan'
    rir_path = 'D:/Dataset/sim_rir_16k/simulated_rirs_16k'
    
    # calculate time
    start_time = time.time()
    
    train_dataset = Train_loader(
        num_frames=200,
        dataset_path=dataset_path,
        data_list_file=data_list_file,
        musan_path=musan_path,
        rir_path=rir_path,
        augment=False,
        num_people=1000
    )
    
    print("take time to load dataset:", time.time() - start_time)
    for i in range(len(train_dataset)):
        waveform, identity_id, age_group_id = train_dataset[i]
        train_loader = DataLoader(waveform, batch_size=64, shuffle=True)
    end_time = time.time()
    print(f"DataLoader time: {end_time - start_time} seconds")