import argparse
import random
from pathlib import Path
import sys
import concurrent.futures

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

TARGET_SAMPLE_RATE = 16000
TARGET_DURATION_SEC = 2.0
TARGET_NUM_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_DURATION_SEC)
ALLOWED_AUDIO_EXTS = {".wav", ".m4a", ".flac", ".mp3"}

NOISE_TYPES = ["noise", "speech", "music"]
NOISE_SNR = {
    "noise": (0.0, 15.0),
    "speech": (13.0, 20.0),
    "music": (5.0, 15.0),
}
NOISE_COUNT = {
    "noise": (1, 1),
    "speech": (3, 8),
    "music": (1, 1),
}

def parse_args():
    p = argparse.ArgumentParser(description="GPU-Accelerated Offline augmentation generator")
    p.add_argument("--audio_dir", required=True, help="根資料夾，包含 speaker/utt/*.m4a 結構")
    p.add_argument("--meta_csv", required=True, help="metadata csv")
    p.add_argument("--musan_path", required=True, help="musan 資料夾路徑")
    p.add_argument("--rir_path", required=True, help="RIR 資料夾路徑")
    p.add_argument("--methods", nargs="*", default=["rev", "noise_speech", "noise_music", "noise_noise", "mix_speech_music"])
    p.add_argument("--output_ext", default=".wav", help="輸出檔案副檔名 (預設 .wav)")
    p.add_argument("--output_dir", required=False, default=None)
    p.add_argument("--batch_size", type=int, default=128, help="GPU 批次大小")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader 讀取檔案的 CPU 核心數")
    return p.parse_args()

# ==========================================
# 基礎輔助函數 (CPU 執行)
# ==========================================
def ensure_mono(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] > 1:
        tensor = torch.mean(tensor, dim=0, keepdim=True)
    return tensor

def resample_if_needed(waveform: torch.Tensor, source_sr: int, target_sr: int = 16000) -> torch.Tensor:
    if source_sr == target_sr:
        return waveform
    return T.Resample(orig_freq=source_sr, new_freq=target_sr)(waveform)

def wrap_pad(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    current_len = waveform.shape[-1]
    if current_len >= target_len:
        return waveform
    repeat_count = (target_len + current_len - 1) // current_len
    return waveform.repeat(1, repeat_count)[..., :target_len]

def random_segment(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    waveform = wrap_pad(waveform, target_len)
    max_start = waveform.shape[-1] - target_len
    if max_start <= 0:
        return waveform[..., :target_len]
    start = random.randint(0, max_start)
    return waveform[..., start:start + target_len]

# ==========================================
# Dataset 定義 (負責 I/O 與前處理)
# ==========================================
class AudioDataset(Dataset):
    def __init__(self, file_paths: list[Path], audio_root: Path):
        self.file_paths = file_paths
        self.audio_root = audio_root

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        src_path = self.file_paths[idx]
        try:
            waveform, sr = torchaudio.load(str(src_path))
            audio = ensure_mono(waveform)
            audio = resample_if_needed(audio, sr, TARGET_SAMPLE_RATE)
            audio = random_segment(audio, TARGET_NUM_SAMPLES)
        except Exception as e:
            # 發生錯誤時回傳全零的 dummy tensor，後續可略過
            print(f"Read Error {src_path}: {e}")
            audio = torch.zeros(1, TARGET_NUM_SAMPLES)

        # 處理輸出相對路徑
        try:
            rel_dir = src_path.relative_to(self.audio_root).parent
        except ValueError:
            parts = src_path.parts
            rel_dir = Path(parts[-3]) / parts[-2]

        return audio, src_path.stem, str(rel_dir)

# ==========================================
# 批次 GPU 增強函數
# ==========================================
def load_and_prep_noise(noiselist: list[Path], noisecat: str) -> torch.Tensor:
    """在 CPU 上隨機載入指定數量的噪音並疊加 (為單一音檔準備)"""
    noise_cnt = random.randint(NOISE_COUNT[noisecat][0], NOISE_COUNT[noisecat][1])
    selected = random.sample(noiselist, min(noise_cnt, len(noiselist)))
    
    noises = []
    for path in selected:
        n, sr = torchaudio.load(str(path))
        n = ensure_mono(n)
        n = resample_if_needed(n, sr, TARGET_SAMPLE_RATE)
        n = random_segment(n, TARGET_NUM_SAMPLES)
        # 單獨計算能量比例以混音
        n_power = n.pow(2).mean().clamp_min(1e-8)
        n_db = 10.0 * torch.log10(n_power + 1e-4)
        snr = random.uniform(NOISE_SNR[noisecat][0], NOISE_SNR[noisecat][1])
        # 先以 0dB 為基準記錄，稍後在 GPU 會根據乾淨音檔的 dB 進行縮放
        scale = torch.sqrt(10 ** ((- n_db - snr) / 10.0)) 
        noises.append(scale * n)
    
    if not noises:
        return torch.zeros(1, TARGET_NUM_SAMPLES)
    return torch.sum(torch.stack(noises, dim=0), dim=0)

def apply_noise_mix_batched(audio_b: torch.Tensor, noiselist: list[Path], noisecat: str, device: torch.device) -> torch.Tensor:
    """GPU 批次噪音混合"""
    B, C, T_len = audio_b.shape
    
    # 1. 在 CPU 預備這批次的噪音
    noise_cpu_list = [load_and_prep_noise(noiselist, noisecat) for _ in range(B)]
    noise_b = torch.stack(noise_cpu_list, dim=0).to(device) # [B, 1, T]

    # 2. 在 GPU 進行平行運算 (計算乾淨訊號強度，並縮放噪音)
    audio_power = audio_b.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8)
    clean_db = 10.0 * torch.log10(audio_power + 1e-4)
    scale = torch.sqrt(10 ** (clean_db / 10.0))
    
    return audio_b + (noise_b * scale)

def apply_reverb_batched(audio_b: torch.Tensor, rir_files: list[Path], device: torch.device) -> torch.Tensor:
    """GPU 批次殘響處理 (使用 Grouped Convolution 大幅加速)"""
    B, C, T_len = audio_b.shape
    
    # 1. 在 CPU 讀取 B 個 RIR
    rir_tensors = []
    max_len = 0
    for _ in range(B):
        r, sr = torchaudio.load(str(random.choice(rir_files)))
        r = ensure_mono(r)
        r = resample_if_needed(r, sr, TARGET_SAMPLE_RATE)
        rir_tensors.append(r)
        max_len = max(max_len, r.shape[-1])
    
    # 2. Padding 使 RIR 等長，並移至 GPU
    padded_rirs = []
    for r in rir_tensors:
        pad_size = max_len - r.shape[-1]
        padded_rirs.append(F.pad(r, (0, pad_size)))
    
    # weight shape: [B, 1, max_len]
    weight = torch.cat(padded_rirs, dim=0).unsqueeze(1).to(device)
    
    # RIR 正規化 (GPU 運算)
    weight = weight / torch.sqrt(torch.sum(weight ** 2, dim=-1, keepdim=True)).clamp_min(1e-6)
    weight_flipped = weight.flip(-1) # 卷積需要的翻轉
    
    # 3. 準備 Grouped Convolution 
    # audio 必須 reshape 成 [1, B, T] 才能讓每個 batch item 對應不同的 RIR
    x = audio_b.transpose(0, 1) 
    
    convolved = F.conv1d(
        x, 
        weight_flipped, 
        groups=B, 
        padding=max_len - 1
    )
    
    # 轉回 [B, 1, T] 並裁切至原始長度
    convolved = convolved.transpose(0, 1)[..., :T_len]
    return convolved

# ==========================================
# 解析與主要流程
# ==========================================
def build_musan_lists(musan_root: str | Path) -> dict[str, list[Path]]:
    noise_files = {key: [] for key in NOISE_TYPES}
    for file_path in Path(musan_root).glob("*/*/*.wav"):
        if len(file_path.parts) >= 3:
            noise_type = file_path.parts[-3]
            if noise_type in noise_files:
                noise_files[noise_type].append(file_path)
    return noise_files

def build_utterance_file_map(audio_dir: str | Path, meta_csv: str | Path) -> list[Path]:
    audio_root = Path(audio_dir)
    df = pd.read_csv(meta_csv, sep=",", header=0)
    path_columns = [col for col in ("speaker1_path", "speaker2_path") if col in df.columns]
    
    seen_rel_paths = set()
    for col in path_columns:
        for raw_path in df[col].dropna().astype(str):
            rel = raw_path.strip().replace("\\", "/")
            if rel: seen_rel_paths.add(rel)

    valid_files = []
    for rel_path in seen_rel_paths:
        parts = Path(rel_path).parts
        if len(parts) < 2: continue
        utt_root = audio_root / parts[0] / parts[1]
        valid_files.extend([f for f in utt_root.rglob("*") if f.is_file() and f.suffix.lower() in ALLOWED_AUDIO_EXTS])
    
    return list(set(valid_files))

def async_save(tensor: torch.Tensor, path: Path, sr: int):
    """背景存檔任務"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), tensor, sr)

def augment_and_save(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 啟動加速模式，使用裝置: {device}")

    audio_root = Path(args.audio_dir)
    out_root = Path(args.output_dir)
    
    noise_files = build_musan_lists(args.musan_path)
    rir_files = list(Path(args.rir_path).glob("*/*/*.wav"))
    
    all_files = build_utterance_file_map(audio_root, args.meta_csv)
    print(f"📦 找到 {len(all_files)} 個待處理音檔")

    # 建立 Dataset 與 DataLoader
    dataset = AudioDataset(all_files, audio_root)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    # 啟動 ThreadPool 負責背景存檔
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    futures = []

    for batch_idx, (audio_b, base_names, rel_dirs) in enumerate(tqdm(dataloader, desc="GPU Augmenting")):
        # audio_b shape: [B, 1, T]
        B = audio_b.shape[0]
        audio_b = audio_b.to(device)

        # 1. 存儲未增強的原始檔案 (需丟回 CPU 存檔)
        cpu_orig = audio_b.cpu()
        for i in range(B):
            if cpu_orig[i].sum() == 0: continue # 略過讀取失敗的檔
            out_p = out_root / rel_dirs[i] / f"{base_names[i]}{args.output_ext}"
            futures.append(executor.submit(async_save, cpu_orig[i], out_p, TARGET_SAMPLE_RATE))

        # 2. 進行增強
        for method in args.methods:
            if method == "rev":
                aug_b = apply_reverb_batched(audio_b, rir_files, device)
            elif method == "noise_speech":
                aug_b = apply_noise_mix_batched(audio_b, noise_files["speech"], "speech", device)
            elif method == "noise_music":
                aug_b = apply_noise_mix_batched(audio_b, noise_files["music"], "music", device)
            elif method == "noise_noise":
                aug_b = apply_noise_mix_batched(audio_b, noise_files["noise"], "noise", device)
            elif method == "mix_speech_music":
                aug_b = apply_noise_mix_batched(audio_b, noise_files["speech"], "speech", device)
                aug_b = apply_noise_mix_batched(aug_b, noise_files["music"], "music", device)
            else:
                continue
            
            # 將運算完的批次移回 CPU 準備存檔
            aug_b_cpu = aug_b.cpu()
            for i in range(B):
                if cpu_orig[i].sum() == 0: continue
                out_name = f"{base_names[i]}_aug_{method}{args.output_ext}"
                out_p = out_root / rel_dirs[i] / out_name
                futures.append(executor.submit(async_save, aug_b_cpu[i], out_p, TARGET_SAMPLE_RATE))

    print("💾 正在等待背景存檔寫入完成...")
    concurrent.futures.wait(futures)
    print("✅ 全部處理完成！")

if __name__ == "__main__":
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = str(Path(args.audio_dir).parent / f"{Path(args.audio_dir).name}_augmented")
    
    augment_and_save(args)