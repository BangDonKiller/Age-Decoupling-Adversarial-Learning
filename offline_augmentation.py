import argparse
import random
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from data.vox2_loader import Vox2Dataset


TARGET_SAMPLE_RATE = 16000
TARGET_DURATION_SEC = 2.0
TARGET_NUM_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_DURATION_SEC)
FRAME_SIZE = 160
FRAME_BUFFER = 240

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
	p = argparse.ArgumentParser(description="Offline augmentation generator")
	p.add_argument("--audio_dir", required=True, help="根資料夾，包含 speaker/utt/*.m4a 結構")
	p.add_argument("--meta_csv", required=True, help="metadata csv，與 vox2_loader 相容")
	p.add_argument("--musan_path", required=True, help="musan 資料夾路徑")
	p.add_argument("--rir_path", required=True, help="RIR 資料夾路徑")
	p.add_argument("--methods", nargs="*", default=["rev", "noise_speech", "noise_music", "noise_noise", "mix_speech_music"],
				   help="要產生的增強方法，預設包含 original 加上 5 種增強方式")
	p.add_argument("--output_ext", default=".wav", help="輸出檔案副檔名 (預設 .wav)")
	p.add_argument("--output_dir", required=False, default=None, help="輸出根資料夾，會保留 speaker/utt 結構。若未提供會在 audio_dir 同層建立 <audio_dir>_augmented")
	return p.parse_args()


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


def match_length(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
	current_len = waveform.shape[-1]
	if current_len == target_len:
		return waveform
	if current_len > target_len:
		return waveform[..., :target_len]
	padding = target_len - current_len
	return F.pad(waveform, (0, padding))


def wrap_pad(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
	current_len = waveform.shape[-1]
	if current_len >= target_len:
		return waveform
	repeat_count = (target_len + current_len - 1) // current_len
	repeated = waveform.repeat(1, repeat_count)
	return repeated[..., :target_len]


def random_segment(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
	waveform = wrap_pad(waveform, target_len)
	max_start = waveform.shape[-1] - target_len
	if max_start <= 0:
		return waveform[..., :target_len]
	start = random.randint(0, max_start)
	return waveform[..., start:start + target_len]


def list_audio_files(root: str | Path) -> list[Path]:
	root_path = Path(root)
	patterns = ["*.wav", "*.m4a", "*.flac", "*.mp3"]
	files = []
	for pattern in patterns:
		files.extend(root_path.rglob(pattern))
	return sorted(files)


def load_random_audio(root: str | Path) -> tuple[torch.Tensor, int]:
	files = list_audio_files(root)
	if not files:
		raise FileNotFoundError(f"找不到可用音檔: {root}")
	path = random.choice(files)
	waveform, sample_rate = torchaudio.load(str(path))
	return ensure_mono(waveform), sample_rate


def load_random_noise(noiselist: list[Path], target_len: int, sample_rate: int) -> torch.Tensor:
	noise_path = random.choice(noiselist)
	noise, noise_sr = torchaudio.load(str(noise_path))
	noise = ensure_mono(noise)
	noise = resample_if_needed(noise, noise_sr, sample_rate)
	noise = random_segment(noise, target_len)
	return noise


def load_random_rir(rir_files: list[Path], target_len: int, sample_rate: int) -> torch.Tensor:
	rir_path = random.choice(rir_files)
	rir, rir_sr = torchaudio.load(str(rir_path))
	rir = ensure_mono(rir)
	rir = resample_if_needed(rir, rir_sr, sample_rate)
	rir = match_length(rir, min(rir.shape[-1], target_len))
	return rir


def apply_reverb(audio: torch.Tensor, rir_files: list[Path], sample_rate: int) -> torch.Tensor:
	rir = load_random_rir(rir_files, audio.shape[-1], sample_rate)
	rir = rir / torch.sqrt(torch.sum(rir ** 2)).clamp_min(1e-6)
	convolved = F.conv1d(
		audio.unsqueeze(0),
		rir.flip(-1).unsqueeze(0),
		padding=rir.shape[-1] - 1,
	)
	return convolved.squeeze(0)[..., : audio.shape[-1]]


def apply_noise_mix(audio: torch.Tensor, noiselist: list[Path], sample_rate: int, noisecat: str) -> torch.Tensor:
	noise_cnt = random.randint(NOISE_COUNT[noisecat][0], NOISE_COUNT[noisecat][1])
	selected_noises = random.sample(noiselist, min(noise_cnt, len(noiselist)))

	audio_power = audio.pow(2).mean().clamp_min(1e-8)
	clean_db = 10.0 * torch.log10(audio_power + 1e-4)
	noises = []
	for noise_path in selected_noises:
		noise, noise_sr = torchaudio.load(str(noise_path))
		noise = ensure_mono(noise)
		noise = resample_if_needed(noise, noise_sr, sample_rate)
		noise = random_segment(noise, audio.shape[-1])
		noise_power = noise.pow(2).mean().clamp_min(1e-8)
		noise_db = 10.0 * torch.log10(noise_power + 1e-4)
		noisesnr = random.uniform(NOISE_SNR[noisecat][0], NOISE_SNR[noisecat][1])
		scale = torch.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10.0))
		noises.append(scale * noise)

	if not noises:
		return audio
	noise = torch.sum(torch.stack(noises, dim=0), dim=0)
	return audio + noise


def build_musan_lists(musan_root: str | Path) -> dict[str, list[Path]]:
	noise_files: dict[str, list[Path]] = {key: [] for key in NOISE_TYPES}
	for file_path in Path(musan_root).glob("*/*/*.wav"):
		parts = file_path.parts
		if len(parts) >= 3:
			noise_type = parts[-3]
			if noise_type in noise_files:
				noise_files[noise_type].append(file_path)
	return noise_files


def build_rir_list(rir_root: str | Path) -> list[Path]:
	return list(Path(rir_root).glob("*/*/*.wav"))




def augment_and_save(dataset: Vox2Dataset, methods, output_ext=".wav", output_dir: str = None):
	out_root = Path(output_dir)
	out_root.mkdir(parents=True, exist_ok=True)
	noise_files = build_musan_lists(dataset.musan_path)
	rir_files = build_rir_list(dataset.rir_path)
	if not any(noise_files.values()):
		raise FileNotFoundError(f"MUSAN 資料夾內找不到可用噪音檔: {dataset.musan_path}")
	if not rir_files:
		raise FileNotFoundError(f"RIR 資料夾內找不到可用檔案: {dataset.rir_path}")

	for idx, item in enumerate(tqdm(dataset.datalist, desc="Augmenting", total=len(dataset.datalist))):
		path, speaker_id, gender, age = item
		src_path = Path(path)
		parent = src_path.parent

		try:
			waveform, sr = torchaudio.load(str(src_path))
		except Exception as e:
			print(f"跳過無法讀取的檔案 {src_path}: {e}")
			continue

		audio = ensure_mono(waveform)
		audio = resample_if_needed(audio, sr, TARGET_SAMPLE_RATE)
		audio = ensure_mono(audio)
		audio = random_segment(audio, TARGET_NUM_SAMPLES)
		sr = TARGET_SAMPLE_RATE

		base_name = src_path.stem

		# 計算相對路徑以保留 speaker/utt 結構，並建立對應的輸出資料夾
		try:
			rel = src_path.relative_to(dataset.audio_dir)
		except Exception:
			# fallback: just use speaker/utt from parent
			rel = Path(speaker_id) / src_path.parent.name / src_path.name
		parent_rel = Path(rel).parent
		out_parent = out_root / parent_rel
		out_parent.mkdir(parents=True, exist_ok=True)

		# 儲存未經增強的處理後版本
		orig_out_name = f"{base_name}_orig{output_ext}"
		orig_out_path = out_parent / orig_out_name
		try:
			torchaudio.save(str(orig_out_path), audio, sr)
		except Exception as e:
			print(f"儲存未增強版本失敗 {orig_out_path}: {e}")


		for method in methods:
			if method == "rev":
				aug = apply_reverb(audio.clone(), rir_files, sr)
			elif method == "noise_speech":
				aug = apply_noise_mix(audio.clone(), noise_files["speech"], sr, "speech")
			elif method == "noise_music":
				aug = apply_noise_mix(audio.clone(), noise_files["music"], sr, "music")
			elif method == "noise_noise":
				aug = apply_noise_mix(audio.clone(), noise_files["noise"], sr, "noise")
			elif method == "mix_speech_music":
				aug = apply_noise_mix(audio.clone(), noise_files["speech"], sr, "speech")
				aug = apply_noise_mix(aug, noise_files["music"], sr, "music")
			else:
				print(f"Unknown method {method}, skipping")
				continue

			aug = match_length(ensure_mono(aug), TARGET_NUM_SAMPLES)

			# 確保 shape 為 [channels, T]
			if aug.ndim == 1:
				aug = aug.unsqueeze(0)

			out_name = f"{base_name}_aug_{method}{output_ext}"
			out_path = out_parent / out_name

			try:
				torchaudio.save(str(out_path), aug, sr)
			except Exception as e:
				print(f"儲存失敗 {out_path}: {e}")
				continue


if __name__ == "__main__":
	args = parse_args()

	# 確保工作目錄可 import data.vox2_loader
	repo_root = Path(__file__).resolve().parent
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

	dataset = Vox2Dataset(
		audio_dir=args.audio_dir,
		audio_meta_dir=args.meta_csv,
		musan_path=args.musan_path,
		rir_path=args.rir_path,
		augment=False,
		suffix=".m4a",
	)

	# 如果使用者沒提供 output_dir，建立預設資料夾在 audio_dir 同層
	if args.output_dir is None:
		audio_dir_path = Path(args.audio_dir)
		default_out = audio_dir_path.parent / f"{audio_dir_path.name}_augmented"
		default_out.mkdir(parents=True, exist_ok=True)
		print(f"未提供 --output_dir，使用預設: {default_out}")
		args.output_dir = str(default_out)

	augment_and_save(dataset, args.methods, output_ext=args.output_ext, output_dir=args.output_dir)
