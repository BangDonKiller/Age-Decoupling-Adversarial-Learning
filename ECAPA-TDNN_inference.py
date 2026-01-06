import torch
from model.feature_extractor.speechbrain_model import SpeakerEmbeddingExtractor
# from data.vox1_loader import InferenceDataset
from data.vox2_loader import InferenceDataset
# from data.librispeech_loader import InferenceDataset
# from data.GLOBE_loader import InferenceDataset
# from data.TIMIT_loader import InferenceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from params.param import DEVICE, MODEL_ID, MODEL_list, DATASET_INFO, BATCH_SIZE

DATASET = "VoxCeleb2"
AUDIO_DIR = DATASET_INFO[DATASET]["AUDIO_DIR"]
AUDIO_META_DIR = DATASET_INFO[DATASET]["AUDIO_META_DIR"]
AUDIO_SUFFIX = DATASET_INFO[DATASET]["audio_suffix"]


# ========== 載入模型 ==========
print(f"使用設備: {DEVICE}")
speaker_extractor = SpeakerEmbeddingExtractor(
    model_id=MODEL_ID,
    device=DEVICE
)

# ========== Dataloader ==========
dataset = InferenceDataset(AUDIO_DIR, AUDIO_META_DIR, suffix=AUDIO_SUFFIX)
# dataset = InferenceDataset(AUDIO_DIR, suffix=AUDIO_SUFFIX)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========== Batch 推論 ==========
all_embeddings = []
speaker_ids = []
genders = []
ages = []

with torch.no_grad():
    for waveforms, speaker_id, gender, age in tqdm(loader):
        emb = speaker_extractor(waveforms)
        all_embeddings.append(emb.cpu())
        speaker_ids.extend(speaker_id)
        genders.extend(gender)
        ages.extend(age)
all_embeddings = torch.cat(all_embeddings, dim=0)

# ========== 儲存 embeddings ==========
MODEL = next((k for k, v in MODEL_list.items() if v == MODEL_ID), None)
torch.save({"embeddings": all_embeddings, "speaker_ids": speaker_ids, "genders": genders, "ages": ages}, f"result/{MODEL}/{DATASET}_{MODEL}_embeddings.pt")
print(f"推論完成，共 {len(all_embeddings)} 個向量")


