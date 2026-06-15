import torch
import torch.nn as nn
import torch.nn.functional as F

from model.feature_extractor.ecapa_tdnn import ECAPA_TDNN

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # 使用 ECAPA-TDNN 作為特徵提取器，載入非LoRA預訓練權重
        self.encoder = ECAPA_TDNN(C=1024)
        self.load_parameters("pretrained_models/pretrain.model")
        
    def forward(self, x1, x2, spec_aug):
        # 提取兩個輸入的特徵
        feat1 = self.encoder(x1, aug=spec_aug)
        feat2 = self.encoder(x2, aug=spec_aug)
        
        # 計算兩個特徵之間的距離（例如餘弦距離）
        distance = F.cosine_similarity(feat1, feat2)
        
        return feat1, feat2, distance
    
    def load_parameters(self, path):
        checkpoint = torch.load(path, map_location="cpu")

        if isinstance(checkpoint, dict):
            loaded_state = checkpoint.get("model_state_dict", checkpoint)
        else:
            loaded_state = checkpoint

        if not isinstance(loaded_state, dict):
            raise TypeError(f"不支援的 checkpoint 型別: {type(loaded_state)!r}")

        encoder_state = self.encoder.state_dict()

        for name, param in loaded_state.items():
            if name in ("speaker_loss.weight", "speaker_loss.bias"):
                continue

            candidate_names = [name]
            if name.startswith("module."):
                candidate_names.append(name.replace("module.", "", 1))

            matched_name = None
            for candidate in candidate_names:
                if candidate in encoder_state and encoder_state[candidate].shape == param.shape:
                    matched_name = candidate
                    break

            if matched_name is None:
                continue

            encoder_state[matched_name].copy_(param)

        self.encoder.load_state_dict(encoder_state)