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
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        distance = F.cosine_similarity(feat1, feat2, dim=1)
        
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

        loaded_weights = []
        skipped_weights = []
        unmatched_weights = []

        for name, param in loaded_state.items():
            if name in ("speaker_loss.weight", "speaker_loss.bias"):
                skipped_weights.append(name)
                continue

            candidate_names = [name]
            
            # 處理 module. 前綴（DataParallel）
            if name.startswith("module."):
                candidate_names.append(name.replace("module.", "", 1))
            
            # 處理 speaker_encoder. 前綴（預訓練模型中使用）
            if name.startswith("speaker_encoder."):
                candidate_names.append(name.replace("speaker_encoder.", "", 1))

            matched_name = None
            for candidate in candidate_names:
                if candidate in encoder_state and encoder_state[candidate].shape == param.shape:
                    matched_name = candidate
                    break

            if matched_name is None:
                unmatched_weights.append(name)
                continue

            encoder_state[matched_name].copy_(param)
            loaded_weights.append(matched_name)

        self.encoder.load_state_dict(encoder_state)
        
        # 打印載入摘要
        # print("=" * 60)
        # print(f"預訓練權重載入摘要 ({path})")
        # print("=" * 60)
        # print(f"✓ 成功載入 {len(loaded_weights)} 個權重:")
        # for w in sorted(loaded_weights):
        #     print(f"  - {w}")
        
        # if skipped_weights:
        #     print(f"\n⊘ 跳過 {len(skipped_weights)} 個權重 (不需要):")
        #     for w in sorted(skipped_weights):
        #         print(f"  - {w}")
        
        # if unmatched_weights:
        #     print(f"\n⚠ 無法匹配 {len(unmatched_weights)} 個權重:")
        #     for w in sorted(unmatched_weights)[:10]:  # 只顯示前 10 個
        #         print(f"  - {w}")
        #     if len(unmatched_weights) > 10:
        #         print(f"  ... 以及其他 {len(unmatched_weights) - 10} 個權重")
        
        # print("=" * 60)