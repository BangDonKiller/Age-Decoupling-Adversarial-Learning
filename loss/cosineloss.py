import torch
import torch.nn as nn
import torch.nn.functional as F

class RelaxedCosineLoss(nn.Module):
    def __init__(self, m_pos=0.6, m_neg=0.2):
        super().__init__()
        self.m_pos = m_pos  # 同人及格線：大於 0.6 就不算 Loss
        self.m_neg = m_neg  # 異人及格線：小於 0.2 就不算 Loss

    def forward(self, feat1, feat2, same_label):
        """
        same_label: 1 代表同人，0 代表異人 (不用轉成 -1)
        """
        # 確保 L2 歸一化
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)

        # 計算 Cosine Similarity
        cos_sim = torch.sum(feat1 * feat2, dim=1)

        # 同人 Loss: max(0, 0.6 - cos_sim)
        loss_pos = F.relu(self.m_pos - cos_sim)
        # 異人 Loss: max(0, cos_sim - 0.2)
        loss_neg = F.relu(cos_sim - self.m_neg)

        # 根據標籤給 Loss
        loss = torch.where(same_label > 0.5, loss_pos, loss_neg)
        return loss.mean()
    
class SmoothCosineLoss(nn.Module):
    def __init__(self, scale=10.0, margin=0.2):
        super().__init__()
        self.scale = scale    # 放大學習梯度
        self.margin = margin  # 分隔正負樣本的中心邊界

    def forward(self, feat1, feat2, same_label):
        """
        same_label: 1 代表同人，0 代表異人 (不用轉成 -1)
        """
        # feat1 = F.normalize(feat1, p=2, dim=1)
        # feat2 = F.normalize(feat2, p=2, dim=1)
        cos_sim = torch.sum(feat1 * feat2, dim=1) # 分數介於 -1 到 1

        # 將標籤轉換為 1 (同人) 和 -1 (異人)
        y = torch.where(same_label > 0.5, 1.0, -1.0)

        # 核心：使用 LogSumExp (類似 Softplus) 進行平滑優化
        # 如果是同人(y=1)：希望 cos_sim 大於 margin，此時括號內為負，Loss 趨近 0
        # 如果是異人(y=-1)：希望 cos_sim 小於 margin，此時括號內為負，Loss 趨近 0
        loss = torch.log(1 + torch.exp(-self.scale * y * (cos_sim - self.margin)))

        return loss.mean()