import torch
import torch.nn as nn

class GlobalStatisticalPooling(nn.Module):
    def __init__(self):
        super(GlobalStatisticalPooling, self).__init__()

    def forward(self, x):
        # x 的形狀是 (batch_size, channels, height, width_frames)
        # 1. 沿著時間維度 (dim=-1) 計算均值和標準差
        mean = x.mean(dim=[-2, -1], keepdim=False)
        std = x.std(dim=[-2, -1], keepdim=False) + 1e-6  # 防止除以零
        
        # 2. 將均值和標準差連接起來
        # z 的形狀是 (batch_size, channels * 2)
        z = torch.cat((mean, std), dim=1)
        
        return z