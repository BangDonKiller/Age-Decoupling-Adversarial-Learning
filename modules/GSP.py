import torch
import torch.nn as nn

class GlobalStatisticalPooling(nn.Module):
    def __init__(self):
        super(GlobalStatisticalPooling, self).__init__()

    def forward(self, x):
        # x 的形狀是 (batch_size, channels, height, width_frames)
        # 例如：(B, 512, 3, W_final)
        # 1. 沿著時間維度 (dim=-1) 計算均值和標準差
        # 輸出形狀： (batch_size, channels, height)
        mean = x.mean(dim=-1)
        std = x.std(dim=-1)

        # 2. 將均值和標準差在新的維度上拼接起來
        # 輸出形狀： (batch_size, 2 * channels, height)
        pooled_features = torch.cat((mean, std), dim=1)
        
        # 3. 將這些pooled features展平為一個向量，送入全連接層
        # 輸出形狀： (batch_size, 2 * channels * height)
        z = pooled_features.view(x.size(0), -1) 
        
        return z