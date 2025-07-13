import torch
import torch.nn as nn
import torch.nn.functional as F

# 根據論文描述，Attentive Statistical Pooling (ASP) 類似於 Global Statistical Pooling (GSP)，
# 但應用於經過注意力加權的特徵圖。
# 它會計算特徵圖在空間維度（頻率和時間）上的均值和標準差，然後將它們拼接起來。
class AttentiveStatisticalPooling(nn.Module):
    def forward(self, x):
        # x 的形狀預期為 (batch_size, channels, freq_bins, time_frames)
        # 例如，從 ResNet34 的 layer4 輸出，可能是 (B, 512, H_feat, W_feat)

        # 計算空間維度上的均值
        # dim=[-2, -1] 表示對倒數第二個和倒數第一個維度（即頻率和時間維度）求均值
        # keepdim=False 則會將這些維度壓縮掉
        mean = x.mean(dim=[-2, -1], keepdim=False)

        # 計算空間維度上的標準差
        # 添加一個很小的常數 (eps) 是為了數值穩定性，防止標準差為零導致除以零
        std = x.std(dim=[-2, -1], keepdim=False) + 1e-5 
        
        # 將均值和標準差在通道維度上拼接起來
        # 輸出形狀將是 (batch_size, channels * 2)
        return torch.cat((mean, std), dim=1)

class ARE_Module(nn.Module):
    def __init__(self, input_channels, output_dim):
        """
        年齡相關提取器模組 (ARE)。
        從高層特徵圖中提取年齡相關的嵌入 z_age。

        Args:
            input_channels (int): 輸入特徵圖的通道數 (例如 ResNet34 layer4 的 512)。
            output_dim (int): 輸出年齡嵌入 z_age 的維度 (例如論文中的 128)。
        """
        super(ARE_Module, self).__init__()

        # 第一部分: σ(x) - 注意力模組 (Attention Module)
        # 根據論文公式 `x ⊙ σ(x)`，σ(x) 應該是一個與 x 空間維度相同的注意力權重圖。
        # 典型的注意力機制會用卷積層來學習這些權重。
        # 這裡我們使用一個簡單的 Conv2d 序列，將通道數壓縮到1，然後用 Sigmoid 歸一化到 [0, 1]。
        self.attention_layer = nn.Sequential(
            # 先用 1x1 卷積減少通道數，提高效率，同時學習跨通道的關係
            nn.Conv2d(input_channels, input_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            # 再用 1x1 卷積將通道數縮減到 1，得到空間注意力圖
            nn.Conv2d(input_channels // 4, 1, kernel_size=1),
            # Sigmoid 函數將輸出值映射到 (0, 1) 範圍，作為注意力權重
            nn.Sigmoid() 
        )

        # 第二部分: pool(...) - Attentive Statistical Pooling (ASP)
        self.asp_pooling = AttentiveStatisticalPooling()

        # 第三部分: fc(...) - 全連接層
        # ASP 的輸出維度是 input_channels * 2 (因為均值和標準差拼接)。
        # 這個全連接層將其映射到所需的 output_dim (例如 128)。
        self.fc_layer = nn.Linear(input_channels * 2, output_dim)

    def forward(self, x):
        """
        前向傳播。

        Args:
            x (torch.Tensor): 高層特徵圖，形狀為 (batch_size, input_channels, freq_bins, time_frames)。
        
        Returns:
            torch.Tensor: 年齡嵌入 z_age，形狀為 (batch_size, output_dim)。
        """
        # 1. 獲取注意力權重 σ(x)
        attention_weights = self.attention_layer(x) # 形狀: (B, 1, F_reduced, T_reduced)

        # 2. 將注意力權重應用到原始特徵圖上 (x ⊙ σ(x))
        # attention_weights 會自動廣播到 x 的所有通道
        x_attended = x * attention_weights # 形狀: (B, input_channels, F_reduced, T_reduced)

        # 3. 執行注意力統計池化
        pooled_features = self.asp_pooling(x_attended) # 形狀: (B, input_channels * 2)

        # 4. 透過全連接層得到最終的年齡嵌入 z_age
        z_age = self.fc_layer(pooled_features) # 形狀: (B, output_dim)

        return z_age