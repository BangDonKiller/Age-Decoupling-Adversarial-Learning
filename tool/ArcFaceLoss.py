import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    """
    ArcFace 損失模塊。它計算經過角度邊距懲罰和縮放的 logits。
    這些 logits 後續會被傳遞給交叉熵損失函數。

    參考：
    - Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR 2019.
    - 論文中提及的參數 s=64, m=0.2。
    """
    def __init__(self, embedding_size: int, num_classes: int, s: float = 64.0, m: float = 0.20):
        """
        初始化 ArcFaceLoss 模塊。

        Args:
            embedding_size (int): 輸入說話者嵌入的維度 (即論文中的 feature_dim)。
            num_classes (int): 總共的身份類別數量 (即說話者數量)。
            s (float): 縮放因子。將最終的餘弦相似度值縮放，增加類間可分離性。
            m (float): 角度邊距。添加到真實類別的角度懲罰，使學習更困難。
        """
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.embedding_size = embedding_size
        self.num_classes = num_classes

        # 每個身份類別的權重向量 (代表該類別的中心/原型)
        # nn.Parameter 會使其成為模型可學習的參數
        self.weights = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        # 使用 Xavier 均勻分佈初始化權重，有助於訓練穩定性
        nn.init.xavier_uniform_(self.weights)

        # 預先計算 cos(m) 和 sin(m) 以提高效率
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # 計算一個閾值 `th`，用於處理角度 `theta + m` 可能會超過 `pi` 的情況。
        # 如果 `theta + m` > `pi`，那麼 `cos(theta + m)` 的值會變得不那麼負，甚至可能是正數，
        # 這會導致懲罰失效，讓難樣本變「容易」。
        # `th = cos(pi - m)` 是判斷原始 `cos(theta)` 是否過小的閾值。
        self.th = math.cos(math.pi - m)
        
        # 當 `cos(theta)` 小於 `th` 時，將使用 `cos(theta) - self.mm` 作為懲罰項。
        # 這裡的 `self.mm` 是根據 ArcFace 論文中提出的 `sin(m) * m` 懲罰項。
        self.mm = self.sin_m * m 

    def forward(self, input_features: torch.Tensor, labels: torch.Tensor):
        """
        前向傳播。計算經過 ArcFace 處理的 logits。

        Args:
            input_features (torch.Tensor): 輸入的說話者嵌入，形狀為 (batch_size, embedding_size)。
            labels (torch.Tensor): 每個輸入嵌入對應的真實身份標籤，形狀為 (batch_size)。

        Returns:
            torch.Tensor: 經過 ArcFace 處理後的 logits，形狀為 (batch_size, num_classes)。
                          這些 logits 可以直接傳遞給 CrossEntropyLoss。
        """
        # 1. 對輸入特徵和類別權重進行 L2 歸一化
        # 歸一化是 ArcFace 的關鍵步驟，確保餘弦相似度僅受角度影響。
        norm_input_features = F.normalize(input_features) # shape: (batch_size, embedding_size)
        norm_weights = F.normalize(self.weights) # shape: (num_classes, embedding_size)

        # 2. 計算原始的餘弦相似度 (cos(theta))
        # 這是輸入特徵向量與每個類別權重向量之間的點積，由於兩者都已歸一化，所以結果是餘弦值。
        cosine = F.linear(norm_input_features, norm_weights) # shape: (batch_size, num_classes)

        # 3. 提取每個樣本的真實標籤對應的餘弦值 (cos(theta_j) for true class j)
        # `gather` 方法根據 `labels` 中的索引，從 `cosine` 矩陣中選取對應的值。
        # `labels.view(-1, 1)` 將標籤轉換為列向量形狀，以便於 `gather` 操作。
        cosine_theta_j = cosine.gather(1, labels.view(-1, 1)).squeeze() # shape: (batch_size,)

        # 4. 計算 sin(theta_j)
        # 利用三角恆等式 sin^2(x) + cos^2(x) = 1
        # `clamp(0, 1)` 是為了數值穩定性，防止 `sqrt` 內部出現負數 (由於浮點數精度問題)。
        sin_theta_j = torch.sqrt(1.0 - torch.pow(cosine_theta_j, 2)).clamp(0, 1) # shape: (batch_size,)

        # 5. 計算經過角度邊距懲罰後的目標類別餘弦值 (phi_j = cos(theta_j + m))
        # 根據三角函數的和角公式：cos(A+B) = cosAcosB - sinAsinB
        phi_j = cosine_theta_j * self.cos_m - sin_theta_j * self.sin_m # shape: (batch_size,)

        # 6. 處理角度 `theta_j + m` 超過 `pi` 的特殊情況
        # 這是 ArcFace 論文中提出的健壯性處理：
        # 如果原始角度 `theta_j` 已經很大 (即 `cosine_theta_j` 很小，小於 `self.th = cos(pi - m)`)，
        # 則 `theta_j + m` 可能會超過 `pi`。此時 `cos(theta_j + m)` 的值會變得不那麼負，甚至為正，
        # 這會導致邊距懲罰失效，反讓模型更容易將難樣本分對。
        # 為了解決這個問題，當 `cosine_theta_j` 小於 `self.th` 時，
        # 我們強制使用 `cosine_theta_j - self.mm` (`self.mm = sin(m)*m`) 作為懲罰項。
        # `torch.where` 根據條件選擇 `phi_j` 或 `cosine_theta_j - self.mm`。
        final_target_logit_j = torch.where(cosine_theta_j > self.th, phi_j, cosine_theta_j - self.mm) # shape: (batch_size,)

        # 7. 將所有 logits 乘以縮放因子 `s`
        # 這樣做的目的是讓 logits 有足夠大的尺度，便於後續交叉熵損失的優化。
        output = cosine * self.s # shape: (batch_size, num_classes)

        # 8. 將真實類別的 logit 替換為經過邊距懲罰和縮放後的 `final_target_logit_j`
        # 創建一個 one-hot 矩陣來精確定位要替換的元素
        # `labels.view(-1, 1)` 作為索引，指定要替換的列
        # `final_target_logit_j.unsqueeze(1) * self.s` 是要填充的值，`unsqueeze(1)` 匹配 `scatter_` 的維度要求
        output.scatter_(1, labels.view(-1, 1), final_target_logit_j.unsqueeze(1) * self.s)

        return output