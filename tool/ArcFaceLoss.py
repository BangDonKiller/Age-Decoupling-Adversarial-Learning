'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from tool.eval_metric import accuracy

class AAMsoftmax(nn.Module):
    """
    AAMsoftmax (Additive Angular Margin Softmax) 損失函數的實現。
    這是一種用於深度學習中，特別是語音識別或說話人識別任務的損失函數，
    旨在增加類間可分性並減少類內變異性。

    它在標準 Softmax 的基礎上，引入了一個角度邊距 (angular margin) 'm'
    和一個縮放因子 (scale factor) 's'，以增強特徵的區分能力。
    """
    def __init__(self, n_class, m, s):
        """
        初始化 AAMsoftmax 損失函數。

        Args:
            n_class (int): 分類類別的數量（例如，說話人的數量）。
            m (float): 角度邊距 (angular margin)。它增加了不同類別之間的角度距離。
                       值越大，類別邊界越清晰，但可能使訓練更困難。
            s (float): 縮放因子 (scale factor)。它擴大了特徵的尺度，使模型對分類邊界更敏感。
                       值越大，特徵的區分能力越強。
        """
        super(AAMsoftmax, self).__init__()
        self.m = m  # 角度邊距
        self.s = s  # 縮放因子

        # 可訓練的權重矩陣，每個類別對應一個嵌入向量
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 128), requires_grad=True)
        
        # 交叉熵損失函數，用於計算最終的損失
        self.ce = nn.CrossEntropyLoss()
        
        # 使用 Xavier 正態分佈初始化權重，有助於保持訓練過程中的梯度穩定性
        nn.init.xavier_normal_(self.weight, gain=1)

        # 預先計算一些常用值，以提高前向傳播的效率
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        
        # 閾值，用於處理角度超出 pi-m 範圍的情況，以確保角度單調性
        self.th = math.cos(math.pi - self.m)
        
        # 修正因子，用於在角度超出閾值時應用
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        """
        執行 AAMsoftmax 損失函數的前向傳播計算。

        Args:
            x (torch.Tensor): 輸入的特徵嵌入向量。形狀通常為 (batch_size, embedding_dim)
            label (torch.Tensor, optional): 每個特徵向量對應的真實類別標籤。
                                            形狀通常為 (batch_size,)。
                                            在訓練時必須提供，預測時可為 None。

        Returns:
            tuple: 包含兩個元素的元組。
                   - loss (torch.Tensor): 計算得到的 AAMsoftmax 損失值。
                   - prec1 (torch.Tensor): 模型的 Top-1 準確度。
        """
        # 1. 特徵和權重歸一化 (L2 normalization)
        # 使得特徵向量和權重向量都位於單位球面上，這樣它們的點積就是它們之間夾角的餘弦值。
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        # 2. 計算 sin(theta)
        # 利用 sin^2(theta) + cos^2(theta) = 1 的關係，從餘弦值計算正弦值。
        # clamp(0, 1) 用於防止浮點數精度問題導致的負值。
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        
        # 3. 角度邊距的應用 (cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m))
        # 這是 AAMsoftmax 的核心，它將原始餘弦值 (cos(theta)) 替換為 (cos(theta + m)) 的變形，增加了類別之間的角度距離。
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # 4. 邊界修正：當角度 (theta) 大於 pi - m 時的處理
        # 為了保證角度的單調性，當 theta + m 跨越 pi 時，我們需要特殊的處理。
        # 如果 cos(theta) - th > 0 (即 theta < pi - m)，則使用 phi；
        # 否則 (即 theta >= pi - m)，使用 cosine - mm。這可以防止邊界附近的不穩定性。
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        
        # 5. 構建 One-Hot 編碼標籤
        # 創建一個與 cosine 形狀相同的零張量，然後在真實標籤對應的位置設置為 1。
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        
        # 6. 將邊距僅應用於正確類別的得分
        # 對於正確的類別（由 one_hot 標示為 1），使用 phi (帶邊距的餘弦值)；
        # 對於錯誤的類別（由 one_hot 標示為 0），保持原始的餘弦值。
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 7. 縮放輸出
        # 將最終的得分乘以縮放因子 's'，以擴大分類空間，使得模型對分類邊界更敏感，有助於更快的收斂和更好的區分度。
        output = output * self.s
        
        # 8. 計算交叉熵損失
        # loss = self.ce(output, label)
        
        # 9. 計算 Top-1 準確度
        # prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        # return loss, prec1
        return output