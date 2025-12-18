import torch
import torch.nn as nn
import torchaudio.transforms as T
from params import param

# 這是論文的核心，負責計算「表示分離損失」。
# 它內部包含三個小組件來估計互信息 I(h,x), I(h,y), I(h,z)。
class AuxiliaryNetwork(nn.Module):
    def __init__(self, embedding_dim, num_main_classes, num_attribute_classes, input_channels=1, input_size=128):
        """
        初始化輔助網絡。
        Args:
            embedding_dim (int): 提取器輸出的 embedding 維度 (h)。
            num_main_classes (int): 主要任務的類別數 (y, 即說話者數量)。
            num_attribute_classes (int): 要遺忘屬性的類別數 (z, 即年齡段數量)。
            input_channels (int): 原始輸入的通道數 (例如梅爾頻譜圖為1)。
            input_size (int): 原始輸入的尺寸 (假設為正方形，用於解碼器)。
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_channels = input_channels
        self.input_size = input_size

        # 1a. 用於估計 I(h, y) 的輔助分類器 (y 是主要任務標籤)
        # 根據 h 預測 y
        self.y_classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, num_main_classes)
        )

        # 1b. 用於估計 I(h, z) 的輔助分類器 (z 是屬性標籤)
        # 根據 h 預測 z
        self.z_classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, num_attribute_classes)
        )

        # 1c. 用於估計 I(h, x) 的解碼器 (Decoder)
        # 根據 h 重建原始輸入 x。論文中提到用重建誤差來近似 I(h,x)。
        # 這裡我們使用一個簡單的轉置卷積網絡。
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256 * (input_size // 16) * (input_size // 16)),
            nn.ReLU(),
            View((-1, 256, input_size // 16, input_size // 16)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> size*8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> size*4
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> size*2
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1), # -> size
            nn.Sigmoid() # 將輸出壓縮到 [0, 1]
        )
        
        # 用於計算損失的標準
        self.criterion_ce = nn.CrossEntropyLoss()
        # self.ID_criterion_ce = CDWCELoss(num_classes=num_main_classes, alpha=1.0)
        # self.Age_criterion_ce = CDWCELoss(num_classes=num_attribute_classes, alpha=1.0)
        self.criterion_recon = nn.MSELoss()

    def forward(self, h, x, y, z, alpha, beta, gamma):
        """
        計算表示分離損失 L_bar (論文公式3的變體)。
        L_bar = -λ1*I(h,x) + λ2*I(h,y) + λ3*I(h,z)
        我們的目標是最小化 I(h,z) 和 I(h,y)，最大化 I(h,x)。
        因此，損失應該是: L_detach = λ1*I(h,x) - λ2*I(h,y) - λ3*I(h,z)
        其中 I(h,x) 近似為 -ReconstructionError。
        所以 L_detach = -λ1*ReconError - λ2*I(h,y) - λ3*I(h,z)
        為了方便優化器最小化，我們取負號：
        L_detach_to_minimize = λ1*ReconError + λ2*I(h,y) + λ3*I(h,z)

        Args:
            h: 來自提取器的 embedding。
            x: 原始輸入數據 (梅爾頻譜圖)。
            y: 主要任務的標籤 (說話者ID)。
            z: 屬性標籤 (年齡)。
            alpha, beta, gamma: 論文中的超參數。
        Returns:
            torch.Tensor: 表示分離損失。
        """
        # 計算 λ 參數
        lambda1 = alpha * (1 - beta)
        lambda2 = alpha * beta
        lambda3 = alpha * (beta - gamma) # 注意論文公式3有個印刷錯誤，這裡是推導後的正確形式

        # 估計 I(h, x) -> 最小化重建誤差
        # reconstructed_x = self.decoder(h)
        # loss_recon = self.criterion_recon(reconstructed_x, x)
        loss_recon = torch.tensor(0.0).to(h.device)  # 這裡不計算 I(h,x)，因為我們不想保留 x 的信息
        
        # 估計 I(h, y) -> 訓練分類器預測 y
        # H(y|h) 的近似就是交叉熵損失
        y_pred = self.y_classifier(h)
        # loss_y_clf = self.criterion_ce(y_pred, y)
        loss_y_clf = torch.tensor(0.0).to(h.device)  # 這裡不計算 I(h,y)，因為我們不想保留 y 的信息

        # 估計 I(h, z) -> 訓練分類器預測 z
        # H(z|h) 的近似就是交叉熵損失
        z_pred = self.z_classifier(h)
        loss_z_clf = self.criterion_ce(z_pred, z)

        # 組合總的表示分離損失
        # 根據論文，目標是消除屬性信息 (z) 和不必要的任務信息 (y)，保留原始信息 (x)
        # 我們要最小化這個損失，所以符號要對應調整
        detachment_loss = - lambda3 * loss_z_clf
        # detachment_loss = lambda1 * loss_recon - lambda2 * loss_y_clf - lambda3 * loss_z_clf

        return detachment_loss, loss_recon, y_pred, loss_y_clf, z_pred, loss_z_clf


# 輔助類，用於在 nn.Sequential 中改變張量形狀
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)