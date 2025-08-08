import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from params import param
from dlordinal.losses.cdw import CDWCELoss

# --- 1. 輔助網絡 (Auxiliary Network) ---
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
        reconstructed_x = self.decoder(h)
        loss_recon = self.criterion_recon(reconstructed_x, x)
        
        # 估計 I(h, y) -> 訓練分類器預測 y
        # H(y|h) 的近似就是交叉熵損失
        y_pred = self.y_classifier(h)
        loss_y_clf = self.criterion_ce(y_pred, y)

        # 估計 I(h, z) -> 訓練分類器預測 z
        # H(z|h) 的近似就是交叉熵損失
        z_pred = self.z_classifier(h)
        loss_z_clf = self.criterion_ce(z_pred, z)

        # 組合總的表示分離損失
        # 根據論文，目標是消除屬性信息 (z) 和不必要的任務信息 (y)，保留原始信息 (x)
        # 我們要最小化這個損失，所以符號要對應調整
        detachment_loss = -lambda1 * (-loss_recon) + lambda2 * loss_y_clf + lambda3 * loss_z_clf
        
        return detachment_loss, loss_recon, y_pred, loss_y_clf, z_pred, loss_z_clf


# 輔助類，用於在 nn.Sequential 中改變張量形狀
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)

# --- 2. 模型組件 ---
# 將 ShuffleNet 分割為提取器和分類器

class RepresentationDetachmentExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 載入預訓練的 ShuffleNet v2
        shufflenet = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        # 移除原始的分類頭
        self.features = nn.Sequential(*list(shufflenet.children())[:-1])

    def forward(self, x):
        # 提取特徵
        x = self.features(x)
        # 全局平均池化得到 embedding
        x = x.mean([2, 3]) 
        return x

class MainTaskClassifier(nn.Module):
    def __init__(self, embedding_dim, num_main_classes):
        super().__init__()
        # 簡單的線性分類器用於聲紋識別
        self.fc = nn.Linear(embedding_dim, num_main_classes)

    def forward(self, h):
        return self.fc(h)
    
# --- 3. 完整的屬性遺忘模型 ---
class AttributeUnlearningModel(nn.Module):
    def __init__(self, num_main_classes, num_attribute_classes, input_channels=1, input_size=128):
        super().__init__()
        self.extractor = RepresentationDetachmentExtractor()
        # ShuffleNet v2 x1.0 輸出的 embedding 維度是 1024
        embedding_dim = 1024 
        self.classifier = MainTaskClassifier(embedding_dim, num_main_classes)
        self.aux_network = AuxiliaryNetwork(embedding_dim, num_main_classes, num_attribute_classes, input_channels, input_size)

    def forward(self, x):
        h = self.extractor(x)
        main_task_output = self.classifier(h)
        return main_task_output, h


# if __name__ == "__main__":
    # # 測試模型
    # # feature_dim 是論文中提到的 128 維說話者嵌入
    # model = ADAL_Model(feature_dim=128, age_classes=7, identity_classes=1000)
    
    # # 根據論文，輸入應該是 80 維的 Mel-filterbank energies
    # # 假設一個音頻有 300 幀 (約 3 秒)
    # batch_size = 64
    # num_frames = 300 
    
    # # 輸入張量形狀： (batch_size, channels=1, height=80, width=num_frames)
    # audio_sample = torch.randn(batch_size, 1, 80, num_frames)
    
    # # 將模型移到 CPU 或 GPU (如果可用)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # audio_sample = audio_sample.to(device)
    
    # dummy_identity_labels = torch.randint(0, 1000, (batch_size,))
    # dummy_age_labels = torch.randint(0, 7, (batch_size,)) # 0-6 對應 7 個年齡組
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # audio_sample = audio_sample.to(device)
    # dummy_identity_labels = dummy_identity_labels.to(device)
    # dummy_age_labels = dummy_age_labels.to(device)

    # # 打印模型結構，檢查層的輸出形狀
    # print(model) 

    # features, z, z_age, identity_logits, age_logits_from_age, age_logits_from_id_grl = model(audio_sample, dummy_identity_labels)
    
    # print("Input audio_sample shape:", audio_sample.shape)
    # print("Output feature_extractor shape (x):", features.shape) # 應該是 (B, 512, H_feat, W_feat)
    # print("Output z (speaker embedding) shape:", z.shape) # 應該是 (B, 128)
    # print("Output z_age (age embedding) shape:", z_age.shape) # 應該是 (B, 128)
    # print("Identity logits shape:", identity_logits.shape) # 應該是 (B, 1000)
    # print("Age logits from age classifier shape:", age_logits_from_age.shape) # 應該是 (B, 7)
    
    # lambda_id = 1.0 # 假設權重
    # lambda_age = 0.1 # 根據論文 4.1 節
    # lambda_grl = 0.1 # 根據論文 4.1 節

    # # 身份損失 (使用 F.cross_entropy，因為 ArcFaceLoss 已經輸出了 logits)
    # loss_id = F.cross_entropy(identity_logits, dummy_identity_labels)

    # # 年齡損失 (監督 z_age)
    # loss_age = F.cross_entropy(age_logits_from_age, dummy_age_labels)

    # # 對抗年齡損失 (讓 z_id 無法預測年齡)
    # # GRL 層已經處理了梯度反轉，這裡只需正常計算交叉熵，其反向梯度會被 GRL 處理
    # loss_grl = F.cross_entropy(age_logits_from_id_grl, dummy_age_labels)

    # # 總損失
    # total_loss = lambda_id * loss_id + lambda_age * loss_age + lambda_grl * loss_grl
    
    # print(f"\nDummy Loss Calculations:")
    # print(f"Identity Loss (L_id): {loss_id.item():.4f}")
    # print(f"Age Loss (L_age): {loss_age.item():.4f}")
    # print(f"Adversarial Age Loss (L_grl): {loss_grl.item():.4f}")
    # print(f"Total Loss: {total_loss.item():.4f}")