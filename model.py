import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from modules.GSP import GlobalStatisticalPooling
from modules.ARE import ARE_Module 
from modules.GRL import GradientReversalLayer
import torchaudio
from tool.ArcFaceLoss import AAMsoftmax
from params import param


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        # input = input.unsqueeze(1)
        # padding 的〈反射〉方式可以減少邊界處突兀的「全零填充」效應，保持邊緣訊號的連續性
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class ADAL_Model(nn.Module):
    def __init__(self, feature_dim, age_classes, identity_classes):
        super(ADAL_Model, self).__init__()
        
         # 特徵提取器
        self.model = models.shufflenet_v2_x0_5(weights=torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        self.model.fc = nn.Identity()  # 去除最後的全連接層  

        # 根據 backbone 最後輸出特徵維度
        feature_dim = self.model.fc.in_features if hasattr(self.model.fc, 'in_features') else 1024
        # self.model.requires_grad_(False)  # 冻結預訓練權重
        
        self.identity_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, identity_classes)
        )
        # 年齡特徵提取模塊 (ARE)
        # self.age_extractor_module = ARE_Module(input_channels=512, output_dim=feature_dim)
        self.age_extractor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, feature_dim)
        )

        # 這裡的 GRL 是用在身份特徵 z_id 上，讓 z_id 變得年齡不相關
        self.grl = GradientReversalLayer()

        # 身份分類器 (基於 z_id)
        # self.ArcFace = AAMsoftmax(n_class=identity_classes, m = param.ARC_FACE_M, s = param.ARC_FACE_S)

        # 年齡分類器 (基於 z_age)
        self.age_classifier_on_age = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim // 2, age_classes)
        )


    def forward(self, audio, mode=None, id_label=None):
        # 1. 提取整體特徵 x (embedding)
        z = self.model(audio)
        
        # 複製一個與z相同的隨機張量 z_r
        # z_r = torch.randn_like(z)

        # 3. 提取年齡特徵 z_age
        z_age = self.age_extractor(z)

        # 3. 計算身份特徵 z_id = z - z_age
        z_id = z - z_age
        z_id_RG = torch.randn_like(z_id)  # 隨機生成與 z_id 相同形狀的張量

        if mode == "train":
            # 4. 計算身份分類損失
            id = self.identity_classifier(z_id)
            
            # 5. 預測年齡
            age = self.age_classifier_on_age(z_age)
            
            # 6. 計算年齡對抗
            z_age_grl = self.grl(z_id_RG)
            # z_age_grl = self.grl(z_id)
            
            # 7. 預測年齡 (對抗性)
            pred_grl_age = self.age_classifier_on_age(z_age_grl)

            return id, age, pred_grl_age

        else:
            return z_id

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