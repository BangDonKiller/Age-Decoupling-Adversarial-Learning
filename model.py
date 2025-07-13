import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.GSP import GlobalStatisticalPooling
from modules.ARE import ARE_Module 
from modules.GRL import GradientReversalLayer
from tool.ArcFaceLoss import ArcFaceLoss


class ADAL_Model(nn.Module):
    def __init__(self, feature_dim, age_classes, identity_classes):
        super(ADAL_Model, self).__init__()
        
         # 特徵提取器
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT) 
        # 修改 ResNet 的第一個卷積層以適應單通道輸入 (音頻 Mel 頻譜通常是單通道)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 替換 ResNet 的第一個卷積層以及去除最後一個分類層
        # self.resnet.conv1 = self.conv1 # 直接替換 ResNet 的 conv1
        # 為了更安全，我們用 Sequential 包裝起來，並只包含 ResNet 的特徵提取部分
        self.feature_extractor = nn.Sequential(
            self.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4 # layer4 的輸出通道數是 512
        )
        
        # 全局統計池化層 + 說話者嵌入層 => 提取出 z
        # 根據 ResNet34 layer4 的輸出通道數 512，GSP 應輸出 512 * 2 = 1024 維
        self.speaker_embedding_layer = nn.Sequential(
            GlobalStatisticalPooling(),
            nn.Linear(512 * 2 * 3, feature_dim) # 修正此處的輸入維度為 1024
        )
        
        
        # 年齡特徵提取模塊 (ARE)
        # ARE 也接收 ResNet34 layer4 的輸出，所以 input_channels 也是 512
        self.age_extractor_module = ARE_Module(input_channels=512, output_dim=feature_dim)

        # 這裡的 GRL 是用在身份特徵 z_id 上，將其送到一個年齡分類器，
        # 但這個分類器的目的是讓 z_id 變得年齡不相關
        self.grl = GradientReversalLayer()
        self.age_classifier_on_id = nn.Linear(feature_dim, age_classes)

        # 身份分類器 (基於 z_id)
        self.identity_classifier = ArcFaceLoss(embedding_size=feature_dim, num_classes=identity_classes) # 需要傳入總類別數

        # 年齡分類器 (基於 z_age)
        self.age_classifier_on_age = nn.Linear(feature_dim, age_classes)


    def forward(self, audio, labels=None):
        # 1. 提取整體特徵 x (論文中的 Feature maps)
        features = self.feature_extractor(audio) # features 現在是 (B, 512, H_feat, W_feat)
        
        # 2. 提取 z (整個說話者嵌入)
        z = self.speaker_embedding_layer(features) # z 現在是 (B, feature_dim)
        

        # 3. 提取年齡特徵 z_age
        z_age = self.age_extractor_module(features) # z_age 現在是 (B, feature_dim)

        # 3. 計算身份特徵 z_id = z - z_age
        z_id = z - z_age

        # 4. 身份分類 (用於L_id)
        identity_logits = self.identity_classifier(z_id, labels) # ArcFace計算

        # 5. 年齡分類 (用於L_age，監督z_age)
        age_logits_from_age = self.age_classifier_on_age(z_age)

        # 6. 對抗年齡分類 (用於L_grl，將z_id通過GRL後再送入年齡分類器)
        z_id_grl = self.grl(z_id)
        age_logits_from_id_grl = self.age_classifier_on_id(z_id_grl)

        return features, z, z_age, identity_logits, age_logits_from_age, age_logits_from_id_grl


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