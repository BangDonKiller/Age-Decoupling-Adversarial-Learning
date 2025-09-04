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
        self.fc1 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dim // 2, num_main_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, h, mode=None):
        h = self.fc1(h)
        
        # if mode == "train":
        h = self.relu(h)
        h = self.fc2(h)
        # else:
        #     h = self.sigmoid(h)
        return h

class ADAL_Model(nn.Module):
    def __init__(self, feature_dim, age_classes, identity_classes):
        super(ADAL_Model, self).__init__()
        
        self.feature_extractor = RepresentationDetachmentExtractor()
        
        self.main_task_classifier = MainTaskClassifier(embedding_dim=feature_dim, num_main_classes=identity_classes)
        
        # 年齡特徵提取模塊 (ARE)
        # self.age_extractor_module = ARE_Module(input_channels=512, output_dim=feature_dim)
        # self.age_extractor = nn.Sequential(
        #     nn.Linear(feature_dim, feature_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(feature_dim, feature_dim)
        # )

        # 這裡的 GRL 是用在身份特徵 z_id 上，讓 z_id 變得年齡不相關
        self.grl = GradientReversalLayer()

        # 身份分類器 (基於 z_id)
        self.speaker_loss = AAMsoftmax(n_class=identity_classes, m = param.ARC_FACE_M, s = param.ARC_FACE_S)

        # 年齡分類器 (基於 z_age)
        self.age_classifier_on_age = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim // 2, age_classes)
        )
        
        # 年齡分類器 (基於 z_id 的對抗性年齡預測)
        self.age_classifier_on_grl_age = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim // 2, age_classes)
        )


    def forward(self, audio, mode=None, id_label=None):
        z = self.feature_extractor(audio)

        if mode == "train":
            z_id = self.main_task_classifier(z)
            return z_id

        else:
            return z_id

    
# --- 3. 完整的屬性遺忘模型 ---
class AttributeUnlearningModel(nn.Module):
    def __init__(self, num_main_classes, num_attribute_classes, input_channels=1, input_size=128):
        super().__init__()
        self.extractor = RepresentationDetachmentExtractor()
        # ShuffleNet v2 x1.0 輸出的 embedding 維度是 1024
        embedding_dim = 1024 
        self.classifier = AAMsoftmax(n_class=num_main_classes, m=param.ARC_FACE_M, s=param.ARC_FACE_S)

    def forward(self, x, id_label=None, mode=None):
        if mode == "train":
            h = self.extractor(x)
            # main_task_output = self.classifier(h)
            loss, acc = self.classifier(h, label=id_label)
            # return main_task_output, h
            return loss, acc, h
        else:
            h = self.extractor(x)
            return h



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