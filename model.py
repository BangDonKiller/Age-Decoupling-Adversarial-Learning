import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.models as models
from modules.GSP import GlobalStatisticalPooling
# from module.ARE import ARE_Module
# from module.GRL import GradientReversalLayer
# from tool.ArcFaceLoss import ArcFaceLoss


class ADAL_Model(nn.Module):
    def __init__(self, feature_dim, age_classes, identity_classes):
        super(ADAL_Model, self).__init__()
        
         # 特徵提取器
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT) 
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 替換 ResNet 的第一個卷積層以適應單通道輸入以及去除最後一個分類層
        self.feature_extractor = nn.Sequential(
            self.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        )
        
        # 全局統計池化層 + 說話者嵌入層 => 提取出 z
        self.speaker_embedding_layer = nn.Sequential(
            GlobalStatisticalPooling(),
            nn.Linear(3072, feature_dim) # feature_dim 就是 128
        )
        
        
        # 年齡特徵提取模塊 (ARE)
        # self.age_extractor_module = ARE_Module(input_channels=512, output_dim=feature_dim)

        # 這裡的 GRL 是用在身份特徵 z_id 上，將其送到一個年齡分類器，
        # 但這個分類器的目的是讓 z_id 變得年齡不相關
        # self.grl = GradientReversalLayer()
        # self.age_classifier_on_id = nn.Linear(feature_dim, age_classes)

        # 身份分類器 (基於 z_id)
        # self.identity_classifier = ArcFaceLoss(embedding_size=feature_dim, num_classes=identity_classes) # 需要傳入總類別數

        # 年齡分類器 (基於 z_age)
        # self.age_classifier_on_age = nn.Linear(feature_dim, age_classes)


    def forward(self, audio):
        # 1. 提取整體特徵 z
        features = self.feature_extractor(audio)
        z = self.speaker_embedding_layer(features)
        

        # 2. 提取年齡特徵 z_age
        # z_age = self.age_extractor_module(features)

        # 3. 計算身份特徵 z_id = z - z_age
        # z_id = z - z_age

        # 4. 身份分類 (用於L_id)
        # identity_logits = self.identity_classifier(z_id) # 這可能是ArcFace計算

        # 5. 年齡分類 (用於L_age，監督z_age)
        # age_logits_from_age = self.age_classifier_on_age(z_age)

        # 6. 對抗年齡分類 (用於L_grl，將z_id通過GRL後再送入年齡分類器)
        # z_id_grl = self.grl(z_id)
        # age_logits_from_id_grl = self.age_classifier_on_id(z_id_grl)

        # return identity_logits, age_logits_from_age, age_logits_from_id_grl
        return features, z


if __name__ == "__main__":
    # 測試模型
    # feature_dim 是論文中提到的 128 維說話者嵌入
    model = ADAL_Model(feature_dim=128, age_classes=7, identity_classes=1000)
    
    # 根據論文，輸入應該是 80 維的 Mel-filterbank energies
    # 假設一個音頻有 300 幀 (約 3 秒)
    batch_size = 4
    num_frames = 300 
    
    # 輸入張量形狀： (batch_size, channels=1, height=80, width=num_frames)
    audio_sample = torch.randn(batch_size, 1, 80, num_frames)
    
    # 將模型移到 CPU 或 GPU (如果可用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    audio_sample = audio_sample.to(device)

    features, z = model(audio_sample)
    
    print("Input audio_sample shape:", audio_sample.shape)
    print("Output feature_extractor shape:", features.shape)
    print("Output global_statistical_pooling shape:", z.shape)
