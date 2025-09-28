import torch
import torch.nn as nn
import torchaudio.transforms as T
from params import param
from dlordinal.losses.cdw import CDWCELoss
from tool.ArcFaceLoss import AAMsoftmax
from speechbrain.lobes.models.ResNet import ResNet
import warnings

# 精準地忽略關於 speechbrain.pretrained 的 UserWarning
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message="Module 'speechbrain.pretrained' was deprecated"
)

class SNNClassifier(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, mode=None):
        x = torch.cat([x1, x2], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.sigmoid(x)
        return output

class AttributeUnlearningModel(nn.Module):
    def __init__(self, num_main_classes, num_attribute_classes, input_channels=1, input_size=128):
        super().__init__()
        embedding_dim = 1024
        self.fbank = nn.Sequential(
            # PreEmphasis(), 
            T.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
        ) 
        self.extractor = ResNet(lin_neurons=embedding_dim)
        self.num_main_classes = num_main_classes
        self.classifier = AAMsoftmax(n_class=num_main_classes, m=param.ARC_FACE_M, s=param.ARC_FACE_S)
        # self.aux_network = AuxiliaryNetwork(embedding_dim, num_main_classes, num_attribute_classes, input_channels, input_size)
        self.SNN_classifier = SNNClassifier(embedding_dim, 1)

    def forward(self, x, id_label=None, mode=None):
        with torch.no_grad():
            x = self.fbank(x) + 1e-6  # (B, 1, n_mels, time_frames)
            x = x.squeeze(1)  # (B, n_mels, time_frames)
            x = x.transpose(1, 2)  # (B, time_frames, n_mels)
            x = x.log()
            # 零均值歸一化   
            x = x - torch.mean(x, dim=-1, keepdim=True)        
            
        h = self.extractor(x)
        if mode == "train":
            loss, acc = self.classifier(h, label=id_label)
            return loss, acc, h
        elif mode == "val" or mode == "finetune":
            return h
        else:
            raise ValueError(f"Unknown mode: {mode}")


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