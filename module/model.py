import torch
import torch.nn as nn
from torch.autograd import Function
from module.GRL import GradientReversalLayer
from module.ARE import ARE_Module
from tool.ArcFaceLoss import ArcFaceLoss


class ADAL_Model(nn.Module):
    def __init__(self, feature_dim, age_classes):
        super(ADAL_Model, self).__init__()
        self.feature_extractor = ResNet34() # 假設這是特徵提取器
        self.age_extractor_module = ARE_Module() # 假設這是 ARE 模組

        # 這裡的 GRL 是用在身份特徵 z_id 上，將其送到一個年齡分類器，
        # 但這個分類器的目的是讓 z_id 變得年齡不相關
        self.grl = GradientReversalLayer()
        self.age_classifier_on_id = nn.Linear(feature_dim, age_classes) # 這個分類器用於對抗

        # 身份分類器 (基於 z_id)
        self.identity_classifier = ArcFaceLoss() # 這裡可能是一個Linear層 + ArcFace功能

        # 年齡分類器 (基於 z_age)
        self.age_classifier_on_age = nn.Linear(feature_dim, age_classes)


    def forward(self, audio):
        # 1. 提取整體特徵 z
        features = self.feature_extractor(audio)
        z = self.global_statistical_pooling(features) # 假設有這個池化層

        # 2. 提取年齡特徵 z_age
        z_age = self.age_extractor_module(features)

        # 3. 計算身份特徵 z_id = z - z_age
        z_id = z - z_age

        # 4. 身份分類 (用於L_id)
        identity_logits = self.identity_classifier(z_id) # 這可能是ArcFace計算

        # 5. 年齡分類 (用於L_age，監督z_age)
        age_logits_from_age = self.age_classifier_on_age(z_age)

        # 6. 對抗年齡分類 (用於L_grl，將z_id通過GRL後再送入年齡分類器)
        z_id_grl = self.grl(z_id)
        age_logits_from_id_grl = self.age_classifier_on_id(z_id_grl)

        return identity_logits, age_logits_from_age, age_logits_from_id_grl

