import torch.nn as nn
import torch.nn.functional as F

def calculate_adal_loss(identity_logits, age_logits_from_age, age_logits_from_id_grl,
                        true_identity_labels, true_age_labels,
                        lambda_id=1.0, lambda_age=0.1, lambda_grl=0.1):

    # L_id: 身份分類損失
    loss_id = F.cross_entropy(identity_logits, true_identity_labels) # 這裡簡化為交叉熵，實際是ArcFace

    # L_age: 年齡分類損失 (監督z_age)
    loss_age = F.cross_entropy(age_logits_from_age, true_age_labels)

    # L_grl: 對抗年齡損失 (讓z_id無法預測年齡)
    # 這個損失的目標是讓 age_classifier_on_id 無法正確分類年齡，
    # 也就是說，這個損失越大（分類越準），通過GRL後的梯度反向傳播會讓 z_id 變得更難預測年齡。
    loss_grl = F.cross_entropy(age_logits_from_id_grl, true_age_labels)

    # 總損失
    total_loss = lambda_id * loss_id + lambda_age * loss_age + lambda_grl * loss_grl
    return total_loss