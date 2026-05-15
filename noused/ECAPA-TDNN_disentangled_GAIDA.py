import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.disentangled_model.GAIDA import G_AIDA, G_AIDA_Loss
from tool.EER import compute_eer
from data.vox2_loader import Vox2Dataset
from params.param import DATASET_INFO, BATCH_SIZE, MODEL_ID
from torch.utils.tensorboard import SummaryWriter
import os
import csv
from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\.load_with_torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
)

# ==========================================
# 1. 資料準備與前處理
# ==========================================
dataset = 'VoxCeleb2'
val_dataset = 'Vox-CA20'

def build_eval_dataset(audio_dirs, audio_meta_dir, max_pairs=20000):
    def find_audio_path(relative_path):
        for audio_dir in audio_dirs:
            audio_path = Path(audio_dir) / relative_path
            if audio_path.exists():
                return str(audio_path)
        raise FileNotFoundError(f"{relative_path} not found in audio_dirs")

    datalist = []
    with open(audio_meta_dir, "r") as f:
        lines = f.readlines()[:max_pairs]

    for line in lines:
        line = line.strip().split(" ")
        is_same = int(line[0])

        spk1_rel = line[1]
        spk2_rel = line[2]

        spk1_path = find_audio_path(spk1_rel)
        spk2_path = find_audio_path(spk2_rel)

        spk1_id = spk1_rel.split("/")[0]
        spk2_id = spk2_rel.split("/")[0]

        datalist.append((is_same, spk1_id, spk2_id, spk1_path, spk2_path))

    pos = sum(1 for x in datalist if x[0] == 1)
    neg = sum(1 for x in datalist if x[0] == 0)
    print(f"[Eval] Positive pairs: {pos}, Negative pairs: {neg}")

    return datalist

train_dataset = Vox2Dataset(
    audio_dir=DATASET_INFO[dataset]['AUDIO_DIR'],
    audio_meta_dir=DATASET_INFO[dataset]['AUDIO_META_DIR'],
    musan_path=DATASET_INFO["MUSAN"]['AUDIO_DIR'],
    rir_path=DATASET_INFO["RIR"]['AUDIO_DIR'],
    suffix=DATASET_INFO[dataset]['audio_suffix'],
    age_target_mode="group"
)

eval_dataset = build_eval_dataset(
    audio_dirs=DATASET_INFO['VoxCeleb1'][val_dataset]['AUDIO_DIR'],
    audio_meta_dir=DATASET_INFO['VoxCeleb1'][val_dataset]['AUDIO_DATALIST'],
    max_pairs=20000
)

def eval_network(model, datalist):
    """
    Speaker verification evaluation on pairwise trials (e.g. Vox-O)

    Returns:
        eer_before (float)
        eer_after  (float)
        final_before_embs (Tensor)
        final_after_embs  (Tensor)
        final_ids (List[str])
    """

    model.eval()

    # =========================
    # Containers
    # =========================
    before_scores = []
    after_scores = []
    labels = []

    before_embs = []
    after_embs = []
    final_ids = []

    # =========================
    # Forward (pairwise)
    # =========================
    with torch.no_grad():
        for is_same, id1, id2, path1, path2 in tqdm(datalist, desc="Evaluating"):

            emb1, sr1 = torchaudio.load(path1)
            emb2, sr2 = torchaudio.load(path2)

            emb1 = emb1.to(device)
            emb2 = emb2.to(device)

            out1 = model(emb1)
            out2 = model(emb2)

            # -------- Before disentangle (h_spk) --------
            h1 = F.normalize(out1["spkr_emb"], p=2, dim=1)
            h2 = F.normalize(out2["spkr_emb"], p=2, dim=1)

            score_before = F.cosine_similarity(h1, h2).cpu()
            before_scores.append(score_before)

            before_embs.append(out1["spkr_emb"].cpu())
            before_embs.append(out2["spkr_emb"].cpu())

            # -------- After disentangle (z_id) --------
            w1 = F.normalize(out1["mu_id"], p=2, dim=1)
            w2 = F.normalize(out2["mu_id"], p=2, dim=1)

            score_after = F.cosine_similarity(w1, w2).cpu()
            after_scores.append(score_after)

            after_embs.append(out1["mu_id"].cpu())
            after_embs.append(out2["mu_id"].cpu())

            labels.append(is_same)
            final_ids.extend([id1, id2])

    # =========================
    # EER
    # =========================
    final_labels = torch.tensor(labels).numpy()
    final_before_scores = torch.cat(before_scores).numpy()
    final_after_scores = torch.cat(after_scores).numpy()

    eer_before = compute_eer(final_before_scores, final_labels)
    eer_after = compute_eer(final_after_scores, final_labels)

    final_before_embs = torch.cat(before_embs, dim=0)
    final_after_embs = torch.cat(after_embs, dim=0)

    return (
        eer_before,
        eer_after,
        final_before_embs,
        final_after_embs,
        final_ids
    )        

# 封裝成 DataLoader
BATCH_SIZE = BATCH_SIZE

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 2. 模型初始化
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 根據你的資料自動設定參數
model = G_AIDA(
    MODEL_ID,
    input_dim=192, 
    zid_dim=256, 
    zbio_dim=256, 
    num_speakers=5990
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 對抗分類器獨立優化器（第一階段：detach 預訓練）
optimizer_adv = torch.optim.Adam([
    *model.classifier_adv_age_from_id.parameters(),
    *model.classifier_adv_id_from_age.parameters(),
], lr=0.001)

# Loss Functions
criterion = G_AIDA_Loss(
    lambda_spk=1.0,
    lambda_age=1.0,
    lambda_gender=0.0,
    lambda_kl_id=0.05,
    lambda_kl_bio=0.05,
    lambda_recon=0.1,
    lambda_adv_entropy=0.1,
)

# 訓練參數
EPOCHS = 50
WARMUP_EPOCHS = 12
ADV_PRETRAIN_EPOCHS = 10   # 第一階段：只訓練對抗分類器（detach）
ADV_RAMP_START_EPOCH = ADV_PRETRAIN_EPOCHS  # 第二階段起點：啟用 GRL 對抗
ADV_RAMP_EPOCHS = 30       # 之後 30 個 epoch 線性拉升到目標強度
best_val_loss = float('inf')
best_score_balanced = -float('inf')
best_spk_acc = 0.0
best_leak_privacy = float('inf')
best_EER = float('inf')

ADV_LAMBDA_MAX_AGE = criterion.lambda_adv_age
ADV_LAMBDA_MAX_SPK = criterion.lambda_adv_spk


def get_adv_lambda(epoch_idx, start_epoch, ramp_epochs, max_lambda):
    if epoch_idx < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return max_lambda

    progress = (epoch_idx - start_epoch + 1) / ramp_epochs
    progress = max(0.0, min(1.0, progress))
    return max_lambda * progress

# LR Scheduler (Cosine Annealing)
MIN_LR = 1e-5
COSINE_STEP_EPOCHS = 1  # 依目前 CSV 的 epoch 粒度，建議每 1 個 epoch 更新一次
cosine_updates = max(1, EPOCHS // COSINE_STEP_EPOCHS)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cosine_updates,
    eta_min=MIN_LR
)

# ==========================================
# TensorBoard & CSV Logger
# ==========================================
log_dir = "logs/GAIDA"
checkpoint_dir = "checkpoints"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

writer = SummaryWriter(log_dir=log_dir)

csv_path = os.path.join(log_dir, f"{val_dataset}_metrics.csv")
csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "epoch",
    "lr",
    "warmup_factor",
    "adv_lambda_age",
    "adv_lambda_spk",
    "train_loss",
    "train_loss_kl_id",
    "train_loss_kl_bio",
    "train_loss_mi",
    "train_loss_gender",
    "train_loss_spk",
    "train_loss_age",
    "train_loss_adv_age",
    "train_loss_adv_spk",
    "train_recon_loss",
    "train_spk_acc",
    "train_age_acc",
    "train_gender_acc",
    "train_adv_age_acc",
    "train_adv_spk_acc",
    "val_eer_before",
    "val_eer_after"
])

print("Logger initialized.")

# ==========================================
# 3. 訓練迴圈
# ==========================================
print(f"Start training on {device}...")

for epoch in range(EPOCHS):
    model.train()
    current_lr = optimizer.param_groups[0]["lr"]
    is_adv_pretrain_phase = epoch < ADV_PRETRAIN_EPOCHS

    warmup_factor = min(1.0, (epoch + 1) / WARMUP_EPOCHS)
    if is_adv_pretrain_phase:
        # 第一階段：不啟用 GRL，主損失不含對抗項
        adv_lambda_age = 0.0
        adv_lambda_spk = 0.0
        model.grl.set_lambda(0.0)
        criterion.lambda_adv_age = 0.0
        criterion.lambda_adv_spk = 0.0
    else:
        # 第二階段：啟用 GRL 並逐步拉升對抗強度
        adv_lambda_age = get_adv_lambda(
            epoch,
            ADV_RAMP_START_EPOCH,
            ADV_RAMP_EPOCHS,
            ADV_LAMBDA_MAX_AGE,
        )
        adv_lambda_spk = get_adv_lambda(
            epoch,
            ADV_RAMP_START_EPOCH,
            ADV_RAMP_EPOCHS,
            ADV_LAMBDA_MAX_SPK,
        )
        model.grl.set_lambda(max(adv_lambda_age, adv_lambda_spk))
        criterion.lambda_adv_age = adv_lambda_age
        criterion.lambda_adv_spk = adv_lambda_spk

    total_train_loss = 0.0
    train_loss_kl_id = 0.0
    train_loss_kl_bio = 0.0
    train_loss_mi = 0.0
    train_loss_gender = 0.0
    train_loss_spk = 0.0
    train_loss_age = 0.0
    train_loss_adv_age = 0.0
    train_loss_adv_spk = 0.0
    train_recon_loss = 0.0

    correct_spk = 0
    correct_age = 0
    correct_gender = 0
    correct_adv_age = 0
    correct_adv_spk = 0
    total_samples = 0

    for emb, label_spk, label_gender, label_age in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        emb = emb.to(device)
        label_spk = label_spk.to(device)
        label_gender = label_gender.to(device)
        label_age = label_age.to(device).long()

        # Forward
        outputs = model(emb)

        # 計算 Loss
        loss, loss_dict = criterion(
            outputs,
            target_spk=label_spk,
            target_age=label_age,
            target_gender=label_gender
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 第一階段：使用 detach 特徵單獨預訓練對抗分類器
        if is_adv_pretrain_phase:
            z_id_det = outputs['z_id'].detach()
            z_bio_det = outputs['z_bio'].detach()

            logits_adv_age_pt = model.classifier_adv_age_from_id(z_id_det)
            logits_adv_spk_pt = model.classifier_adv_id_from_age(z_bio_det)

            loss_adv_age_pt = F.cross_entropy(logits_adv_age_pt, label_age)
            loss_adv_spk_pt = F.cross_entropy(logits_adv_spk_pt, label_spk)
            loss_adv_pretrain = loss_adv_age_pt + loss_adv_spk_pt

            optimizer_adv.zero_grad()
            loss_adv_pretrain.backward()
            optimizer_adv.step()

        total_train_loss += loss.item()
        train_loss_kl_id += loss_dict['loss_kl_id']
        train_loss_kl_bio += loss_dict['loss_kl_bio']
        train_loss_mi += loss_dict['loss_mi']
        train_loss_gender += loss_dict['loss_gender']
        train_loss_spk += loss_dict['loss_spk']
        train_loss_age += loss_dict['loss_age']
        if is_adv_pretrain_phase:
            train_loss_adv_age += loss_adv_age_pt.item()
            train_loss_adv_spk += loss_adv_spk_pt.item()
        else:
            train_loss_adv_age += loss_dict['loss_adv_age']
            train_loss_adv_spk += loss_dict['loss_adv_spk']
        train_recon_loss += loss_dict['loss_recon']

        # 計算準確率 (監控用)
        _, pred_s_main = torch.max(outputs['logits_id'], 1)  # 從 z_id 預測說話者
        _, pred_a_main = torch.max(outputs['logits_age'], 1)  # 從 z_bio 分類年齡群
        _, pred_gender = torch.max(outputs['gender_prob'], 1)  # 性別自感知分類
        if is_adv_pretrain_phase:
            _, pred_adv_age = torch.max(logits_adv_age_pt, 1)  # detach 預訓練結果
            _, pred_adv_spk = torch.max(logits_adv_spk_pt, 1)  # detach 預訓練結果
        else:
            _, pred_adv_age = torch.max(outputs['logits_adv_age_from_id'], 1)  # 從 z_id 對抗預測年齡
            _, pred_adv_spk = torch.max(outputs['logits_adv_id_from_age'], 1)  # 從 z_bio 對抗預測說話者
        correct_spk += (pred_s_main == label_spk).sum().item()
        correct_age += (pred_a_main == label_age.long()).sum().item()
        correct_gender += (pred_gender == label_gender.long()).sum().item()
        correct_adv_age += (pred_adv_age == label_age.long()).sum().item()
        correct_adv_spk += (pred_adv_spk == label_spk).sum().item()
        total_samples += emb.size(0)

    avg_loss = total_train_loss / len(train_loader)
    avg_loss_kl_id = train_loss_kl_id / len(train_loader)
    avg_loss_kl_bio = train_loss_kl_bio / len(train_loader)
    avg_loss_mi = train_loss_mi / len(train_loader)
    avg_loss_gender = train_loss_gender / len(train_loader)
    avg_loss_spk = train_loss_spk / len(train_loader)
    avg_loss_age = train_loss_age / len(train_loader)
    avg_loss_adv_age = train_loss_adv_age / len(train_loader)
    avg_loss_adv_spk = train_loss_adv_spk / len(train_loader)
    avg_recon_loss = train_recon_loss / len(train_loader)


    acc_spk = 100 * correct_spk / total_samples
    acc_age = 100 * correct_age / total_samples
    acc_gender = 100 * correct_gender / total_samples
    acc_adv_age = 100 * correct_adv_age / total_samples
    acc_adv_spk = 100 * correct_adv_spk / total_samples

    # ==========================================
    # 4. 驗證迴圈
    # ==========================================
    eer_before, eer_after, before_embs, after_embs, final_ids = eval_network(
        model,
        eval_dataset
    )


    if best_EER > eer_after:
        best_EER = eer_after
        torch.save(model.state_dict(), f'./checkpoints/{val_dataset}_best_model.pth')
        torch.save({
            'embeddings': after_embs,
            'before_embeddings': before_embs,
            'ids': final_ids,
        }, os.path.join(checkpoint_dir, f'{val_dataset}_best_disentangled_embeddings.pt'))
        print(f"✓ 儲存最佳模型 EER: {best_EER * 100:.2f}%")

    if epoch == EPOCHS - 1:
        torch.save(model.state_dict(), f'./checkpoints/{val_dataset}_last_model.pth')
        torch.save({   
            'embeddings': after_embs,
            'before_embeddings': before_embs,
            'ids': final_ids,
        }, os.path.join(checkpoint_dir, f'{val_dataset}_last_disentangled_embeddings.pt'))
        print("✓ 儲存最終模型。")

    # ==========================================
    # 每個 Epoch 結束後打印完整資訊
    # ==========================================
    phase_name = "ADV-PRETRAIN(detach)" if is_adv_pretrain_phase else "JOINT-GRL"
    print("\n" + "="*120)
    print(f"EPOCH [{epoch+1:3d}/{EPOCHS}] [{phase_name}]")
    print("="*120)
    
    # 1. 超參數
    print(f"【超參數】LR: {current_lr:.6f} | Warmup Factor: {warmup_factor:.4f} | AdvLambda(Age/Spk): {adv_lambda_age:.4f}/{adv_lambda_spk:.4f}")
    
    # 2. 訓練損失 - 主任務
    print(f"【主任務損失】Total: {avg_loss:.4f} | Spk CE: {avg_loss_spk:.4f} | Age CE: {avg_loss_age:.4f} | Gender NLL: {avg_loss_gender:.4f}")
    
    # 3. 訓練損失 - 正則化
    print(f"【正則化損失】KL_ID: {avg_loss_kl_id:.4f} | KL_BIO: {avg_loss_kl_bio:.4f} | MI: {avg_loss_mi:.4f} | Recon: {avg_recon_loss:.4f}")
    
    # 4. 訓練損失 - 對抗任務
    print(f"【對抗任務損失】Adv_Age: {avg_loss_adv_age:.4f} | Adv_Spk: {avg_loss_adv_spk:.4f}")
    
    # 5. 訓練準確率 - 主任務
    print(f"【主任務準確率】Spk: {acc_spk:6.2f}% | Age: {acc_age:6.2f}% | Gender: {acc_gender:6.2f}%")
    
    # 6. 訓練準確率 - 對抗任務
    print(f"【對抗任務準確率】Adv_Age: {acc_adv_age:6.2f}% | Adv_Spk: {acc_adv_spk:6.2f}%")
    
    # 7. 驗證指標
    print(f"【驗證 EER】Before Disentangle: {eer_before*100:6.2f}% | After Disentangle: {eer_after*100:6.2f}%")
    
    print("="*120 + "\n")

    # ==========================================
    # TensorBoard logging
    # ==========================================
    writer.add_scalar("Loss/Train", avg_loss, epoch)
    writer.add_scalar("Lambda/AdvAge", adv_lambda_age, epoch)
    writer.add_scalar("Lambda/AdvSpk", adv_lambda_spk, epoch)
    writer.add_scalar("Loss/KL_ID", avg_loss_kl_id, epoch)
    writer.add_scalar("Loss/KL_BIO", avg_loss_kl_bio, epoch)
    writer.add_scalar("Loss/MI", avg_loss_mi, epoch)
    writer.add_scalar("Loss/Gender", avg_loss_gender, epoch)
    writer.add_scalar("Loss/Spk_CE", avg_loss_spk, epoch)
    writer.add_scalar("Loss/Age_CE", avg_loss_age, epoch)
    writer.add_scalar("Loss/Adv_Age_From_ID", avg_loss_adv_age, epoch)
    writer.add_scalar("Loss/Adv_Spk_From_Age", avg_loss_adv_spk, epoch)
    writer.add_scalar("Loss/Train_Recon", avg_recon_loss, epoch)
    writer.add_scalar("Warmup/KL_MI_Factor", warmup_factor, epoch)

    writer.add_scalar("Accuracy/Train_Spk", acc_spk, epoch)
    writer.add_scalar("Accuracy/Train_Age", acc_age, epoch)
    writer.add_scalar("Accuracy/Train_Gender", acc_gender, epoch)
    writer.add_scalar("Accuracy/Train_Adv_Age_From_ID", acc_adv_age, epoch)
    writer.add_scalar("Accuracy/Train_Adv_Spk_From_Age", acc_adv_spk, epoch)
    writer.add_scalar("LR", current_lr, epoch)
    
    writer.add_scalar("EER/Val_Before_Disentangle", eer_before, epoch)
    writer.add_scalar("EER/Val_After_Disentangle", eer_after, epoch)

    # ==========================================
    # CSV logging
    # ==========================================
    csv_writer.writerow([
        epoch + 1,
        current_lr,
        warmup_factor,
        adv_lambda_age,
        adv_lambda_spk,
        avg_loss,
        avg_loss_kl_id,
        avg_loss_kl_bio,
        avg_loss_mi,
        avg_loss_gender,
        avg_loss_spk,
        avg_loss_age,
        avg_loss_adv_age,
        avg_loss_adv_spk,
        avg_recon_loss,
        acc_spk,
        acc_age,
        acc_gender,
        acc_adv_age,
        acc_adv_spk,
        eer_before,
        eer_after
    ])
    csv_file.flush() 

    if (epoch + 1) % COSINE_STEP_EPOCHS == 0:
        scheduler.step()

writer.close()
csv_file.close()