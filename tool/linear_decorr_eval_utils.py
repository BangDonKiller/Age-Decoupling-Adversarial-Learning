import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from tool.EER import compute_eer
from tool.linear_decorr_training_utils import RunningAgeCorrelation


def compute_age_auc(age_logits: torch.Tensor, age_targets: torch.Tensor, num_age_groups: int) -> float:
    """根據 age logits 與標籤計算 AUC；多分類時使用 macro OVR AUC。"""
    if age_logits is None or age_targets is None:
        return float("nan")
    if age_logits.numel() == 0 or age_targets.numel() == 0:
        return float("nan")

    y_true = age_targets.detach().cpu().numpy().astype(int)
    y_score = torch.softmax(age_logits.detach(), dim=1).cpu().numpy()

    try:
        if num_age_groups <= 2:
            if y_score.shape[1] < 2:
                return float("nan")
            return float(roc_auc_score(y_true, y_score[:, 1]))

        return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def plot_age_corr_heatmap(age_corr, save_path, title, age_dim: int = 1):
    """繪製年齡神經元與其他神經元相關係數熱力圖（age_dim x N）。"""
    if age_corr is None:
        return

    values = np.asarray(age_corr[:, age_dim:], dtype=np.float32)
    if values.size == 0:
        return

    heat = values
    plt.figure(figsize=(12, max(2.5, 0.8 * heat.shape[0] + 1.5)))
    im = plt.imshow(heat, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.yticks(range(age_dim), [f"age neuron {i}" for i in range(age_dim)])
    plt.xlabel("Other neuron index")
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label("Correlation")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def eval_network(model, speaker_extractor, eval_dataloader, num_age_groups, latent_dim: int, age_dim: int, device):
    model.eval()
    speaker_extractor.eval()

    before_scores = []
    after_scores = []
    labels = []

    before_embs = []
    after_embs = []
    final_ids = []
    age_logits_all = []
    age_targets_all = []
    corr_meter = RunningAgeCorrelation(latent_dim=latent_dim, age_dim=age_dim)

    with torch.no_grad():
        for is_same, id1, id2, wav1, wav2, age1, age2 in tqdm(eval_dataloader, desc="Evaluating"):
            wav1 = wav1.to(device, non_blocking=True)
            wav2 = wav2.to(device, non_blocking=True)

            spk_emb1 = speaker_extractor(wav1)
            spk_emb2 = speaker_extractor(wav2)

            out1 = model(spk_emb1)
            out2 = model(spk_emb2)

            h1 = F.normalize(spk_emb1, p=2, dim=1)
            h2 = F.normalize(spk_emb2, p=2, dim=1)
            score_before = F.cosine_similarity(h1, h2).cpu()
            before_scores.append(score_before)

            w1 = F.normalize(out1["z_id"], p=2, dim=1)
            w2 = F.normalize(out2["z_id"], p=2, dim=1)
            score_after = F.cosine_similarity(w1, w2).cpu()
            after_scores.append(score_after)

            before_embs.append(spk_emb1.cpu())
            before_embs.append(spk_emb2.cpu())
            after_embs.append(out1["z_id"].cpu())
            after_embs.append(out2["z_id"].cpu())
            age_logits_all.append(out1["logits_age"].cpu())
            age_logits_all.append(out2["logits_age"].cpu())
            age_targets_all.extend(age1.cpu().tolist())
            age_targets_all.extend(age2.cpu().tolist())
            corr_meter.update(out1["z"])
            corr_meter.update(out2["z"])

            labels.extend(is_same.cpu().tolist())
            final_ids.extend(list(id1))
            final_ids.extend(list(id2))

    final_labels = np.array(labels)
    final_before_scores = torch.cat(before_scores).numpy() if before_scores else np.array([])
    final_after_scores = torch.cat(after_scores).numpy() if after_scores else np.array([])

    eer_before = compute_eer(final_before_scores, final_labels) if len(final_before_scores) > 0 else float("nan")
    eer_after = compute_eer(final_after_scores, final_labels) if len(final_after_scores) > 0 else float("nan")

    final_before_embs = torch.cat(before_embs, dim=0) if before_embs else torch.tensor([])
    final_after_embs = torch.cat(after_embs, dim=0) if after_embs else torch.tensor([])
    eval_age_corr = corr_meter.correlations()
    eval_age_auc = compute_age_auc(
        torch.cat(age_logits_all, dim=0) if age_logits_all else torch.tensor([]),
        torch.tensor(age_targets_all, dtype=torch.long),
        num_age_groups=num_age_groups,
    ) if age_logits_all and age_targets_all else float("nan")

    return eer_before, eer_after, final_before_embs, final_after_embs, final_ids, eval_age_corr, eval_age_auc


def eval_classification_network(model, speaker_extractor, val_dataloader, criterion, latent_dim: int, age_dim: int, device):
    """在同資料集切出的驗證集上，評估多分類與年齡分類表現。"""
    model.eval()
    speaker_extractor.eval()

    val_total_loss = 0.0
    val_spk_loss = 0.0
    val_age_loss = 0.0
    val_decorr_loss = 0.0
    val_correct_spk = 0
    val_correct_age = 0
    val_total_samples = 0
    val_age_logits_all = []
    val_age_targets_all = []
    corr_meter = RunningAgeCorrelation(latent_dim=latent_dim, age_dim=age_dim)

    with torch.no_grad():
        for waveform, label_spk, _, label_age in tqdm(val_dataloader, desc="Validating"):
            waveform = waveform.to(device, non_blocking=True)
            label_spk = label_spk.to(device, non_blocking=True)
            label_age = label_age.to(device, non_blocking=True).long()

            speaker_emb = speaker_extractor(waveform)
            outputs = model(speaker_emb)
            loss, loss_dict = criterion(outputs=outputs, target_spk=label_spk, target_age=label_age)

            val_total_loss += float(loss.item())
            val_spk_loss += float(loss_dict["loss_spk"].item())
            val_age_loss += float(loss_dict["loss_age"].item())
            val_decorr_loss += float(loss_dict["loss_decorr"].item())

            pred_spk = loss_dict["arcface_logits"].argmax(dim=1)
            pred_age = outputs["logits_age"].argmax(dim=1)
            val_correct_spk += (pred_spk == label_spk).sum().item()
            val_correct_age += (pred_age == label_age).sum().item()
            val_total_samples += label_spk.size(0)
            val_age_logits_all.append(outputs["logits_age"].detach().cpu())
            val_age_targets_all.append(label_age.detach().cpu())
            corr_meter.update(outputs["z"])

    num_batches = max(1, len(val_dataloader))
    val_total_loss /= num_batches
    val_spk_loss /= num_batches
    val_age_loss /= num_batches
    val_decorr_loss /= num_batches
    val_spk_acc = 100.0 * val_correct_spk / max(1, val_total_samples)
    val_age_acc = 100.0 * val_correct_age / max(1, val_total_samples)
    val_age_corr = corr_meter.correlations()
    val_age_auc = compute_age_auc(
        torch.cat(val_age_logits_all, dim=0) if val_age_logits_all else torch.tensor([]),
        torch.cat(val_age_targets_all, dim=0) if val_age_targets_all else torch.tensor([], dtype=torch.long),
        num_age_groups=model.age_head.out_features,
    )
    val_age_corr_values = val_age_corr[:, age_dim:] if val_age_corr is not None else None
    val_age_corr_abs_mean = (
        float(np.mean(np.abs(val_age_corr_values))) if val_age_corr_values is not None else float("nan")
    )

    return {
        "val_total_loss": val_total_loss,
        "val_spk_loss": val_spk_loss,
        "val_age_loss": val_age_loss,
        "val_decorr_loss": val_decorr_loss,
        "val_spk_acc": val_spk_acc,
        "val_age_acc": val_age_acc,
        "val_age_auc": val_age_auc,
        "val_age_corr": val_age_corr,
        "val_age_corr_abs_mean": val_age_corr_abs_mean,
    }
