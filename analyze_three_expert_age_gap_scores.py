#!/usr/bin/env python3
"""
Analyze three-expert inference scores by age-gap bins.

Generates for each expert two figures (positive pairs, negative pairs) showing
score distributions across age-gap bins, and prints each expert's overall
max/min score.

Usage:
    python scripts/analyze_three_expert_age_gap_scores.py \
        --csv logs/ecapa_tdnn_lora/three_expert_inference/Vox-merge_three_expert_scores.csv \
        --outdir logs/ecapa_tdnn_lora/three_expert_inference/analysis

"""
import os
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from tool.EER import compute_eer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


AGE_BIN_LABELS = ["0", "1-4", "5-9", "10-19", "20+"]


def age_bin_label(age_gap: int) -> str:
    try:
        g = int(age_gap)
    except Exception:
        return "unknown"
    if g <= 0:
        return "0"
    if 1 <= g <= 4:
        return "1-4"
    if 5 <= g <= 9:
        return "5-9"
    if 10 <= g <= 19:
        return "10-19"
    return "20+"


# compute_eer is provided by tool/EER.py and returns (eer, tuned_threshold)


def summarize_and_plot(df: pd.DataFrame, expert_idx: int, outdir: str):
    col = f"expert_{expert_idx}_cosine"
    if col not in df.columns:
        print(f"Column {col} not found in CSV. Skipping expert {expert_idx}.")
        return

    # ensure numeric
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df["age_gap_year"] = pd.to_numeric(df["age_gap_year"], errors="coerce")
    df["is_same"] = pd.to_numeric(df["is_same"], errors="coerce")

    valid = df.dropna(subset=[col, "age_gap_year", "is_same"]).copy()
    if valid.shape[0] == 0:
        print(f"No valid rows for {col}.")
        return

    # EER threshold is computed over all positive/negative pairs for this expert.
    scores_all = valid[col].to_numpy(dtype=float)
    labels_all = valid["is_same"].to_numpy(dtype=int)
    # compute_eer returns (eer, tuned_threshold)
    eer, threshold = compute_eer(scores_all, labels_all)
    # compute FPR/FNR at the chosen threshold for reporting
    fpr_arr, tpr_arr, thr_arr = roc_curve(labels_all, scores_all, pos_label=1)
    # find index in roc thresholds closest to tuned threshold
    try:
        idx = int(np.nanargmin(np.abs(thr_arr - threshold)))
        fpr_at_eer = float(fpr_arr[idx])
        fnr_at_eer = float(1.0 - tpr_arr[idx])
    except Exception:
        fpr_at_eer = float("nan")
        fnr_at_eer = float("nan")

    # Score each pair with the EER threshold: score >= threshold -> same speaker.
    pred_same = (scores_all >= threshold).astype(int)
    correct_mask = pred_same == labels_all
    correct_count = int(correct_mask.sum())
    accuracy = float(correct_mask.mean())

    print(
        f"Expert {expert_idx}: EER={eer:.4f}, threshold={threshold:.6f}, "
        f"FPR={fpr_at_eer:.4f}, FNR={fnr_at_eer:.4f}, "
        f"threshold-accuracy={accuracy:.4%} ({correct_count}/{len(valid)})"
    )

    overall_max = float(valid[col].max())
    overall_min = float(valid[col].min())
    print(f"Expert {expert_idx}: max score = {overall_max:.6f}, min score = {overall_min:.6f}")

    # per-bin stats
    stats_records = []
    for is_same_val, label in [(1, "positive"), (0, "negative")]:
        sub = valid[valid["is_same"] == is_same_val].copy()
        sub["age_bin"] = sub["age_gap_year"].apply(age_bin_label)

        # collect data lists in AGE_BIN_LABELS order
        data_lists = [sub.loc[sub["age_bin"] == b, col].values for b in AGE_BIN_LABELS]

        # Plot boxplot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(data_lists, tick_labels=AGE_BIN_LABELS, showfliers=False)
        ax.axhline(
            threshold,
            color="crimson",
            linestyle="--",
            linewidth=2,
            label=f"EER threshold = {threshold:.4f}",
        )
        ax.set_title(f"Expert {expert_idx} - {label} pairs: score distribution by age gap")
        ax.set_xlabel("Age gap (years)")
        ax.set_ylabel("Cosine score")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="best")
        os.makedirs(outdir, exist_ok=True)
        fig_path = os.path.join(outdir, f"expert_{expert_idx}_{label}_by_age_bin.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        # compute stats per bin and append to records
        for bin_label in AGE_BIN_LABELS:
            arr = sub.loc[sub["age_bin"] == bin_label, col].values
            if arr.size > 0:
                pred_arr = (arr >= threshold).astype(int)
                true_arr = np.full(arr.shape[0], is_same_val, dtype=int)
                threshold_correct = int((pred_arr == true_arr).sum())
            else:
                threshold_correct = 0
            if arr.size > 0:
                rec = {
                    "expert": expert_idx,
                    "pos_or_neg": label,
                    "age_bin": bin_label,
                    "count": int(arr.size),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "median": float(np.median(arr)),
                    "p25": float(np.percentile(arr, 25)),
                    "p75": float(np.percentile(arr, 75)),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "threshold": threshold,
                    "threshold_correct": threshold_correct,
                }
            else:
                rec = {
                    "expert": expert_idx,
                    "pos_or_neg": label,
                    "age_bin": bin_label,
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "p25": np.nan,
                    "p75": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "threshold": threshold,
                    "threshold_correct": 0,
                }
            stats_records.append(rec)

    summary_record = {
        "expert": expert_idx,
        "eer": eer,
        "threshold": threshold,
        "fpr_at_eer": fpr_at_eer,
        "fnr_at_eer": fnr_at_eer,
        "threshold_accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": int(len(valid)),
        "overall_min": overall_min,
        "overall_max": overall_max,
    }

    return summary_record, stats_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="logs/ecapa_tdnn_lora/three_expert_inference/Vox-merge_three_expert_scores.csv")
    parser.add_argument("--outdir", default="logs/ecapa_tdnn_lora/three_expert_inference/analysis")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        return

    df = pd.read_csv(args.csv, low_memory=False)

    # drop footer rows if file has appended summaries (non-numeric pair_index)
    df["pair_index"] = pd.to_numeric(df.get("pair_index", pd.Series(dtype=float)), errors="coerce")
    df = df[pd.notna(df["pair_index"])].copy()

    all_stats = []
    summary_records = []
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    for expert_idx in [1, 2, 3]:
        result = summarize_and_plot(df, expert_idx, outdir)
        if result is None:
            continue
        summary_record, stats_records = result
        summary_records.append(summary_record)
        print(
            f"Expert {expert_idx} overall min={summary_record['overall_min']:.6f}, "
            f"max={summary_record['overall_max']:.6f}"
        )
        all_stats.extend(stats_records)

    # write stats CSV
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_csv = os.path.join(outdir, "per_expert_age_bin_stats.csv")
        stats_df.to_csv(stats_csv, index=False)
        print(f"Wrote per-bin stats to {stats_csv}")

    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_csv = os.path.join(outdir, "per_expert_eer_threshold_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"Wrote threshold summary to {summary_csv}")


if __name__ == "__main__":
    main()
