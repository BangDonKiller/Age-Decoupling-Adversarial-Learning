"""
讀取 CSV 分數檔案並生成兩張獨立的學術圖表：
1) 乾淨的邊緣直方圖 + 散點圖 (無文字遮擋)
2) 統計指標對比柱狀圖
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==================== 設定區 ====================
CSV_PATH = Path("logs") / "positive_pair_scatter" / "Vox-CA20_positive_pair_scores.csv"
OUTPUT_SCATTER_FIG = Path("logs") / "positive_pair_scatter" / "Vox-CA20_academic_scatter_clean.png"
OUTPUT_STATS_FIG = Path("logs") / "positive_pair_scatter" / "Vox-CA20_statistics_bar_chart.png"
# ===============================================


def load_data(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 檔案: {csv_path}，請先運行模型產生 CSV。")
    print(f"正在讀取 CSV 數據: {csv_path}")
    df = pd.read_csv(csv_path)
    x_scores = df["Small_score"].values
    y_scores = df["Large_score"].values
    age_gap = np.abs(df["spk1_age"].values - df["spk2_age"].values)
    return x_scores, y_scores, age_gap, len(df)


def plot_clean_scatter(x_scores, y_scores, age_gap, n_samples, output_path: Path):
    """繪製無文字遮擋的純淨版散點圖+邊緣直方圖"""
    mean_x = np.mean(x_scores)
    mean_y = np.mean(y_scores)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    # 1. 主散點圖 (左下)
    ax_scatter = fig.add_subplot(gs[1, 0])
    scatter = ax_scatter.scatter(
        x_scores,
        y_scores,
        c=age_gap,
        cmap="viridis",
        s=18,
        alpha=0.25,  # 降低透明度以看清重疊密度
        edgecolors="none",
    )
    
    ax_scatter.set_xlim([-0.2, 1.0])
    ax_scatter.set_ylim([-0.2, 1.0])
    ax_scatter.plot([-0.2, 1.0], [-0.2, 1.0], linestyle="--", color="#666666", linewidth=1.2, label="y = x")

    # 2. 小專家邊緣直方圖 (上方)
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_histx.hist(x_scores, bins=60, color="#440154", alpha=0.7, density=True)
    ax_histx.axvline(mean_x, color="red", linestyle="--", linewidth=1.2, label=f"Mean: {mean_x:.3f}")
    ax_histx.axis("off")
    ax_histx.legend(loc="upper right", fontsize=9)

    # 3. 大專家邊緣直方圖 (右方)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
    ax_histy.hist(y_scores, bins=60, color="#fde725", alpha=0.7, density=True, orientation="horizontal")
    ax_histy.axhline(mean_y, color="red", linestyle="--", linewidth=1.2, label=f"Mean: {mean_y:.3f}")
    ax_histy.axis("off")
    ax_histy.legend(loc="lower right", fontsize=9)

    # 4. 裝飾
    ax_scatter.set_xlabel("Small Expert Cosine Similarity", fontsize=12)
    ax_scatter.set_ylabel("Large Expert Cosine Similarity", fontsize=12)
    ax_scatter.grid(True, linestyle=":", alpha=0.6)
    ax_scatter.legend(loc="upper left")
    ax_scatter.set_title(f"Vox-CA20 Positive Pairs Scatter Plot (N = {n_samples})", fontsize=13, pad=10)

    # 5. 色條
    cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.7])
    fig.colorbar(scatter, cax=cbar_ax, label="Age Gap")

    plt.savefig(str(output_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"🎉 成功！純淨版散點圖已儲存至: {output_path}")


def plot_statistics_bars(x_scores, y_scores, output_path: Path):
    """繪製獨立的指標對比柱狀圖"""
    mean_x, median_x = np.mean(x_scores), np.median(x_scores)
    mean_y, median_y = np.mean(y_scores), np.median(y_scores)
    
    threshold = 0.4
    above_th_x = np.mean(x_scores > threshold) * 100
    above_th_y = np.mean(y_scores > threshold) * 100

    # 建立雙子圖 (左邊比分數，右邊比通過率)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # 子圖 1: Mean & Median 對比
    labels = ['Mean Score', 'Median Score']
    small_vals = [mean_x, median_x]
    large_vals = [mean_y, median_y]
    
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, small_vals, width, label='Small Expert', color='#440154', alpha=0.8)
    rects2 = ax1.bar(x + width/2, large_vals, width, label='Large Expert', color='#fde725', alpha=0.8)
    
    ax1.set_ylabel('Cosine Similarity', fontsize=11)
    ax1.set_title('Average Score Comparison', fontsize=12, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim([0, 0.6])
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.5)
    
    # 在柱狀圖上標註數值
    ax1.bar_label(rects1, padding=3, fmt='%.3f')
    ax1.bar_label(rects2, padding=3, fmt='%.3f')

    # 子圖 2: 識別率 (Score > 0.4) 對比
    categories = ['Ratio > 0.4']
    rects3 = ax2.bar([0 - 0.2], [above_th_x], 0.4, label='Small Expert', color='#440154', alpha=0.8)
    rects4 = ax2.bar([0 + 0.2], [above_th_y], 0.4, label='Large Expert', color='#fde725', alpha=0.8)
    
    ax2.set_ylabel('Percentage (%)', fontsize=11)
    ax2.set_title(f'Pass Rate Comparison (Threshold = {threshold})', fontsize=12, pad=10)
    ax2.set_xticks([0])
    ax2.set_xticklabels(categories)
    ax2.set_ylim([0, 100])
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.5)
    
    ax2.bar_label(rects3, padding=3, fmt='%.1f%%')
    ax2.bar_label(rects4, padding=3, fmt='%.1f%%')

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()
    print(f"🎉 成功！統計對比圖已儲存至: {output_path}")

    # 同時在終端機印出文字報告，方便複製
    print("\n" + "="*40)
    print("【Vox-CA20 同人樣本統計報告】")
    print(f"小專家 (Small Expert) 均值: {mean_x:.4f} | 中位數: {median_x:.4f} | 通過率: {above_th_x:.2f}%")
    print(f"大專家 (Large Expert) 均值: {mean_y:.4f} | 中位數: {median_y:.4f} | 通過率: {above_th_y:.2f}%")
    print("="*40)


if __name__ == "__main__":
    x_sc, y_sc, gap, n_samp = load_data(CSV_PATH)
    
    # 1. 畫純淨版散點圖 (無文字遮擋)
    plot_clean_scatter(x_sc, y_sc, gap, n_samp, OUTPUT_SCATTER_FIG)
    
    # 2. 畫獨立柱狀圖
    plot_statistics_bars(x_sc, y_sc, OUTPUT_STATS_FIG)