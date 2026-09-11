# Cross-Age Speaker Verification with Age-Decoupled Embeddings
*(基於集成學習之跨年齡說話者驗證與年齡資訊解耦研究)*

本 repository 是一個以 ECAPA-TDNN 為主要聲音特徵擷取器的跨年齡說話者驗證研究實作。研究目標是降低年齡差異對 speaker embedding 與驗證分數的影響，同時保留辨識說話者所需的資訊。

目前包含以下幾類實驗：

- 使用線性解耦 MLP 與 decorrelation loss 的年齡資訊抑制實驗。
- Siamese ECAPA-TDNN 的全參數微調。
- 以多個專家模型與 router 組成的 Cross-Gap Ensemble。
- 使用固定 router 的 Ensemble 變體，以及校準器與不同跨年齡 gap 的比較。
- 以 EER、minDCF、年齡分類 AUC、embedding 分布與統計分析評估模型。

## Repository Structure

- `ECAPA-TDNN_linear_decorr_mlp_step_based.py`: 使用 step-based scheduler 訓練線性解耦 MLP，並評估年齡相關性與 speaker verification。
- `ECAPA-TDNN_siamese_full_finetune_train.py`: 使用 Siamese 架構對 ECAPA-TDNN 進行全參數微調。
- `dynamic_ensemble_train.py`: 訓練具有動態 router 與 per-expert calibrator 的 CrossGapEnsemble。
- `fixed_weight_ensemble_train.py`: 訓練與評估固定 router 的 CrossGapFixedRouterEnsemble。
- `model/`: ECAPA-TDNN 特徵擷取器、Siamese network、解耦模型與 Ensemble 模型。
- `loss/`: Circle Loss、Cosine Loss 與其他訓練目標。
- `data/`: VoxCeleb1、VoxCeleb2 的資料載入器。
- `tool/`: EER、minDCF、checkpoint 管理、解耦訓練與評估工具。
- `analysis/`: embedding、相似度分布、降維、shortcut dependency 與模型錯誤分析腳本。
- `params/param.py`: 裝置、batch size、預訓練模型與資料集路徑設定。
- `checkpoints/`: 模型 checkpoint 輸出位置。
- `logs/`: TensorBoard 與各項實驗的輸出紀錄。
- `Dockerfile`: Docker 建置與 GPU container 執行說明。

## Environment Setup

本專案建議使用 Linux、CUDA GPU 與 Python 3.10 以上環境。Docker image 目前以 PyTorch 2.8.0、CUDA 12.9 與 cuDNN 9 為基礎。

### Using Requirements

```bash
pip install -r requirements.txt
```

### Using Docker

```bash
docker build -t adal .
docker run --gpus device=0 --rm -it \
	-v "$(pwd):/app" \
	-v /path/to/Dataset:/app/dataset \
	adal
```

請將 `/path/to/Dataset` 替換成實際資料集根目錄。

### Dataset Configuration

執行訓練前，請確認 `params/param.py` 中的 `DATASET_INFO` 指向本機資料位置。預設設定包含：

- VoxCeleb2：`small`、`medium`、`large`、`mixture` 與 `ensemble` 訓練變體。
- VoxCeleb1：`Vox-O`、`Vox-E`、`Vox-H`、`Vox-CA5`、`Vox-CA10`、`Vox-CA15`、`Vox-CA20` 等測試組合。

資料集下載來源與 metadata 檔案：
- VoxCeleb 官方資料集載點: [VoxCeleb](https://mm.kaist.ac.kr/datasets/voxceleb/)
- VoxCeleb metadata: [qinxiaoyi CASV](https://github.com/qinxiaoyi/Cross-Age_Speaker_Verification)

## ECAPA-TDNN 模型預訓練權重

本專案使用 ECAPA-TDNN 作為主要的 speaker embedding 特徵擷取器。預設模型設定位於 `params/param.py`，目前使用的 Hugging Face 模型為 `yangwang825/ecapa-tdnn-vox2`。

若使用本地預訓練權重，請將權重放置於：

```text
pretrained_models/pretrain.model
```

預訓練模型下載來源、權重格式與載入方式：[TaoRuijie ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)

## Quick Start Guide

各入口腳本目前以檔案開頭的設定常數控制訓練模式、資料集變體、seed、checkpoint 與 log 路徑；請先修改對應設定，再從 repository 根目錄執行。

### 1. Age-Decoupling Training

編輯 `ECAPA-TDNN_linear_decorr_mlp_step_based.py` 中的資料集與超參數後執行：

```bash
python ECAPA-TDNN_linear_decorr_mlp_step_based.py
```

此實驗會訓練 speaker embedding 與 age-related representation，並使用 decorrelation loss 進行解耦。輸出位置與實驗命名由腳本中的 checkpoint 和 log 設定決定。

### 2. Siamese Full Fine-Tuning

在 `ECAPA-TDNN_siamese_full_finetune_train.py` 中設定 `RUN_MODE`、訓練資料變體與 inference datasets：

```bash
python ECAPA-TDNN_siamese_full_finetune_train.py
```

### 3. Dynamic-Router Ensemble

CrossGapEnsemble 使用多個 expert、動態 router 與 router guidance loss：

```bash
python dynamic_ensemble_train.py
```

若要執行不使用 score calibrator 的固定 router 變體：

```bash
python fixed_weight_ensemble_train.py
```

### 4. Evaluation and Analysis

訓練完成後，可使用 `analysis/` 中的工具分析模型輸出的 embedding、分數分布與錯誤案例。常用分析包括：

1. `Centered_Kernel_Alignment.py`：比較不同表示之間的 CKA 相似度。
2. `analyze_shortcut_dependency.py`：分析模型是否依賴非目標的 shortcut information。
3. `noused/analyze_gender_separability_sb_sw.py`：比較性別組內與組間的分布距離。
4. `plot_age_gap_vs_cosine.py`：觀察年齡差距與 cosine similarity 的關係。
5. `Score_Distribution.py`、`DET_curve.py`、`dprime.py`：分析 verification score、DET curve 與 d-prime。
6. `SVD.py`、`pre_normalize_analysis.py`：分析 embedding 的奇異值與正規化效果。
7. `topk_hard_negative.py`、`Venn_Diagram_of_Hard_Errors.py`：分析 hard negative 與不同模型的錯誤交集。
8. `Gradual_Acoustic_Corruption_Stress_Test.py`：測試逐步聲學干擾下的模型穩健性。

## Evaluation Metrics

本專案主要使用以下指標：

- EER (Equal Error Rate)
- minDCF (minimum Detection Cost Function)
- speaker / age embedding 的相關性與可分性

## Windows Notes

使用 Windows 時，請先安裝 shared build 的 ffmpeg，並將其加入系統環境變數。下載位置可參考 [GyanD/codexffmpeg releases](https://github.com/GyanD/codexffmpeg/releases)。

部分 torchaudio 版本在 Windows 上可能需要調整 `torchaudio/_extension/__init__.py` 的 Python 版本判斷：

```python
# 原始條件
if os.name == "nt" and (3, 8) <= sys.version_info < (3, 9):

# 原專案使用的修正條件
if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
```

此修改取決於實際 torchaudio 版本；若環境可以正常載入音訊，無需修改套件原始碼。

## Author & Contact

**Ming-Yu, Shieh (謝名祐)**

Master's Thesis, November 2026

Department of Computer Science and Information Engineering

National Central University (國立中央大學)

Advisor: Dr. Hung-Hsuan Chen (陳弘軒 博士)

If you have any questions about the code, the paper, or the methodology, feel free to reach out:
Email：[parisdata@gmail.com](mailto:parisdata@gmail.com)