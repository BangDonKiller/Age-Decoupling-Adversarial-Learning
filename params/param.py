import os
import torch

# --- 1. 通用設定 (General Settings) ---
# 設置使用的計算設備：'cuda' 代表 GPU，'cpu' 代表 CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# 設定隨機種子，確保實驗可重現性
RANDOM_SEED = 42
PRETRAIN = True
AUGMENT = False

# --- 2. 路徑設定 (Path Configurations) ---
# 數據集根目錄
DATA_ROOT = ['D:/Dataset/VoxCeleb2/vox2_dev_wav/dev/aac']
# 數據列表文件路徑 (包含音頻路徑、身份ID、年齡組ID)
DATA_LIST_FILE = "D:/Dataset/Cross-Age_Speaker_Verification/vox2dev/segment2age.npy"
# 訓練數據列表文件路徑
VAL_DATA_ROOT = [
    "D:/Dataset/VoxCeleb1/vox1_dev_wav/wav", 
    "D:/Dataset/VoxCeleb1/vox1_test_wav/wav",
    "D:/Dataset/VoxCeleb2/vox2_dev_wav/dev/aac"
]
# 驗證數據列表文件路徑
FINETUNE_DATA_LIST_FILE = "D:/Dataset/VoxCeleb2/vox2celeb2pair.txt"
VAL_DATA_LIST_FILE = "D:/Dataset/Cross-Age_Speaker_Verification/trials/Vox-CA5/test.txt"
# 訓練日誌、模型檢查點的儲存目錄
LOG_DIR = './logs'
CHECKPOINT_DIR = './checkpoints'
SCORE_DIR = './scores' # 儲存驗證結果 (EER, mDCF)
TENSOR_BOARD_DIR = './tensorboard_logs'  # TensorBoard 日誌目錄
FINETUNE_DIR = './finetune'
PRETRAINED_WEIGHTS_PATH = './checkpoints/best_pretrain_model.pth' # 預訓練模型權重路徑
# 確保目錄存在
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SCORE_DIR, exist_ok=True)
os.makedirs(FINETUNE_DIR, exist_ok=True)

# 數據增強所需的外部數據集路徑
MUSAN_DIR = 'D:/Dataset/musan/musan'
RIR_NOISE_DIR = 'D:/Dataset/sim_rir_16k/simulated_rirs_16k'

# --- 3. 音頻特徵提取參數 (Audio Feature Extraction Parameters) ---
# 音頻採樣率 (Hz) - 論文中未明確提及，常用值
SAMPLE_RATE = 16000
NUM_FRAMES = 200  # 每個音頻片段的幀數
# 梅爾濾波器組數量 (80維對數梅爾濾波器組能量)
N_MELS = 80
# 幀長 (毫秒)
FRAME_LENGTH_MS = 25
# 幀步長 (毫秒)
HOP_LENGTH_MS = 10
# 將毫秒轉換為樣本數
FRAME_LENGTH_SAMPLES = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000)
HOP_LENGTH_SAMPLES = int(SAMPLE_RATE * HOP_LENGTH_MS / 1000)

# --- 4. 數據增強參數 (Data Augmentation Parameters) ---
# 論文中提到的四種增強類型
AUGMENTATION_TYPES = [
    'noise',          # 使用 MUSAN 數據集
    'reverberation',  # 使用 RIR Noise 數據集
    'amplification',  # 改變音量
    'speed'           # 改變音頻速度，音高不變
]
# 其他增強相關參數，例如噪音/混響的比例、音量變化範圍、速度變化範圍等 (論文未詳述，需根據實際實現添加)
NOISE_SNR_RANGE = (0, 15) # SNR 範圍 (dB)
REVERB_PROB = 0.5 # 應用混響的概率
SPEED_CHANGE_RANGE = (0.9, 1.1) # 速度變化因子

# --- 5. 模型架構參數 (Model Architecture Parameters) ---
# 骨幹網路名稱
BACKBONE_NAME = 'ResNet34'
# ResNet殘差塊的通道數
RESNET_BLOCK_WIDTHS = [32, 64, 128, 256]
# 說話者嵌入向量的維度 (池化層後接128維全連接層)
EMBEDDING_DIM = 1024
# 身份分類器的輸出類別數 (即數據集中說話者的總數)
NUM_SPEAKERS = 4000 # 已被標記年齡的VoxCeleb2說話者數量(原始為5994)
# 年齡分類器的年齡組數量 (0-20, 21-30, ..., 70-100，共7組)
NUM_AGE_GROUPS = 7

# --- 6. 損失函數參數 (Loss Function Parameters) ---
# ArcFace損失參數 (s: 縮放因子, m: 邊距)
ARC_FACE_S = 64 # 64
ARC_FACE_M = 0.2
# ADAL損失函數中各損失項的權重
LAMBDA_ID = 1.0   # 身份分類損失的權重 (通常為1.0，因為其他損失是相對於它的權重)
LAMBDA_AGE = 0.1  # 年齡分類損失的權重
LAMBDA_GRL = 0.1  # 對抗年齡損失的權重 (GRL的lambda_val也可能在此設定)
ALPHA = 0.0 # 0.0
BETA = 0.005
GAMMA = 0.003
ALPHA_SCHEDULE = {
    10: 0.1,
    20: 0.5,
    30: 1.0
}



# --- 7. 訓練超參數 (Training Hyperparameters) ---
# 訓練輪數
EPOCHS = 50 # 實際訓練可能更多，這裡是一個示例
WARM_UP_EPOCHS = 10
FINETUNE_EPOCHS = 100
# 每批次訓練樣本數
BATCH_SIZE = 64
# 優化器類型
OPTIMIZER = 'Adam'
# 初始學習率
INITIAL_LR = 0.001
LEARNING_RATE_DETACH = 0.0001  # 提取器和輔助網絡的學習率
FINETUNE_LR = 0.001
# 學習率調度器設定 (多步學習率調度器)
LR_SCHEDULER_TYPE = 'MultiStepLR'
LR_DECAY_STEPS = [10, 20, 30] # 學習率在哪個epoch衰減 (論文寫10，但通常是多個點)
LR_DECAY_FACTOR = 0.1 # 學習率每次衰減的因子
# 線性預熱學習率的輪數
LR_WARMUP_EPOCHS = 2
# 最小學習率 (達到此值停止訓練)
MIN_LR = 1e-5

# --- 8. 評估設定 (Evaluation Settings) ---
# 評估指標
EVAL_METRICS = ['EER', 'mDCF']
# mDCF 計算所需的參數
P_TARGET = 1e-2
C_FA = 1
C_MISS = 1

# --- 9. 數據集相關參數 (Dataset Specific Parameters) ---
# VoxCeleb1/2 的使用設定 (例如，哪個用於訓練，哪個用於測試集構建)
VOXCELEB_TRAIN_SET = 'VoxCeleb2'
VOXCELEB_TEST_SET_CONSTRUCTION = 'VoxCeleb1'