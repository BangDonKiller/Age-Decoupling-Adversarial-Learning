有關於 ffmpeg 說明(for Windows environment)
- 請先至 https://github.com/GyanD/codexffmpeg/releases 下載檔案 (必須是shared)
- 加入系統變數後，且裝完 torchaudio 後，請至 .venv\Lib\site-packages\torchaudio\_extension\__init__.py 修改此行

原 code: if os.name == "nt" and (3, 8) <= sys.version_info < (3, 9):
修正後: if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):


各項分析工具介紹:
1. PCA: 主成分分析，線性找出嵌入空間中，變異量最大的幾個維度
2. LDA: 在高維嵌入空間中，盡可能找到一條直線或是一個平面，可以將目標特徵線性分得越乾淨
3. T-SNE: 將原始嵌入空間中的樣本點做降維，同時保留他們在高維空間中的鄰近關係，但不保證空間距離關係
4. SHAP: 觀察說話者嵌入當中的每個維度對於目標特徵(性別)預測的貢獻程度
5. PCA_correlation: PCA 主成分所涵蓋的變異量與目標特徵(性別)的相關程度
6. Analysis_variance_ratio: 計算目標特徵的組內距離以及組間距離，觀察模型是否可以能夠把目標特徵分得夠開(組間)以及計算除了目標特徵以外的樣本點變異量(組內)，越大代表性別以外的資訊影響越大
7. test: 做了 KNN, LR, silhouette 分數分析，觀察目標特徵於嵌入空間中的分布以及線性可分程度
8. Acoustic_Feature_Correlation: 觀察目標特徵(目前為性別)、環境因子與主成分的相關程度