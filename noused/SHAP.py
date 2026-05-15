# 讓模型做預測性別的任務，然後用 SHAP 來解釋模型的預測結果
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import torch

print("\n開始 SHAP 分析…")

data = torch.load("./result/ECAPA-TDNN/VoxCeleb2_ECAPA-TDNN_embeddings.pt")
all_embeddings = data["embeddings"]          # Tensor [N, D]
genders = data["genders"]      # e.g. ['m', 'f', 'm', ...]

# 1. 取得 embeddings 和 gender
X = all_embeddings.numpy()
# 假設 gender 是 ["male","female","male"...]
# 我們將 gender 轉成 0/1
y = np.array([1 if g == "m" else 0 for g in genders])

print("Classifier training…")

# 2. 訓練一個簡單的性別預測分類器
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

print("用 classifier 進行預測…")
preds = clf.predict(X)

print("建立 SHAP explainer…")

# 3. 建立 SHAP explainer
# 使用線性模型，要用 LinearExplainer
explainer = shap.LinearExplainer(clf, X)

print("計算 SHAP values，可能會稍久…")
shap_values = explainer.shap_values(X)

print("SHAP 分析完成！")

# 4. 輸出 summary plot
print("畫出 summary plot…")
shap.summary_plot(shap_values, X, show=False)

plt.title("SHAP Summary Plot for Gender Prediction")
plt.savefig("shap_summary_plot.png")
plt.close()

print("已將 SHAP summary plot 儲存為 shap_summary_plot.png")
print("SHAP 分析結束！")


import numpy as np

# shap_values shape: [num_samples, num_features]
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

# 轉成 (feature_index, importance) pair
feat_importance = list(enumerate(mean_abs_shap))

# 按重要性排序
feat_importance_sorted = sorted(feat_importance, key=lambda x: x[1], reverse=True)

# 取前 10
top10 = feat_importance_sorted[:10]
print("Top 10 features by mean absolute SHAP:", top10)
