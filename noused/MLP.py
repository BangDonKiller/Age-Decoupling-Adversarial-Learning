import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split

# =========================
# 1. Load data
# =========================
data = torch.load("./result/ECAPA-TDNN/TIMIT_ECAPA-TDNN_embeddings.pt")

X = data["embeddings"].numpy()      # [N, D]
y = np.array(data["ages"])          # 年齡組 label，例如 0~6

# =========================
# 2. Device setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 3. Logistic Regression by torch
# =========================
MLP = nn.Sequential(
    nn.Linear(X.shape[1], 96),
    nn.ReLU(),
    nn.Linear(96, 7),
    nn.LogSoftmax(dim=1),
).to(device)  # 將模型丟到 GPU

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(MLP.parameters(), lr=0.001)
# =========================

# 4. Train / test split
# ========================= 
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Convert to torch tensors and move to device
X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32).to(device)
y_tr_tensor = torch.tensor(y_tr, dtype=torch.long).to(device)
X_te_tensor = torch.tensor(X_te, dtype=torch.float32).to(device)
y_te_tensor = torch.tensor(y_te, dtype=torch.long).to(device)

# =========================
# 5. Training loop
# =========================
num_epochs = 100
for epoch in range(num_epochs):
    MLP.train()
    optimizer.zero_grad()
    outputs = MLP(X_tr_tensor)
    loss = criterion(outputs, y_tr_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        
# =========================
# 6. Evaluation
# =========================
MLP.eval()
with torch.no_grad():
    outputs = MLP(X_te_tensor)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y_te_tensor).sum().item()
    total = y_te_tensor.size(0)
    accuracy = correct / total
    print("MLP (age group) acc:", accuracy)