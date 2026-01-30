import torch
import random
import copy

class PCGrad:
    def __init__(self, optimizer):
        self._optim = optimizer

    def step(self, losses):
        """
        按照 PCGrad 論文 Algorithm 1 實作：
        1. 分別計算每個任務的梯度 g_i
        2. 對每個 g_i，將其投影到與其衝突的其他任務 g_j 的法平面上
        3. 將所有手術後的梯度加總 (Sum) 作為最終更新方向
        """
        task_grads = []

        # 1. 取得每個任務的獨立梯度
        for loss in losses:
            self._optim.zero_grad()
            # retain_graph=True 是必要的，因為多個任務共享同一個前向傳播圖
            loss.backward(retain_graph=True)
            task_grads.append(self._get_grad())

        # 2. 準備存放「手術後」梯度的列表 (複製一份原始梯度)
        # 論文中定義 Gi_pc = gi
        pc_grads = [g.clone() for g in task_grads]

        num_conflicts = 0
        sum_cos_sim = 0.0
        total_pairs = 0

        # 3. 執行梯度手術 (兩兩比對)
        for i in range(len(task_grads)):
            # 隨機打亂其他任務的順序，增加穩定性
            others = list(range(len(task_grads)))
            others.remove(i)
            random.shuffle(others)

            for j in others:
                g_j = task_grads[j] # 參考基準是原始梯度
                
                # 計算內積
                dot_product = torch.dot(pc_grads[i], g_j)
                
                # 統計用：只計算一次每對任務的原始相似度
                if i < j:
                    norm_prod = (torch.norm(pc_grads[i]) * torch.norm(g_j) + 1e-8)
                    sum_cos_sim += (dot_product / norm_prod).item()
                    total_pairs += 1

                # 如果衝突 (內積為負)
                if dot_product < 0:
                    num_conflicts += 1
                    # 手術投影：gi = gi - ( (gi · gj) / ||gj||^2 ) * gj
                    pc_grads[i] -= (dot_product / (torch.norm(g_j) ** 2 + 1e-8)) * g_j

        # 4. 關鍵修正：將所有手術後的梯度加總 (Sum)
        # 這是論文 Algorithm 1 的最後一步，確保更新力道不會縮水
        merged_grad = torch.stack(pc_grads).sum(dim=0)

        # 5. 寫回模型並執行更新
        self._set_grad(merged_grad)
        self._optim.step()

        return {
            "conflicts": num_conflicts,
            "avg_cos_sim": sum_cos_sim / total_pairs if total_pairs > 0 else 0.0
        }

    def _get_grad(self):
        """取得模型所有參與優化的參數梯度，展平為長向量"""
        grads = []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None:
                    # 處理沒有梯度的參數 (如 backbone 被 freeze 時)
                    grads.append(torch.zeros_like(p.data).view(-1))
                else:
                    grads.append(p.grad.detach().clone().view(-1))
        return torch.cat(grads)

    def _set_grad(self, grad_vec):
        """將展平的手術後梯度寫回各個參數的 .grad 中"""
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                numel = p.data.numel()
                if p.grad is not None:
                    p.grad.data.copy_(grad_vec[idx:idx + numel].view_as(p.data))
                idx += numel