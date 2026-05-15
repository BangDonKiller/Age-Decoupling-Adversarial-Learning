import torch
import random
import copy

class PCGrad:
    def __init__(self, optimizer):
        self._optim = optimizer

    def step(self, losses):
        task_grads = []
        for loss in losses:
            self._optim.zero_grad()
            loss.backward(retain_graph=True)
            task_grads.append(self._get_grad())

        # --- 新增：觀察手術前（原始總梯度）的量級 ---
        # 這是如果不做 PCGrad，模型原本會拿到的總更新向量
        raw_grad_sum = torch.stack(task_grads).sum(dim=0)
        pre_norm = torch.norm(raw_grad_sum).item()
        # ----------------------------------------

        pc_grads = [g.clone() for g in task_grads]
        num_conflicts = 0
        sum_cos_sim = 0.0
        total_pairs = 0

        for i in range(len(task_grads)):
            others = list(range(len(task_grads)))
            others.remove(i)
            random.shuffle(others)
            for j in others:
                g_j = task_grads[j]
                dot_product = torch.dot(pc_grads[i], g_j)
                if i < j:
                    norm_prod = (torch.norm(pc_grads[i]) * torch.norm(g_j) + 1e-8)
                    sum_cos_sim += (dot_product / norm_prod).item()
                    total_pairs += 1

                if dot_product < 0:
                    num_conflicts += 1
                    pc_grads[i] -= (dot_product / (torch.norm(g_j) ** 2 + 1e-8)) * g_j

        merged_grad = torch.stack(pc_grads).sum(dim=0)

        # --- 新增：觀察手術後（投影後的總梯度）的量級 ---
        post_norm = torch.norm(merged_grad).item()
        # 計算縮減比例：post / pre (如果小於 1 代表梯度變短了)
        shrinkage_ratio = post_norm / (pre_norm + 1e-8)
        # ----------------------------------------

        self._set_grad(merged_grad)
        self._optim.step()

        return {
            "pre_norm": pre_norm,
            "post_norm": post_norm,
            "shrinkage_ratio": shrinkage_ratio
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