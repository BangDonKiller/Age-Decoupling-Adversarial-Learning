import torch.nn as nn
from torch.autograd import Function

class GradientReversalFunction(Function):
    """
    實現梯度反轉層的前向和反向傳播。
    在前向傳播中，它是一個恆等操作（不做任何改變）。
    在反向傳播中，它將梯度乘以一個負數（通常是 -lambda）。
    """
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x) # 在前向傳播中，數據不變

    @staticmethod
    def backward(ctx, grad_output):
        # 在反向傳播中，梯度被乘以 -lambda_val
        return grad_output.neg() * ctx.lambda_val, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_val=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)