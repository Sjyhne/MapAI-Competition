import torch
import torch.nn as nn

__all__ = ["BinaryTverskyLoss"]


class BinaryTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1, gamma: float = 1.):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.gamma = gamma

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)

        # True Positives, False Positives & False Negatives
        TP = (preds * targets).sum(dim=[1,2,3])
        FP = ((1 - targets) * preds).sum(dim=[1,2,3])
        FN = (targets * (1 - preds)).sum(dim=[1,2,3])

        tversky = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        return (1. - tversky).pow(1 / self.gamma).mean()