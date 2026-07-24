import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets = targets.long()
        logpt = log_probs.gather(1, targets.unsqueeze(1))
        pt = probs.gather(1, targets.unsqueeze(1))

        focal = (1 - pt) ** self.gamma
        loss = -focal * logpt

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = loss * alpha_t.unsqueeze(1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss