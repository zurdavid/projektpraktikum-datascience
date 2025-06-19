import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha:float=0.25, gamma:float =2.0, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.pos_weight = (
            pos_weight if pos_weight else torch.tensor([4.0], dtype=torch.float)
        )
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none", pos_weight=self.pos_weight
        )
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


class PenalizedLoss(nn.Module):
    def __init__(self, pos_weight, false_positive_penalty=2.0):
        super().__init__()
        self.fp_penalty = false_positive_penalty
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    def forward(self, logits, targets):
        losses = self.bce(logits, targets)
        probs = torch.sigmoid(logits.detach())
        preds = (probs > 0.5).float()

        fp_mask = (preds == 1) & (targets == 0)
        loss = losses * (1 + self.fp_penalty * fp_mask.float())
        return loss.mean()


class WertkaufLoss(nn.Module):
    def __init__(
        self, base_loss_weight=1.0, fp_penalty_weight=2.0, threshold=0.5, malus=10
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.base_loss_weight = base_loss_weight
        self.fp_penalty_weight = fp_penalty_weight
        self.threshold = threshold
        self.malus = malus
        self.loss_fn_takes_damage = True

    def forward(self, logits, targets, damage):
        # Compute base BCE loss
        base_loss = self.bce(logits, targets)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        yhat = (probs > self.threshold).float()

        res = torch.zeros_like(damage, dtype=torch.float32)
        # Case 1: FRAUD caught
        res += ((targets == 1) & (yhat == 1)).float() * 5
        # res += ((y == 1) & (yhat == 1)) * damage
        # Case 2: False positive
        res -= ((targets == 0) & (yhat == 1)).float() * self.malus
        # Case 3: FRAUD missed
        res -= ((targets == 1) & (yhat == 0)) * damage

        # Penalize false positives
        penalty = res.sum().float() / logits.size(0) * -1

        # Combine loss
        total_loss = (
            self.base_loss_weight * base_loss + self.fp_penalty_weight * penalty
        )

        return total_loss
