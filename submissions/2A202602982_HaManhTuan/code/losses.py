"""Loss functions and Mixup/CutMix."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float))

    def forward(self, logits, target):
        log_pt = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -(1 - log_pt.exp()).pow(self.gamma) * log_pt
        return (loss if self.alpha is None else loss * self.alpha[target]).mean()

def build_criterion(kind="ce", **kw):
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        return nn.CrossEntropyLoss(weight=kw["weight"])
    raise ValueError(f"Unknown loss: {kind}")

def class_weights(counts, beta=0.0):
    counts = np.asarray(counts, dtype=float)
    if np.any(counts <= 0) or not 0 <= beta < 1:
        raise ValueError("Invalid counts or beta")
    w = 1 / counts if beta == 0 else (1 - beta) / (1 - np.power(beta, counts))
    return torch.tensor(w / w.mean(), dtype=torch.float)

def mix_batch(x, y, alpha=1.0, mode="cutmix"):
    if mode not in ("mixup", "cutmix") or alpha <= 0:
        raise ValueError("Invalid mix mode/alpha")
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        out = lam * x + (1 - lam) * x[perm]
    else:
        out = x.clone()
        h, w = x.shape[-2:]
        rw, rh = int(w * np.sqrt(1 - lam)), int(h * np.sqrt(1 - lam))
        cx, cy = np.random.randint(w), np.random.randint(h)
        x1, x2 = max(0, cx - rw // 2), min(w, cx + rw // 2)
        y1, y2 = max(0, cy - rh // 2), min(h, cy + rh // 2)
        out[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - (x2 - x1) * (y2 - y1) / (h * w)
    return out, (y, y[perm], lam)

def mixed_loss(criterion, logits, targets):
    a, b, lam = targets
    return lam * criterion(logits, a) + (1 - lam) * criterion(logits, b)
