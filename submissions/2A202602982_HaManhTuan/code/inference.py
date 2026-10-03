"""Inference views, ensembles, calibration, and Conv-BN fusion."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def predict_logits(model, loader, device, view=None):
    model.eval()
    names, labels, logits = [], [], []
    with torch.inference_mode():
        for x, y, filenames in loader:
            x = x.to(device)
            if view is not None:
                x = view(x)
            z = model(x)
            names.extend(filenames)
            labels.append(y.numpy())
            logits.append(z.float().cpu().numpy())
    return names, np.concatenate(labels), np.concatenate(logits)

def view_identity(x):
    return x

def view_hflip(x):
    return torch.flip(x, dims=(-1,))

def views_multicrop(x, crop):
    h, w = x.shape[-2:]
    if crop > h or crop > w:
        raise ValueError("crop exceeds image")
    return [x[:, :, y:y + crop, xx:xx + crop] for y, xx in
            ((0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop), ((h - crop)//2, (w - crop)//2))]

def views_multiscale(x, sizes):
    return [F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False) for size in sizes]

def aggregate_views(logits_per_view, space="prob"):
    views = np.asarray(logits_per_view)
    if views.ndim != 3 or not len(views):
        raise ValueError("Expected K x N x C logits")
    if space == "prob":
        z = views - views.max(axis=2, keepdims=True)
        p = np.exp(z)
        return (p / p.sum(axis=2, keepdims=True)).mean(axis=0)
    if space == "logit":
        z = views.mean(axis=0)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)
    raise ValueError(f"Unknown aggregation: {space}")

def ensemble_probs(list_of_probs):
    arrays = [np.asarray(p) for p in list_of_probs]
    if not arrays or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Ensemble probabilities must have equal shapes")
    return np.mean(arrays, axis=0)

def fit_temperature(val_logits, val_labels):
    z = torch.as_tensor(val_logits, dtype=torch.float64)
    y = torch.as_tensor(val_labels, dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_t.detach().exp())

def apply_temperature(logits, T):
    if T <= 0:
        raise ValueError("T must be positive")
    z = np.asarray(logits, dtype=float) / T
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)

def fuse_conv_bn(model):
    result = copy.deepcopy(model).eval()
    def visit(module):
        names = list(module._modules)
        for prev, cur in zip(names, names[1:]):
            a, b = module._modules[prev], module._modules[cur]
            if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                module._modules[prev] = torch.nn.utils.fusion.fuse_conv_bn_eval(a, b)
                module._modules[cur] = nn.Identity()
        for child in module.children():
            visit(child)
    visit(result)
    return result
