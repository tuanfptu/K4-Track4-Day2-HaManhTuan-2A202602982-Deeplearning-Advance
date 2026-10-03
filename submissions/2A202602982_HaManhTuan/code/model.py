"""Backbones, classifier heads, and parameter groups."""
import torch
import timm

SUGGESTED_BACKBONES = {"resnet50": "resnet50", "resnext50": "resnext50_32x4d", "convnext_tiny": "convnext_tiny", "deit_small": "deit_small_patch16_224", "swin_tiny": "swin_tiny_patch4_window7_224", "efficientnet_b0": "efficientnet_b0", "mobilenetv3": "mobilenetv3_large_100"}

def _head_names(model):
    ids = {id(p) for p in model.get_classifier().parameters()}
    return {name for name, p in model.named_parameters() if id(p) in ids}

def build_model(name, pretrained=True, num_classes=9, drop_rate=0.0, init="finetune"):
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"Unknown init: {init}")
    net = timm.create_model(name, pretrained=bool(pretrained and init != "scratch"), num_classes=num_classes, drop_rate=drop_rate)
    if init == "frozen":
        freeze_backbone(net)
    return net

def freeze_backbone(model):
    head = _head_names(model)
    if not head:
        raise ValueError("Classifier head not found")
    for name, p in model.named_parameters():
        p.requires_grad = name in head

def param_groups(model, lr_backbone, lr_head, weight_decay):
    head = _head_names(model)
    groups = {(h, d): [] for h in (False, True) for d in (False, True)}
    for name, p in model.named_parameters():
        if p.requires_grad:
            groups[(name in head, p.ndim > 1)].append(p)
    return [{"params": ps, "lr": lr_head if h else lr_backbone, "weight_decay": weight_decay if d else 0.0} for (h, d), ps in groups.items() if ps]

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def count_gmacs(model, img_size=224):
    from fvcore.nn import FlopCountAnalysis
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return FlopCountAnalysis(model, torch.zeros(1, 3, img_size, img_size, device=device)).total() / 1e9
    finally:
        model.train(was_training)
