"""One configurable training entry point for DeepWeeds experiments."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

import dataset
import losses
import model as models


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    use_two_gpus: bool = True


def run_dir(cfg):
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg, split):
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(net, cfg):
    return torch.optim.AdamW(models.param_groups(net, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg, steps_per_epoch):
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = int(cfg.warmup_epochs * steps_per_epoch)
    def factor(step):
        if step < warmup:
            return max(1e-5, (step + 1) / max(1, warmup))
        progress = (step - warmup) / max(1, total - warmup)
        return max(1e-5, 0.5 * (1 + math.cos(math.pi * min(1, progress))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    def __init__(self, net, decay):
        self.model = copy.deepcopy(net).eval()
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, net):
        for avg, cur in zip(self.model.parameters(), net.parameters()):
            avg.lerp_(cur.detach(), 1 - self.decay)
        for avg, cur in zip(self.model.buffers(), net.buffers()):
            avg.copy_(cur)


def train_one_epoch(net, loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=None):
    net.train()
    if cfg.init == "frozen":
        net.eval()
        raw = net.module if isinstance(net, nn.DataParallel) else net
        raw.get_classifier().train()
    total, n = 0.0, 0
    for step, (x, y, _) in enumerate(loader):
        if cfg.max_train_batches is not None and step >= cfg.max_train_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        targets = None
        if cfg.mix:
            x, targets = losses.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            logits = net(x)
            loss = (losses.mixed_loss(criterion, logits, targets) if targets else criterion(logits, y))
        scaler.scale(loss).backward()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before:
            scheduler.step()
            if ema:
                ema.update(net.module if isinstance(net, nn.DataParallel) else net)
        total += float(loss.detach()) * len(y)
        n += len(y)
    return {"train_loss": total / n, "lr": optimizer.param_groups[0]["lr"]}


def evaluate(net, loader, criterion, device, max_batches=None):
    net.eval()
    names, labels, logits, total, n = [], [], [], 0.0, 0
    with torch.inference_mode():
        for step, (x, y, filenames) in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            z = net(x)
            total += float(criterion(z, y)) * len(y)
            n += len(y)
            names.extend(filenames)
            labels.append(y.cpu().numpy())
            logits.append(z.float().cpu().numpy())
    return names, np.concatenate(labels), np.concatenate(logits), total / n


def plot_curves(history, path, title):
    frame = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(frame.epoch, frame.train_loss, label="train")
    ax[0].plot(frame.epoch, frame.val_loss, label="val")
    ax[0].set(ylabel="loss", xlabel="epoch")
    ax[0].legend()
    ax[1].plot(frame.epoch, frame.macro_f1_val, label="macro-F1 val")
    ax[1].set(ylabel="macro-F1", xlabel="epoch", ylim=(0, 1))
    ax[1].legend()
    fig.suptitle(title)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _metrics(y, z):
    from eval import compute_metrics
    p = torch.softmax(torch.from_numpy(z), dim=1).numpy()
    return compute_metrics(y, p.argmax(1), p), p


def run(cfg):
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from eval import save_predictions
    if cfg.max_train_batches is not None and cfg.save_test_predictions:
        raise ValueError("Smoke run must not open test")
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    split_info = dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    (out / "split_check.json").write_text(json.dumps(split_info, indent=2))
    train_loader = dataset.make_loader(train_df, cfg.images_dir, dataset.build_transforms(True, cfg.img_size, cfg.aug), cfg.batch_size, True, cfg.sampler, cfg.num_workers)
    val_loader = dataset.make_loader(val_df, cfg.images_dir, dataset.build_transforms(False, cfg.img_size), cfg.batch_size, False, num_workers=cfg.num_workers)
    net = models.build_model(cfg.backbone, init=cfg.init, drop_rate=cfg.drop_rate).to(device)
    tag = ("scratch" if cfg.init == "scratch" else
           str(net.pretrained_cfg.get("hf_hub_id") or net.pretrained_cfg.get("url") or "pretrained"))
    params_m = models.count_params(net)
    optimizer = build_optimizer(net, cfg)
    train_steps = min(len(train_loader), cfg.max_train_batches) if cfg.max_train_batches else len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, train_steps)
    counts = train_df.Label.value_counts().reindex(range(9), fill_value=0).to_numpy()
    weights = losses.class_weights(counts, cfg.class_weight_beta or 0).to(device)
    criterion = losses.build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma,
                                       weight=weights, alpha=weights if cfg.loss == "focal" and cfg.class_weight_beta is not None else None).to(device)
    val_criterion = nn.CrossEntropyLoss()
    ema = EMA(net, cfg.ema_decay) if cfg.ema_decay is not None else None
    wrapped = nn.DataParallel(net) if cfg.use_two_gpus and torch.cuda.device_count() > 1 else net
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    history, best = [], -1.0
    for epoch in range(1, cfg.epochs + 1):
        start = time.perf_counter()
        tr = train_one_epoch(wrapped, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        seconds = time.perf_counter() - start
        eval_net = ema.model if ema else wrapped
        _, y, z, val_loss = evaluate(eval_net, val_loader, val_criterion, device, cfg.max_val_batches)
        metrics, _ = _metrics(y, z)
        row = {"epoch": epoch, **tr, "val_loss": val_loss, "macro_f1_val": metrics["macro_f1"],
               "top1_val": metrics["top1"], "seconds": seconds}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        if row["macro_f1_val"] > best:
            best, best_epoch = row["macro_f1_val"], epoch
            torch.save({"model": copy.deepcopy((ema.model if ema else net).state_dict()),
                        "epoch": epoch, "macro_f1_val": best}, out / "best.pt")
        print(json.dumps({"exp_id": cfg.exp_id, "seed": cfg.seed, **row}), flush=True)
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    net.load_state_dict(checkpoint["model"])
    names, y, z, _ = evaluate(net, val_loader, val_criterion, device)
    metrics, p = _metrics(y, z)
    np.savez_compressed(out / "val_logits.npz", Filename=np.array(names), y_true=y, logits=z)
    save_predictions(pred_path(cfg, "val"), names, y, p)
    result = {"exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone,
              "pretrained_tag": tag, "params_m": params_m, "best_epoch": best_epoch,
              "macro_f1_val": metrics["macro_f1"], "top1_val": metrics["top1"],
              "ece_val": metrics["ece"], "seconds_per_epoch": float(np.mean([r["seconds"] for r in history]))}
    try:
        result["gmac"] = models.count_gmacs(net, cfg.img_size)
    except (ImportError, RuntimeError, ValueError) as exc:
        result["gmac"] = None
        result["gmac_error"] = str(exc)
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(test_df, cfg.images_dir, dataset.build_transforms(False, cfg.img_size), cfg.batch_size, False, num_workers=cfg.num_workers)
        names, y, z, _ = evaluate(net, test_loader, val_criterion, device)
        np.savez_compressed(out / "test_logits.npz", Filename=np.array(names), y_true=y, logits=z)
        save_predictions(pred_path(cfg, "test"), names, y, torch.softmax(torch.from_numpy(z), dim=1).numpy())
    plot_curves(history, Path(cfg.out_dir).parent / "curves" / f"{cfg.exp_id}_seed{cfg.seed}.png", f"{cfg.exp_id} {cfg.backbone} seed {cfg.seed}")
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def parse_overrides(pairs):
    hints = get_type_hints(Config)
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE: {pair}")
        key, value = pair.split("=", 1)
        if key not in hints:
            raise ValueError(f"Unknown config field: {key}")
        kind = hints[key]
        if get_origin(kind) is not None:
            kind = next((a for a in get_args(kind) if a is not type(None)), str)
        if value.lower() in ("none", "null"):
            result[key] = None
        elif kind is bool:
            if value.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"Invalid boolean: {value}")
            result[key] = value.lower() in ("true", "1")
        else:
            result[key] = kind(value)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    print(json.dumps(run(Config(**parse_overrides(args.set))), indent=2))


if __name__ == "__main__":
    main()
