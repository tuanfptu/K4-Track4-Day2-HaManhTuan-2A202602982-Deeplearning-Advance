"""Supplementary validation-only pipeline and backbone checks on the training server."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

import benchmark
import dataset
import model
import train


def run(output):
    output = Path(output).resolve()
    if not (output / "evidence/completed.json").is_file():
        raise FileNotFoundError("Completed original lab required")
    evidence = output / "evidence"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Run this evidence check on the training GPU")
    train.set_seed(0, True)
    configs = []
    for exp_id in [f"B{i:02d}" for i in range(1, 7)]:
        path = output / "runs" / exp_id / "seed0" / "config.json"
        configs.append(train.Config(**json.loads(path.read_text())))
    first = configs[0]
    train_df, _, _ = dataset.load_split(first.labels_dir)
    rows = train_df.groupby("Label", sort=True).head(1).sort_values("Label")
    image_dir = Path(first.images_dir)
    transform = dataset.build_transforms(True, 128, "basic")
    normal = torch.tensor(dataset.IMAGENET_MEAN)[:, None, None]
    scale = torch.tensor(dataset.IMAGENET_STD)[:, None, None]
    images, labels = [], []
    fig, axes = plt.subplots(9, 2, figsize=(6, 24))
    for index, row in enumerate(rows.itertuples()):
        with Image.open(image_dir / row.Filename) as source:
            original = source.convert("RGB")
            augmented = transform(original)
            axes[index, 0].imshow(original)
        axes[index, 0].set_title(dataset.CLASS_NAMES[int(row.Label)] + " original")
        axes[index, 1].imshow((augmented * scale + normal).clamp(0, 1).permute(1, 2, 0))
        axes[index, 1].set_title(dataset.CLASS_NAMES[int(row.Label)] + " augmented")
        for ax in axes[index]:
            ax.axis("off")
        images.append(augmented)
        labels.append(int(row.Label))
    fig.tight_layout()
    fig.savefig(evidence / "augmentation_check.png", dpi=120)
    plt.close(fig)
    x = torch.stack(images).to(device)
    y = torch.tensor(labels, device=device)
    net = model.build_model("mobilenetv3_small_050", pretrained=False, init="scratch").to(device)
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.005, weight_decay=0)
    history = []
    for step in range(201):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(net(x), y)
        history.append({"step": step, "loss": float(loss.detach())})
        if step == 200 or (step >= 20 and loss.item() < 0.05):
            break
        loss.backward()
        optimizer.step()
    pd.DataFrame(history).to_csv(evidence / "one_batch_overfit.csv", index=False)
    check = {"model": "mobilenetv3_small_050", "pretrained": False,
             "classes_in_batch": len(labels), "batch_size": len(labels),
             "initial_loss": history[0]["loss"], "ln_9": 2.197224577,
             "final_loss": history[-1]["loss"], "steps": history[-1]["step"],
             "reached_loss_below_0.05": history[-1]["loss"] < 0.05,
             "timing": "retrospective check after the full lab; not a preflight run"}
    (evidence / "pipeline_check.json").write_text(json.dumps(check, indent=2))
    print("Pipeline check:", check, flush=True)
    del net, optimizer, x, y
    torch.cuda.empty_cache()
    latency_rows = []
    for cfg in configs:
        net = model.build_model(cfg.backbone, pretrained=False, init="scratch").to(device)
        checkpoint = torch.load(train.run_dir(cfg) / "best.pt", map_location=device,
                                weights_only=True)
        net.load_state_dict(checkpoint["model"])
        measured = benchmark.latency_report(net, 1, cfg.img_size, dtype="fp32",
                                             device="cuda", warmup=10, iters=50)
        latency_rows.append({"exp_id": cfg.exp_id, "backbone": cfg.backbone, **measured})
        print(cfg.exp_id, measured, flush=True)
        del net, checkpoint
        torch.cuda.empty_cache()
    pd.DataFrame(latency_rows).to_csv(evidence / "backbone_latency.csv", index=False)
    (evidence / "audit_evidence_completed.json").write_text(json.dumps(
        {"pipeline": check, "backbones_measured": [r["exp_id"] for r in latency_rows]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
