"""Reproducible DeepWeeds experiment runner for Kaggle T4 x2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import timm
import torchvision
from PIL import Image

import benchmark
import dataset
import inference
import model as models
import train


def _root():
    return Path(__file__).resolve().parents[3]


def _eval():
    if str(_root()) not in sys.path:
        sys.path.insert(0, str(_root()))
    import eval as ev
    return ev


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float))


def _load_result(cfg):
    directory = train.run_dir(cfg)
    path = directory / "summary.json"
    if path.exists():
        saved = json.loads((directory / "config.json").read_text())
        if saved != asdict(cfg):
            raise ValueError(f"Existing run has a different config: {directory}")
        return json.loads(path.read_text())
    overrides = [f"{key}={str(value).lower() if isinstance(value, bool) or value is None else value}"
                 for key, value in asdict(cfg).items()]
    subprocess.run([sys.executable, str(Path(train.__file__).resolve()), "--set", *overrides], check=True)
    return json.loads(path.read_text())


def restore_completed_runs(archive_path, output):
    """Restore only runs with config, checkpoint and summary from an evidence zip."""
    output = Path(output)
    count = 0
    with zipfile.ZipFile(archive_path) as saved:
        names = set(saved.namelist())
        complete = set()
        for member in names:
            if member.startswith("runs/") and member.endswith("/summary.json"):
                directory = member.rsplit("/", 1)[0]
                if {f"{directory}/config.json", f"{directory}/best.pt"}.issubset(names):
                    complete.add(directory)
        for member in saved.namelist():
            if member.endswith("/"):
                continue
            run_file = any(member.startswith(directory + "/") for directory in complete)
            artifact = any(member.startswith((f"curves/{directory.split('/')[1]}_{directory.split('/')[2]}.",
                                             f"predictions/{directory.split('/')[1]}_{directory.split('/')[2]}_"))
                           for directory in complete)
            if run_file or artifact:
                target = output / member
                target.parent.mkdir(parents=True, exist_ok=True)
                with saved.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                count += 1
    return count


def restore_completed_tree(source, output):
    """Restore completed runs when Kaggle extracted the previous evidence zip."""
    source, output = Path(source), Path(output)
    count = 0
    for summary in (source / "runs").glob("*/seed*/summary.json"):
        directory = summary.parent
        if not (directory / "config.json").is_file() or not (directory / "best.pt").is_file():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                target = output / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                count += 1
        experiment, seed = directory.parent.name, directory.name
        for pattern in (f"curves/{experiment}_{seed}.*", f"predictions/{experiment}_{seed}_*"):
            for path in source.glob(pattern):
                if path.is_file():
                    target = output / path.relative_to(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    count += 1
    return count


def _config(data, output, **kw):
    if os.environ.get("DEEPWEEDS_FAST") == "1":
        kw["num_workers"] = int(os.environ.get("DEEPWEEDS_WORKERS", "4"))
        kw["fast_mode"] = True
    return train.Config(images_dir=str(data / "images"), labels_dir=str(data / "labels"),
                        out_dir=str(output / "runs"), pred_dir=str(output / "predictions"),
                        **kw)


def _ckpt_predict(cfg, split, view=None):
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir)
    df = val_df if split == "val" else test_df
    loader = dataset.make_loader(df, cfg.images_dir,
                                 dataset.build_transforms(False, cfg.img_size), cfg.batch_size,
                                 False, num_workers=cfg.num_workers)
    net = models.build_model(cfg.backbone, pretrained=False, init="scratch", drop_rate=cfg.drop_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(train.run_dir(cfg) / "best.pt", map_location=device, weights_only=True)
    net.load_state_dict(ckpt["model"])
    net.to(device)
    return inference.predict_logits(net, loader, device, view), net


def _score(y, probs):
    metric = _eval().compute_metrics(y, probs.argmax(1), probs)
    return {key: float(metric[key]) for key in ("macro_f1", "top1", "balanced_acc", "ece", "nll")}


def _eda(data, output):
    frames = dataset.load_split(data / "labels")
    split = dataset.check_split(*frames, data / "images")
    _json(output / "evidence" / "split_check.json", split)
    _json(output / "evidence" / "software.json", {"python": sys.version,
          "torch": torch.__version__, "torchvision": torchvision.__version__,
          "timm": timm.__version__, "numpy": np.__version__, "pandas": pd.__version__})
    manifest = {"images": 17509, "fold": 0, "labels": {}}
    for name in ("labels.csv", "train_subset0.csv", "val_subset0.csv", "test_subset0.csv"):
        manifest["labels"][name] = hashlib.sha256((data / "labels" / name).read_bytes()).hexdigest()
    archive = data / "images.zip"
    if archive.exists():
        manifest["images_md5"] = hashlib.md5(archive.read_bytes()).hexdigest()
    _json(output / "evidence" / "dataset_manifest.json", manifest)
    counts = pd.DataFrame({part: frame.Label.value_counts().reindex(range(9), fill_value=0)
                           for part, frame in zip(("train", "val", "test"), frames)})
    counts.index = dataset.CLASS_NAMES
    counts.to_csv(output / "evidence" / "class_counts.csv")
    ax = counts.plot.bar(figsize=(12, 5))
    ax.set(ylabel="Images", title="DeepWeeds fold 0")
    plt.tight_layout()
    plt.savefig(output / "evidence" / "class_counts.png", dpi=160)
    plt.close()
    frame = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(9, 3, figsize=(8, 22))
    for class_id, row_axes in enumerate(axes):
        examples = frame[frame.Label == class_id].Filename.head(3)
        for ax, name in zip(row_axes, examples):
            with Image.open(data / "images" / name) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(dataset.CLASS_NAMES[class_id], fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "evidence" / "sample_images.png", dpi=120)
    plt.close(fig)
    return split


def smoke(data, output):
    output.mkdir(parents=True, exist_ok=True)
    _eda(data, output)
    if torch.cuda.is_available():
        _json(output / "evidence" / "gpu.json", {"count": torch.cuda.device_count(),
               "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               "torch": torch.__version__})
    cfg = _config(data, output, exp_id="SMOKE", backbone="mobilenetv3_small_050",
                  init="scratch", epochs=1, batch_size=8, img_size=128, num_workers=2,
                  max_train_batches=2, max_val_batches=2)
    result = train.run(cfg)
    prediction = _eval().read_pred(str(train.pred_path(cfg, "val")))
    if len(prediction.y_true) != 3501:
        raise AssertionError("Smoke validation row count mismatch")
    _json(output / "evidence" / "smoke.json", result)
    return result


def _inference_candidates(cfgs):
    ev = _eval()
    rows, by_seed = [], {}
    for cfg in cfgs:
        names, y, base = _ckpt_predict(cfg, "val")[0]
        flip_names, flip_y, flip = _ckpt_predict(cfg, "val", inference.view_hflip)[0]
        if names != flip_names or not np.array_equal(y, flip_y):
            raise AssertionError("TTA changed image order")
        temp = inference.fit_temperature(base, y)
        temp_flip = inference.fit_temperature((base + flip) / 2, y)
        probs = {
            "I00": inference.apply_temperature(base, 1),
            "I01": inference.aggregate_views([base, flip], "prob"),
            "I02": inference.aggregate_views([base, flip], "logit"),
            "I03": inference.apply_temperature(base, temp),
            "I04": inference.apply_temperature((base + flip) / 2, temp_flip),
        }
        by_seed[cfg.seed] = {"cfg": cfg, "names": names, "y": y, "base": base,
                              "flip": flip, "temperature": temp,
                              "temperature_flip": temp_flip, "probs": probs}
        for method, p in probs.items():
            rows.append({"exp_id": method, "seed": cfg.seed, **_score(y, p)})
    frame = pd.DataFrame(rows)
    means = frame.groupby("exp_id")[["macro_f1", "ece"]].mean()
    best = means.sort_values(["macro_f1", "ece"], ascending=[False, True]).index[0]
    return frame, by_seed, best


def _method_probs(method, base, flip, temperature):
    if method == "I00":
        return inference.apply_temperature(base, 1)
    if method == "I01":
        return inference.aggregate_views([base, flip], "prob")
    if method == "I02":
        return inference.aggregate_views([base, flip], "logit")
    if method == "I03":
        return inference.apply_temperature(base, temperature)
    if method == "I04":
        return inference.apply_temperature((base + flip) / 2, temperature)
    raise ValueError(method)


def _final_predictions(cfgs, method, temperatures, output):
    ev = _eval()
    results = []
    for cfg in cfgs:
        (names, y, base), net = _ckpt_predict(cfg, "test")
        flip = None
        if method in ("I01", "I02", "I04"):
            flip_names, flip_y, flip = _ckpt_predict(cfg, "test", inference.view_hflip)[0]
            if names != flip_names or not np.array_equal(y, flip_y):
                raise AssertionError("Test TTA image order mismatch")
        p = _method_probs(method, base, flip, temperatures[cfg.seed])
        path = train.pred_path(cfg, "test")
        ev.save_predictions(path, names, y, p)
        np.savez_compressed(train.run_dir(cfg) / "test_logits.npz", Filename=np.array(names), y_true=y, logits=base)
        if cfg.exp_id == "F01":
            uncal_method = {"I03": "I00", "I04": "I02"}.get(method, method)
            ev.save_predictions(path.with_name(path.stem + "_uncal.csv"), names, y,
                                _method_probs(uncal_method, base, flip, 1))
        results.append({"exp_id": cfg.exp_id, "seed": cfg.seed, "method": method, **_score(y, p)})
    return results


def full(data, output, epochs=10, final_epochs=12):
    output.mkdir(parents=True, exist_ok=True)
    split = _eda(data, output)
    gpu = {"count": torch.cuda.device_count(), "names": [torch.cuda.get_device_name(i)
           for i in range(torch.cuda.device_count())], "torch": torch.__version__}
    _json(output / "evidence" / "gpu.json", gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("Full lab requires GPU")
    # B01/B02 retain their configs so completed Kaggle results can be restored.
    # ConvNeXt and EfficientNet were too slow in observed T4 x2 runs.
    backbones = ["resnet50", "resnet18", "resnet34",
                 "regnetx_002", "vit_tiny_patch16_224"]
    backbone_rows = []
    for index, backbone in enumerate(backbones, 1):
        cfg = _config(data, output, exp_id=f"B{index:02d}", backbone=backbone,
                      epochs=epochs, batch_size=64,
                      num_workers=2 if index == 1 else 0)
        backbone_rows.append(_load_result(cfg))
    best_backbone = max(backbone_rows, key=lambda r: r["macro_f1_val"])["backbone"]
    _json(output / "evidence" / "backbone_selection.json",
          {"criterion": "macro_f1_val", "selected": best_backbone, "candidates": backbone_rows})
    baseline = _config(data, output, exp_id="T00", backbone=best_backbone,
                       epochs=epochs, batch_size=64, num_workers=0)
    training_rows = [_load_result(baseline)]
    choices = [
        ("T01", {"init": "frozen"}, "initialization"),
        ("T02", {"init": "scratch"}, "initialization"),
        ("T03", {"aug": "color"}, "augmentation"),
        ("T04", {"aug": "randaug"}, "augmentation"),
        ("T05", {"loss": "ls", "label_smoothing": 0.1}, "loss"),
        ("T06", {"loss": "focal", "focal_gamma": 2.0}, "loss"),
    ]
    for exp_id, changes, axis in choices:
        cfg = replace(baseline, exp_id=exp_id, **changes)
        training_rows.append({**_load_result(cfg), "axis": axis, "change": json.dumps(changes)})
    # Select each factor on validation only; a combination is then retrained.
    selected = {}
    for axis in ("initialization", "augmentation", "loss"):
        candidates = [r for r in training_rows if r.get("axis") == axis]
        winner = max(candidates + [training_rows[0]], key=lambda r: r["macro_f1_val"])
        if winner["exp_id"] != "T00":
            selected.update(dict(next(c for c in choices if c[0] == winner["exp_id"])[1]))
    _json(output / "evidence" / "recipe_selection.json", selected)
    final_cfgs = [replace(baseline, exp_id="F01", seed=seed, epochs=final_epochs, **selected)
                  for seed in (0, 1, 2)]
    baseline_cfgs = [replace(baseline, exp_id="F00", seed=seed, epochs=final_epochs)
                     for seed in (0, 1, 2)]
    for cfg in baseline_cfgs + final_cfgs:
        _load_result(cfg)
    inf_frame, by_seed, method = _inference_candidates(final_cfgs)
    inf_frame.to_csv(output / "evidence" / "inference_val.csv", index=False)
    _json(output / "evidence" / "inference_selection.json", {"method": method,
          "temperatures": {str(k): {"one_view": v["temperature"],
                                     "two_view": v["temperature_flip"]}
                           for k, v in by_seed.items()}})
    for cfg in final_cfgs:
        info = by_seed[cfg.seed]
        _eval().save_predictions(train.pred_path(cfg, "val"), info["names"], info["y"],
                                 info["probs"][method])
    final_rows = _final_predictions(baseline_cfgs, "I00", {s: 1 for s in (0, 1, 2)}, output)
    final_rows += _final_predictions(final_cfgs, method,
                                    {s: by_seed[s]["temperature_flip" if method == "I04" else "temperature"]
                                     for s in (0, 1, 2)}, output)
    final_frame = pd.DataFrame(final_rows)
    final_frame.to_csv(output / "evidence" / "final_scores.csv", index=False)
    per_class_rows = []
    for cfg in baseline_cfgs + final_cfgs:
        pred = _eval().read_pred(str(train.pred_path(cfg, "test")))
        metric = _eval().compute_metrics(pred.y_true, pred.y_pred, pred.probs)
        for idx, name in enumerate(dataset.CLASS_NAMES):
            per_class_rows.append({"exp_id": cfg.exp_id, "seed": cfg.seed, "class": name,
                                   "support": int(metric["support"][idx]),
                                   "precision": float(metric["precision"][idx]),
                                   "recall": float(metric["recall"][idx]),
                                   "f1": float(metric["f1"][idx])})
    per_class_frame = pd.DataFrame(per_class_rows)
    per_class_frame.to_csv(output / "evidence" / "per_class.csv", index=False)
    _, net = _ckpt_predict(final_cfgs[0], "val")
    latency_rows = []
    for batch in (1, 32):
        for dtype in ("fp32", "amp"):
            latency_rows.append({"method": "I00", **benchmark.latency_report(net, batch, 224,
                                  dtype=dtype, device="cuda", iters=50)})
    two_view = benchmark.tta_latency(net, 2, batch_size=1, img_size=224,
                                     device="cuda", iters=50)
    latency_rows.append({"method": "I01/I02/I04", **two_view})
    latency = pd.DataFrame(latency_rows)
    latency.to_csv(output / "evidence" / "latency.csv", index=False)
    one = latency[(latency.method == "I00") & (latency.batch == 1) & (latency.dtype == "fp32")].iloc[0]
    for metric in ("p50", "p95", "p99"):
        inf_frame[f"latency_{metric}_ms"] = inf_frame.exp_id.map(
            {"I00": one[metric], "I01": two_view[metric], "I02": two_view[metric],
             "I03": one[metric], "I04": two_view[metric]})
    inf_frame["relative_cost"] = inf_frame.latency_p50_ms / one.p50
    inf_frame.to_csv(output / "evidence" / "inference_val.csv", index=False)
    with pd.ExcelWriter(output / "results.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(backbone_rows).to_excel(writer, sheet_name="Backbones", index=False)
        pd.DataFrame(training_rows).to_excel(writer, sheet_name="Training", index=False)
        inf_frame.to_excel(writer, sheet_name="Inference", index=False)
        final_frame.to_excel(writer, sheet_name="Final", index=False)
        per_class_frame.to_excel(writer, sheet_name="PerClass", index=False)
        latency.to_excel(writer, sheet_name="Latency", index=False)
        summary = pd.concat([pd.DataFrame(backbone_rows), pd.DataFrame(training_rows)])
        summary.sort_values("macro_f1_val", ascending=False).head(10).to_excel(writer, sheet_name="Summary", index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
    final_mean = final_frame[final_frame.exp_id == "F01"].macro_f1.mean()
    final_std = final_frame[final_frame.exp_id == "F01"].macro_f1.std(ddof=1)
    base_mean = final_frame[final_frame.exp_id == "F00"].macro_f1.mean()
    report = (f"# DeepWeeds Lab Day 2\n\n"
              f"## Data and method\n\nOriginal DeepWeeds fold 0, 9 classes, {split['n']}. "
              "Train updates weights; validation chooses checkpoints, recipe and inference; "
              "test is scored after selection. ImageNet pretrained weights, AdamW, cosine schedule, "
              "224px input and mixed precision are used unless an experiment config says otherwise. "
              f"GPU: {', '.join(gpu['names'])}; PyTorch {gpu['torch']}.\n\n"
              "## Backbone comparison (validation)\n\n"
              + pd.DataFrame(backbone_rows)[["exp_id", "backbone", "macro_f1_val", "top1_val", "params_m", "gmac", "seconds_per_epoch"]].to_markdown(index=False)
              + f"\n\nSelected: {best_backbone}, highest validation macro-F1.\n\n"
              "## Training recipe (validation)\n\n"
              + pd.DataFrame(training_rows)[["exp_id", "macro_f1_val", "top1_val", "best_epoch"]].to_markdown(index=False)
              + f"\n\nSelected factors by validation: `{json.dumps(selected)}`. "
              "One factor changes at a time relative to T00; the combination is retrained.\n\n"
              "## Inference selection (validation)\n\n"
              + inf_frame.groupby("exp_id")[["macro_f1", "ece"]].mean().reset_index().to_markdown(index=False)
              + f"\n\nSelected method: {method}. Temperature is fitted on validation only.\n\n"
              "## Final test (three seeds)\n\n"
              + final_frame.to_markdown(index=False)
              + f"\n\nFinal macro-F1: {final_mean:.4f} +/- {final_std:.4f}; "
              f"baseline: {base_mean:.4f}; delta: {final_mean - base_mean:+.4f}. "
              "The standard deviation is sample std (ddof=1).\n\n"
              "## Per-class test F1, final configuration\n\n"
              + per_class_frame[per_class_frame.exp_id == "F01"].groupby("class")[["support", "f1"]].mean().reset_index().to_markdown(index=False)
              + "\n\n## Latency\n\n"
              + latency.to_markdown(index=False)
              + "\n\nLatency excludes preprocessing, uses 10 warmup and 50 synchronized iterations. "
              "See `evidence/`, `runs/`, `curves/`, `predictions/`, and `results.xlsx` for raw evidence. "
              "Single-seed screening is subject to noise; results may vary across GPU/software versions.\n")
    (output / "report.md").write_text(report)
    ev = _eval()
    for exp_id in ("F00", "F01"):
        args = [sys.executable, str(_root() / "eval.py"), "score", "--pred",
                str(output / "predictions" / f"{exp_id}_seed*_test.csv"),
                "--test-csv", str(data / "labels" / "test_subset0.csv"),
                "--out", str(output / "evidence" / f"eval_{exp_id}")]
        subprocess.run(args, check=True)
    subprocess.run([sys.executable, str(_root() / "eval.py"), "grade",
                    "--final", str(output / "predictions" / "F01_seed*_test.csv"),
                    "--baseline", str(output / "predictions" / "F00_seed*_test.csv"),
                    "--uncal", str(output / "predictions" / "F01_seed*_test_uncal.csv"),
                    "--final-val", str(output / "predictions" / "F01_seed*_val.csv"),
                    "--latency-p95-ms", str(float(latency.iloc[0].p95)),
                    "--latency-method", "proper",
                    "--test-csv", str(data / "labels" / "test_subset0.csv"),
                    "--val-csv", str(data / "labels" / "val_subset0.csv"),
                    "--out", str(output / "evidence" / "grade")], check=True)
    _json(output / "evidence" / "completed.json", {"selected_backbone": best_backbone,
           "selected_recipe": selected, "selected_inference": method, "gpu": gpu})
    return final_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "full"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--final-epochs", type=int, default=12)
    args = parser.parse_args()
    result = smoke(args.data, args.output) if args.phase == "smoke" else full(args.data, args.output,
                                                args.epochs, args.final_epochs)
    print(result)


if __name__ == "__main__":
    main()
