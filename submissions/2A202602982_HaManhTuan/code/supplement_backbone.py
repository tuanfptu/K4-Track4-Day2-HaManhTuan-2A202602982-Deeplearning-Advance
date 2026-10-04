"""Add the required ResNeXt comparison to a completed server lab."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from PIL import Image

import benchmark
import dataset
import run_lab
import train


def supplement(output):
    output = Path(output)
    completed = output / "evidence" / "completed.json"
    if not completed.is_file():
        raise FileNotFoundError(f"Finish the original lab first: {completed}")
    baseline = output / "runs" / "B01" / "seed0" / "config.json"
    source = train.Config(**json.loads(baseline.read_text()))
    cfg = replace(source, exp_id="B06", backbone="resnext50_32x4d")
    print(f"B06 config: {asdict(cfg)}", flush=True)
    row = run_lab._load_result(cfg)
    if row["exp_id"] != "B06" or row["backbone"] != cfg.backbone:
        raise ValueError("B06 summary does not match its config")
    _, net = run_lab._ckpt_predict(cfg, "val")
    latency = benchmark.latency_report(net, 1, cfg.img_size, dtype="fp32",
                                       device="cuda", iters=50)
    evidence = output / "evidence"
    (evidence / "B06_latency.json").write_text(json.dumps(latency, indent=2))
    book = output / "results.xlsx"
    sheets = pd.read_excel(book, sheet_name=None)
    if "Backbones" not in sheets:
        raise ValueError("Missing Backbones sheet")
    rows = sheets["Backbones"]
    rows = rows[rows.exp_id != "B06"]
    rows = pd.concat([rows, pd.DataFrame([row])], ignore_index=True)
    sheets["Backbones"] = rows
    if "Summary" in sheets:
        sheets["Summary"] = pd.concat([rows, sheets["Training"]], ignore_index=True).sort_values(
            "macro_f1_val", ascending=False).head(10)
    with pd.ExcelWriter(book, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            writer.sheets[name].freeze_panes = "A2"
            writer.sheets[name].auto_filter.ref = writer.sheets[name].dimensions
    selection_path = evidence / "backbone_selection.json"
    selection = json.loads(selection_path.read_text())
    selection["candidates"] = [r for r in selection["candidates"] if r["exp_id"] != "B06"] + [row]
    selection["supplemental_comparison"] = "B06 was added after F00/F01 were completed; final selection was frozen before test."
    selection_path.write_text(json.dumps(selection, indent=2))
    report = output / "report.md"
    content = report.read_text()
    section = ("\n\n## Supplemental backbone comparison\n\n"
               "B06 is ResNeXt-50 (`resnext50_32x4d`), trained with B01's fold, seed, "
               "epoch budget, optimizer and augmentation. It was added after the original "
               "final predictions were produced. The final backbone and test predictions "
               "were frozen before this supplemental experiment; B06 validation results "
               "were not used to choose or rerun test.\n\n"
               f"B06 validation macro-F1: {row['macro_f1_val']:.4f}; "
               f"top-1: {row['top1_val']:.4f}; parameters: {row['params_m']:.2f}M; "
               f"GMAC: {row['gmac']}; train time: {row['seconds_per_epoch']:.2f} s/epoch; "
               f"batch-1 latency p50/p95/p99: {latency['p50']:.2f}/"
               f"{latency['p95']:.2f}/{latency['p99']:.2f} ms on {latency['gpu']}. "
               "See `runs/B06/`, `curves/B06_seed0.png`, "
               "`predictions/B06_seed0_val.csv`, and `evidence/B06_latency.json`.\n")
    marker = "\n\n## Supplemental backbone comparison"
    report.write_text(content.split(marker)[0] + section)
    pred = run_lab._eval().read_pred(str(output / "predictions" / "F01_seed0_val.csv"))
    matrix = run_lab._eval().confusion_matrix(pred.y_true, pred.y_pred)
    pd.DataFrame(matrix, index=dataset.CLASS_NAMES, columns=dataset.CLASS_NAMES).to_csv(
        evidence / "F01_seed0_val_confusion.csv")
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(9), yticks=range(9), xticklabels=dataset.CLASS_NAMES,
           yticklabels=dataset.CLASS_NAMES, xlabel="Predicted", ylabel="True",
           title="F01 seed 0 validation confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(evidence / "F01_seed0_val_confusion.png", dpi=160)
    plt.close(fig)
    wrong = np.flatnonzero(pred.y_true != pred.y_pred)
    focus = [int(i) for i in wrong if pred.y_true[i] in (0, 7)][:9]
    if len(focus) < 9:
        focus += [int(i) for i in wrong if int(i) not in focus][:9-len(focus)]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for ax, i in zip(axes.flat, focus):
        with Image.open(Path(source.images_dir) / pred.filenames[i]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"True: {dataset.CLASS_NAMES[pred.y_true[i]]}\n"
                     f"Pred: {dataset.CLASS_NAMES[pred.y_pred[i]]}", fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[len(focus):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(evidence / "F01_seed0_val_errors.png", dpi=140)
    plt.close(fig)
    with report.open("a") as destination:
        destination.write("\n## Validation error analysis\n\n"
                          "`evidence/F01_seed0_val_confusion.png` shows the validation "
                          "confusion matrix. `evidence/F01_seed0_val_errors.png` shows "
                          "misclassified validation images with their true and predicted "
                          "classes. These images are for diagnosis; no test labels were "
                          "used to select or change the model.\n")
    (evidence / "B06_supplement_completed.json").write_text(json.dumps(
        {"exp_id": "B06", "summary": row, "latency": latency, "final_test_unchanged": True}, indent=2))
    print("B06 complete. Original final test predictions unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    supplement(parser.parse_args().output)
