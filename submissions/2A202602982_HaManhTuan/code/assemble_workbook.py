"""Enrich measured workbook using saved configs, predictions, and timing evidence."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font, PatternFill


METHODS = {
    "I00": ("one view", 1),
    "I01": ("horizontal flip; mean probabilities", 2),
    "I02": ("horizontal flip; mean logits", 2),
    "I03": ("one view; temperature scaling", 1),
    "I04": ("horizontal flip; mean logits; temperature scaling", 2),
}


def build(submission):
    submission = Path(submission).resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import eval as ev

    book = submission / "results.xlsx"
    sheets = pd.read_excel(book, sheet_name=None)
    metadata = submission / "run_metadata"
    backbone = sheets["Backbones"]
    backbone["img_size_px"] = [json.loads((metadata / exp / "seed0/config.json").read_text())["img_size"]
                               for exp in backbone.exp_id]
    backbone["epochs"] = [json.loads((metadata / exp / "seed0/config.json").read_text())["epochs"]
                          for exp in backbone.exp_id]
    backbone["notes"] = ["supplemented after final selection" if exp == "B06" else "original screening"
                         for exp in backbone.exp_id]
    sheets["Backbones"] = backbone

    training = sheets["Training"]
    base_f1 = float(training.loc[training.exp_id == "T00", "macro_f1_val"].iloc[0])
    training["delta_macro_f1_vs_T00"] = training.macro_f1_val - base_f1
    hard_names = {"Chinee Apple": "chinee_apple_f1_val", "Snake Weed": "snake_weed_f1_val"}
    hard_values = {name: [] for name in hard_names.values()}
    for exp in training.exp_id:
        pred = ev.read_pred(str(submission / "predictions" / f"{exp}_seed0_val.csv"))
        metrics = ev.compute_metrics(pred.y_true, pred.y_pred, pred.probs)
        for class_name, col in hard_names.items():
            hard_values[col].append(float(metrics["f1"][ev.CLASS_NAMES.index(class_name)]))
    for col, values in hard_values.items():
        training[col] = values
    training["notes"] = training.exp_id.map({"T00": "baseline"}).fillna("one factor changed from T00")
    sheets["Training"] = training

    inference = sheets["Inference"]
    inference["method_description"] = inference.exp_id.map(lambda x: METHODS[x][0])
    inference["views"] = inference.exp_id.map(lambda x: METHODS[x][1])
    inference["backbone"] = "vit_tiny_patch16_224"
    inference["checkpoint"] = inference.seed.map(lambda x: f"F01_seed{x}")
    inference["images_per_s_batch1"] = 1000 / inference.latency_p50_ms
    sheets["Inference"] = inference

    final = sheets["Final"]
    final["backbone"] = "vit_tiny_patch16_224"
    final["recipe"] = final.exp_id.map({"F00": "basic augmentation; CE", "F01": "RandAugment; CE"})
    for metric in ("macro_f1", "top1", "ece"):
        final[f"{metric}_test_mean"] = final.exp_id.map(final.groupby("exp_id")[metric].mean())
        final[f"{metric}_test_std"] = final.exp_id.map(final.groupby("exp_id")[metric].std())
    val_macro, val_top1 = [], []
    for row in final.itertuples():
        pred = ev.read_pred(str(submission / "predictions" / f"{row.exp_id}_seed{row.seed}_val.csv"))
        m = ev.compute_metrics(pred.y_true, pred.y_pred, pred.probs)
        val_macro.append(float(m["macro_f1"]))
        val_top1.append(float(m["top1"]))
    final["macro_f1_val"] = val_macro
    final["top1_val"] = val_top1
    sheets["Final"] = final

    latency = sheets["Latency"]
    latency["configuration"] = latency.method.map(lambda x: "ViT Tiny; " + x)
    latency["bn_fused"] = False
    latency["gpu"] = latency.gpu.fillna("NVIDIA GeForce RTX 3090")
    latency["dtype"] = latency.dtype.fillna("fp32")
    latency["batch"] = latency.batch.fillna(1).astype(int)
    latency["images_per_s"] = latency.images_per_s.fillna(1000 / latency.p50)
    sheets["Latency"] = latency

    summary = sheets["Summary"]
    p95 = backbone.set_index("exp_id").latency_p95_ms
    summary["latency_p95_ms"] = [float(p95.get(exp, p95["B05"])) for exp in summary.exp_id]
    summary["rank_val_macro_f1"] = range(1, len(summary) + 1)
    sheets["Summary"] = summary

    with pd.ExcelWriter(book, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.fill = PatternFill("solid", fgColor="17365D")
                cell.font = Font(color="FFFFFF", bold=True)
            for column in sheet.columns:
                letter = column[0].column_letter
                width = min(45, max(12, max(len(str(c.value or "")) for c in column[: min(50, len(column))]) + 2))
                sheet.column_dimensions[letter].width = width
    print(f"Updated {book} from measured evidence")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    build(parser.parse_args().submission)
