"""Check submission completeness and recalculate final metrics from predictions."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def verify(submission, data=None):
    submission = Path(submission).resolve()
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    import eval as ev

    required = ("README.md", "report.md", "results.xlsx", "code/train.py",
                "code/deepweeds_kaggle_t4x2.ipynb", "evidence/completed.json",
                "evidence/B06_supplement_completed.json", "evidence/grade/grade_I.json",
                "evidence/audit_evidence_completed.json")
    for name in required:
        if not (submission / name).is_file():
            raise FileNotFoundError(submission / name)

    sheets = pd.read_excel(submission / "results.xlsx", sheet_name=None)
    expected_sheets = {"Backbones", "Training", "Inference", "Final", "PerClass",
                       "Latency", "BackboneLatency", "Summary"}
    if not expected_sheets.issubset(sheets):
        raise AssertionError(f"Missing sheets: {expected_sheets - set(sheets)}")
    expected = [f"B{i:02d}" for i in range(1, 7)] + [f"T{i:02d}" for i in range(7)]
    expected += [f"{group}/seed{seed}" for group in ("F00", "F01") for seed in range(3)]
    if set(sheets["Backbones"].exp_id) != set(expected[:6]):
        raise AssertionError("Backbone experiment IDs differ from expected B01-B06")
    if set(sheets["Training"].exp_id) != set(expected[6:13]):
        raise AssertionError("Training experiment IDs differ from expected T00-T06")
    for entry in expected:
        exp_id, seed_dir = entry.split("/") if "/" in entry else (entry, "seed0")
        cfg_dir = submission / "run_metadata" / exp_id / seed_dir
        cfg = json.loads((cfg_dir / "config.json").read_text())
        summary = json.loads((cfg_dir / "summary.json").read_text())
        history = pd.read_csv(cfg_dir / "history.csv")
        if cfg["exp_id"] != exp_id or len(history) != cfg["epochs"]:
            raise AssertionError(f"Config/history mismatch: {entry}")
        if summary["exp_id"] != exp_id:
            raise AssertionError(f"Summary mismatch: {entry}")
        if not (submission / "curves" / f"{exp_id}_{seed_dir}.png").is_file():
            raise FileNotFoundError(f"Missing curve: {entry}")

    final = sheets["Final"]
    if len(final) != 6 or set(final.exp_id) != {"F00", "F01"}:
        raise AssertionError("Final sheet must contain three seeds for each group")
    for row in final.itertuples():
        path = submission / "predictions" / f"{row.exp_id}_seed{row.seed}_test.csv"
        pred = ev.read_pred(str(path))
        if data is not None:
            ev.check_against_csv(pred, str(Path(data) / "labels/test_subset0.csv"))
        measured = ev.compute_metrics(pred.y_true, pred.y_pred, pred.probs)
        for key in ("macro_f1", "top1", "balanced_acc", "ece", "nll"):
            if abs(measured[key] - getattr(row, key)) > 1e-6:
                raise AssertionError(f"Workbook differs from prediction: {path}, {key}")
    if set(sheets["Inference"].exp_id) != {f"I{i:02d}" for i in range(5)}:
        raise AssertionError("Missing inference comparisons")
    print(f"Verified {len(expected)} training runs, 5 inference methods, "
          "6 final test predictions, and all required artifacts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    verify(args.submission, args.data)
