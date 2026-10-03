# DeepWeeds Lab Day 2

Run [`deepweeds_kaggle_t4x2.ipynb`](deepweeds_kaggle_t4x2.ipynb) as a new Kaggle Notebook. Select **GPU T4 x2**, enable **Internet**, then choose **Save & Run All**. The notebook is self-contained: it writes the completed Python modules and the repository's original `eval.py`, downloads the original fold-0 CSV files and images, verifies the image MD5, runs unit and smoke checks, then starts the full experiment sequence.

The full sequence screens five backbones, evaluates initialization/augmentation/loss choices against one baseline, trains the selected recipe and baseline with three seeds each, compares validation inference methods, and evaluates test once the method is selected. It writes `results.xlsx`, `report.md`, `curves/`, `predictions/`, `runs/`, and `evidence/` under `/kaggle/working/lab_output`. The final cell creates `/kaggle/working/deepweeds_evidence.zip`. Kaggle keeps these files under the notebook version's **Output** tab after Save & Run All completes. Download the zip from there to keep a local copy.

The backbone and recipe screening runs use 10 epochs each; final and baseline seed runs use 12. These values follow the 10-15 epoch baseline range in `GUIDE.md` section 1.4. They are choices for this run, not mandatory values specified by the assignment. Every backbone in the screening comparison receives the same epoch budget. The 12-epoch baseline is named `F00` so its checkpoints remain separate from the 10-epoch `T00` screening run. The best validation macro-F1 checkpoint is kept, but training does not stop early.

The notebook's full GPU run has **not been executed in this workspace**. Only local CPU smoke validation is available here. Do not treat the generated report or workbook as measured results until the Kaggle job finishes and `evidence/completed.json` exists.

If a Kaggle run ends before completion, add its `deepweeds_evidence.zip` as an Input to a new notebook version. Kaggle may expose it as a zip or as an extracted folder; the notebook supports both. It restores completed checkpoints, logs, curves and validation predictions, then skips runs whose saved configs match. Kaggle reported that the first ConvNeXt-Tiny attempt exceeded available memory after two epochs. The retry uses five lighter backbones (`resnet50`, `resnet18`, `efficientnet_b0`, `mobilenetv3_large_100`, `vit_tiny_patch16_224`), runs each experiment in a separate process, and avoids DataLoader worker processes after `B01` to reduce memory use. The batch size and epoch budgets are unchanged. An incomplete run must not be reported as a finished result.

Source files live in [`code/`](code/). Rebuild the self-contained notebook after source edits with:

```bash
python code/make_notebook.py
```

The original split and images can be reused locally by running:

```bash
python code/run_lab.py smoke --data /workspace/data --output /workspace/lab_smoke_test
```

The complete run requires CUDA. It uses the original `eval.py` for scoring and grading and never trains on validation or test. Logs, configs, checkpoints, and raw logits remain in `runs/` for audit. The final report is generated from measured outputs; interpret small differences cautiously because screening uses one seed.
