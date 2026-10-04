# DeepWeeds Lab Day 2

## Measured RTX 3090 results

The completed fold-0 server run and B06 supplement are included here: [`results.xlsx`](results.xlsx), [`report.md`](report.md), [`curves/`](curves/), [`predictions/`](predictions/), and [`evidence/`](evidence/). The selected B05 backbone was ViT Tiny. The selected recipe added RandAugment; final inference used two-view probability averaging (I01). Across three seeds, F01 reached test macro-F1 **0.9327 ± 0.0029** and top-1 **0.9484 ± 0.0031**. The F00 macro-F1 baseline was **0.9187 ± 0.0061**. The provided `eval.py` assigned **14/20** in provisional quality criteria. See the report for per-class results, latency, limitations, and the supplemental ResNeXt comparison.

All figures above come from the attached server artifacts. The self-contained Kaggle notebook below was prepared for the earlier Kaggle workflow; the server commands and committed results are the reproduction path for this completed run. The submission does not include model checkpoints or the original `runs/` history files because they were not in the downloaded artifact.

## Run on an RTX 3090 server

Use a server image with a working NVIDIA driver, Python 3 and CUDA-enabled PyTorch. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r submissions/2A202602982_HaManhTuan/requirements-server.txt
python -c 'import torch; print(torch.cuda.get_device_name(0))'
set -o pipefail
bash submissions/2A202602982_HaManhTuan/run_server.sh 2>&1 | tee server.log
```

The script downloads and checks 17,509 images and fold-0 labels, runs tests and a smoke experiment, then writes the full lab to `lab_output/full/`. It supports one GPU; the two-GPU option has no effect when only one device is visible. Keep the terminal attached or run it in `tmux`. If interrupted, rerun the same command with the same data, output path and epoch settings; finished experiments are reused. Download `lab_output/full/` after `evidence/completed.json` appears. To use existing data or another output disk, set `DEEPWEEDS_DATA` and `DEEPWEEDS_OUTPUT`. To change the training budgets, set `DEEPWEEDS_EPOCHS` and `DEEPWEEDS_FINAL_EPOCHS` before the first run. Defaults are 10 and 12; changing them after a partial run requires a fresh output directory.

The CUDA wheel command is for Linux and a compatible NVIDIA driver. If the rental image already has working CUDA PyTorch, keep that installation and install only `requirements-server.txt`.

On a single RTX 3090, `run_server.sh` enables a speed setting by default: four persistent image-loading workers, cuDNN kernel benchmarking, and fast float32 matrix operations. Mixed precision is already enabled. Set `DEEPWEEDS_WORKERS=8` to try more workers if the GPU is idle, or `DEEPWEEDS_FAST=0` for the previous deterministic behavior. Different speed settings are recorded in each run's config; use a new output directory when changing settings after a run starts. Training time still depends on storage speed and model choice.

### Complete the backbone comparison after the first server run

The first completed run compared B01–B05. The rubric additionally requires ResNeXt or ConvNeXt. B06 trains ResNeXt-50 with the saved B01 setup on the same fold and seed. It updates the Backbones and Summary sheets, appends the measured comparison to the report, saves a latency result, and generates validation confusion and error images. It leaves F00/F01 test predictions unchanged. Run after `lab_output/full/evidence/completed.json` exists:

```bash
source .venv/bin/activate
python submissions/2A202602982_HaManhTuan/code/supplement_backbone.py --output lab_output/full 2>&1 | tee supplement.log
python submissions/2A202602982_HaManhTuan/code/package_submission.py \
  --output lab_output/full --submission submissions/2A202602982_HaManhTuan
```

`evidence/B06_supplement_completed.json` confirms the added comparison. The final model selection was frozen before B06 and must be described as such. Do not claim a new F01 score from B06; the F01 test predictions are those already measured. `package_submission.py` copies the measured workbook, report, plots, predictions and evidence into the submission folder without copying dataset images or model checkpoints.

### Additional audit evidence on the original server

`code/audit_evidence.py` records a retrospective one-batch overfit check, an augmentation image grid, and batch-1 GPU latency for B01–B06. The check is explicitly marked retrospective; it must not be described as evidence that it ran before the original experiments. The script reads existing checkpoints and does not run test inference or alter final predictions. `package_submission.py` also copies each run's small config, history, summary, and split check into `run_metadata/`. Keep server logs with the evidence for stronger provenance.

Run [`deepweeds_kaggle_t4x2.ipynb`](deepweeds_kaggle_t4x2.ipynb) as a new Kaggle Notebook. Select **GPU T4 x2**, enable **Internet**, then choose **Save & Run All**. The notebook is self-contained: it writes the completed Python modules and the repository's original `eval.py`, downloads the original fold-0 CSV files and images, verifies the image MD5, runs unit and smoke checks, then starts the full experiment sequence.

The full sequence screens five backbones, evaluates initialization/augmentation/loss choices against one baseline, trains the selected recipe and baseline with three seeds each, compares validation inference methods, and evaluates test once the method is selected. It writes `results.xlsx`, `report.md`, `curves/`, `predictions/`, `runs/`, and `evidence/` under `/kaggle/working/lab_output`. The final cell creates `/kaggle/working/deepweeds_evidence.zip`. Kaggle keeps these files under the notebook version's **Output** tab after Save & Run All completes. Download the zip from there to keep a local copy.

The backbone and recipe screening runs use 10 epochs each; final and baseline seed runs use 12. These values follow the 10-15 epoch baseline range in `GUIDE.md` section 1.4. They are choices for this run, not mandatory values specified by the assignment. Every backbone in the screening comparison receives the same epoch budget. The 12-epoch baseline is named `F00` so its checkpoints remain separate from the 10-epoch `T00` screening run. The best validation macro-F1 checkpoint is kept, but training does not stop early.

The notebook's full GPU run has **not been executed in this workspace**. Only local CPU smoke validation is available here. Do not treat the generated report or workbook as measured results until the Kaggle job finishes and `evidence/completed.json` exists.

If a Kaggle run ends before completion, add its `deepweeds_evidence.zip` as an Input to a new notebook version. Kaggle may expose it as a zip or as an extracted folder; the notebook supports both. It restores completed checkpoints, logs, curves and validation predictions, then skips runs whose saved configs match. The current base lineup is ResNet-50/18/34, RegNetX-002, and ViT Tiny; the server supplement adds ResNeXt-50 as B06. The Kaggle notebook is an earlier alternative workflow and has not been used to generate the committed measured results.

Source files live in [`code/`](code/). Rebuild the self-contained notebook after source edits with:

```bash
python code/make_notebook.py
```

The original split and images can be reused locally by running:

```bash
python code/run_lab.py smoke --data /workspace/data --output /workspace/lab_smoke_test
```

The complete run requires CUDA. It uses the original `eval.py` for scoring and grading and never trains on validation or test. Logs, configs, checkpoints, and raw logits remain in `runs/` for audit. The final report is generated from measured outputs; interpret small differences cautiously because screening uses one seed.
