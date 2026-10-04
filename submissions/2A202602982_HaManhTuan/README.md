# DeepWeeds Lab Day 2

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
