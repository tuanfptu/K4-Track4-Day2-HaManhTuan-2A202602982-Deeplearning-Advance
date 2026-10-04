# DeepWeeds Lab Day 2

## Data and method

Original DeepWeeds fold 0, 9 classes, {'train': 10501, 'val': 3501, 'test': 3507}. Train updates weights; validation chooses checkpoints, recipe and inference; test is scored after selection. ImageNet pretrained weights, AdamW, cosine schedule, 224px input and mixed precision are used unless an experiment config says otherwise. GPU: NVIDIA GeForce RTX 3090; PyTorch 2.6.0+cu124.

## Backbone comparison (validation)

| exp_id   | backbone             |   macro_f1_val |   top1_val |   params_m |     gmac |   seconds_per_epoch |
|:---------|:---------------------|---------------:|-----------:|-----------:|---------:|--------------------:|
| B01      | resnet50             |       0.79216  |   0.8489   |   23.5265  | 4.10948  |            13.2484  |
| B02      | resnet18             |       0.76874  |   0.830049 |   11.1811  | 1.81856  |             6.57532 |
| B03      | resnet34             |       0.765125 |   0.830905 |   21.2893  | 3.67076  |             8.70184 |
| B04      | regnetx_002          |       0.84455  |   0.884604 |    2.31911 | 0.203019 |             6.47946 |
| B05      | vit_tiny_patch16_224 |       0.919873 |   0.941445 |    5.52615 | 1.07939  |             6.68857 |

Selected: vit_tiny_patch16_224, highest validation macro-F1.

## Training recipe (validation)

| exp_id   |   macro_f1_val |   top1_val |   best_epoch |
|:---------|---------------:|-----------:|-------------:|
| T00      |       0.919873 |   0.941445 |           10 |
| T01      |       0.595767 |   0.705513 |            6 |
| T02      |       0.60492  |   0.708083 |           10 |
| T03      |       0.921356 |   0.939732 |           10 |
| T04      |       0.926875 |   0.942588 |            8 |
| T05      |       0.910274 |   0.932876 |            9 |
| T06      |       0.900918 |   0.925164 |            8 |

Selected factors by validation: `{"aug": "randaug"}`. One factor changes at a time relative to T00; the combination is retrained.

## Inference selection (validation)

| exp_id   |   macro_f1 |        ece |
|:---------|-----------:|-----------:|
| I00      |   0.929012 | 0.0167979  |
| I01      |   0.931652 | 0.0103981  |
| I02      |   0.931554 | 0.0136155  |
| I03      |   0.929012 | 0.00703282 |
| I04      |   0.931554 | 0.007381   |

Selected method: I01. Temperature is fitted on validation only.

## Final test (three seeds)

| exp_id   |   seed | method   |   macro_f1 |     top1 |   balanced_acc |       ece |      nll |
|:---------|-------:|:---------|-----------:|---------:|---------------:|----------:|---------:|
| F00      |      0 | I00      |   0.912938 | 0.934417 |       0.90577  | 0.0239298 | 0.19885  |
| F00      |      1 | I00      |   0.918185 | 0.939835 |       0.922782 | 0.0194035 | 0.189942 |
| F00      |      2 | I00      |   0.925009 | 0.943541 |       0.922739 | 0.0139053 | 0.182001 |
| F01      |      0 | I01      |   0.932979 | 0.949244 |       0.924105 | 0.0105021 | 0.152944 |
| F01      |      1 | I01      |   0.935425 | 0.950955 |       0.926053 | 0.0117514 | 0.152236 |
| F01      |      2 | I01      |   0.929735 | 0.944967 |       0.921227 | 0.0141047 | 0.160393 |

Final macro-F1: 0.9327 +/- 0.0029; baseline: 0.9187; delta: +0.0140. The standard deviation is sample std (ddof=1).

## Per-class test F1, final configuration

| class          |   support |       f1 |
|:---------------|----------:|---------:|
| Chinee Apple   |       226 | 0.863671 |
| Lantana        |       213 | 0.943057 |
| Negatives      |      1822 | 0.966215 |
| Parkinsonia    |       207 | 0.970045 |
| Parthenium     |       205 | 0.942726 |
| Prickly Acacia |       213 | 0.912096 |
| Rubber Vine    |       202 | 0.952703 |
| Siam Weed      |       215 | 0.948349 |
| Snake Weed     |       204 | 0.895558 |

## Latency

| method      |      p50 |      p95 |      p99 |     mean |   n | gpu                     | dtype   |   batch |   img_size |   images_per_s | torch       |   preprocessing_included |   k_views |
|:------------|---------:|---------:|---------:|---------:|----:|:------------------------|:--------|--------:|-----------:|---------------:|:------------|-------------------------:|----------:|
| I00         |  4.49524 |  4.52493 |  4.55031 |  4.44224 |  50 | NVIDIA GeForce RTX 3090 | fp32    |       1 |        224 |        222.458 | 2.6.0+cu124 |                        0 |       nan |
| I00         |  6.04628 |  6.13369 |  6.20205 |  6.04894 |  50 | NVIDIA GeForce RTX 3090 | amp     |       1 |        224 |        165.391 | 2.6.0+cu124 |                        0 |       nan |
| I00         | 10.4221  | 10.457   | 10.5096  | 10.4273  |  50 | NVIDIA GeForce RTX 3090 | fp32    |      32 |        224 |       3070.39  | 2.6.0+cu124 |                        0 |       nan |
| I00         |  6.55759 |  6.91082 |  7.09762 |  6.59888 |  50 | NVIDIA GeForce RTX 3090 | amp     |      32 |        224 |       4879.84  | 2.6.0+cu124 |                        0 |       nan |
| I01/I02/I04 |  8.54725 |  9.16956 |  9.33566 |  8.71155 |  50 | nan                     | nan     |     nan |        nan |        nan     | nan         |                      nan |         2 |

Latency excludes preprocessing, uses 10 warmup and 50 synchronized iterations. See `evidence/`, `curves/`, `predictions/`, and `results.xlsx` for available evidence. Single-seed screening is subject to noise; results may vary across GPU/software versions.


## Supplemental backbone comparison

B06 is ResNeXt-50 (`resnext50_32x4d`), trained with B01's fold, seed, epoch budget, optimizer and augmentation. It was added after the original final predictions were produced. The final backbone and test predictions were frozen before this supplemental experiment; B06 validation results were not used to choose or rerun test.

B06 validation macro-F1: 0.7866; top-1: 0.8355; parameters: 23.00M; GMAC: 4.25735168; train time: 16.03 s/epoch; batch-1 latency p50/p95/p99: 5.23/5.44/5.49 ms in its first measurement on NVIDIA GeForce RTX 3090. See `run_metadata/B06/seed0/`, `curves/B06_seed0.png`, `predictions/B06_seed0_val.csv`, and `evidence/B06_latency.json`.

## Validation error analysis

`evidence/F01_seed0_val_confusion.png` shows the validation confusion matrix. `evidence/F01_seed0_val_errors.png` shows misclassified validation images with their true and predicted classes. These images are for diagnosis; no test labels were used to select or change the model.

## Interpretation and limitations

The original fold-0 split has 9,106 Negatives (52.0% of all images); each weed class has 1,009–1,125. This agrees with the class counts in the assignment's Table 1. Macro-F1 therefore matters alongside top-1 accuracy: a model could predict the dominant class often and still miss important weed species. Train, validation, and test contain 10,501, 3,501, and 3,507 distinct images, respectively (`evidence/split_check.json`).

B05 (ViT Tiny) achieved validation macro-F1 0.9199, versus 0.8446 for the next best screened backbone B04 (RegNetX-002). B06 (ResNeXt-50) reached 0.7866 after the final recipe had already been chosen, so the added ResNeXt comparison does not challenge that choice. B05 also trained in 6.69 seconds per epoch in this run, while B06 took 16.03 seconds. These are single-seed screening results; pretrained weights differ between model families, so architecture alone does not explain every difference.

Within the ViT training comparison, RandAugment increased validation macro-F1 from 0.9199 (T00) to 0.9269 (T04), a gain of 0.0070. Frozen-backbone and scratch initialization fell to 0.5958 and 0.6049. Label smoothing and focal loss were below the cross-entropy baseline in this run. Only RandAugment was selected; F01 retrained that choice with three seeds. The screening comparisons themselves use one seed, so small differences are uncertain.

F01 with two-view probability averaging (I01) achieved test macro-F1 **0.9327 ± 0.0029** and top-1 **0.9484 ± 0.0031** across three seeds. F00 achieved macro-F1 **0.9187 ± 0.0061**; the difference is **+0.0140**, greater than the larger observed seed standard deviation. On validation, I01 averaged 0.9317 macro-F1 versus 0.9290 for one-view I00, but its batch-1 p95 latency was **9.17 ms**, compared with **4.52 ms** for I00. Both are below the rubric's 100 ms budget. The automatic grade's I5 note quotes 4.5 ms for the one-view candidate; the selected I01 method is the 9.17 ms measurement in `evidence/latency.csv`. Latency excludes image loading and preprocessing.

Temperature scaling reduced validation ECE for some inference candidates, but I01 was selected by mean validation macro-F1 and does not use temperature scaling. Thus the automatic grade correctly gives I4a zero for the submitted final predictions. No test results were used to change the selected inference method.

The validation confusion matrix for F01 seed 0 shows 22 Chinee Apple images predicted as Negatives and 14 as Snake Weed; 16 Snake Weed images were predicted as Negatives. The final test recall for Chinee Apple is 79.6%, below the paper's 88.5% reference. More varied images of those species or a separately validated class-focused training experiment would be the next areas to study; this run does not establish that either would improve test performance.

The study uses one predefined fold and only one seed for backbone and training screening. It uses fixed 10/12-epoch budgets and keeps the best validation checkpoint rather than stopping training early. A split from the same collection may overestimate performance on a new location, season, or camera. The submission includes plots, predictions, workbook, server logs, and `run_metadata/` with each run's config and history; large model checkpoints are omitted. Do not treat the measured GPU latency as end-to-end camera latency.

## Additional pipeline and backbone evidence

The retrospective GPU check in `evidence/pipeline_check.json` used a scratch MobileNetV3 Small and a fixed batch of nine images, one per class. Its initial cross-entropy was 2.0495 (the uniform nine-class reference is ln 9 = 2.1972); loss fell to 0.00000163 after 20 optimizer steps. This supports that the model can learn a tiny batch, but **this check was run after the full experiment**, not as a preflight. `evidence/augmentation_check.png` shows each selected original image beside its transformed input after reversing ImageNet normalization. The full per-step trace is in `evidence/one_batch_overfit.csv`.

The six backbone checkpoints were measured again on the RTX 3090 with batch size 1, fp32, 10 warmup iterations, 50 synchronized timing iterations, and no preprocessing. The results are in the `BackboneLatency` sheet and `evidence/backbone_quality_latency.png`. B05 had p95 **4.23 ms** and validation macro-F1 **0.9199**. B02 was fastest at **2.35 ms** but had macro-F1 **0.7687**; B04 reached **0.8446** at **4.87 ms**. B06's repeat p95 was **5.62 ms**, compared with **5.44 ms** in its first measurement, illustrating normal timing variation. B05 offered the strongest validation quality among the measured candidates while staying far below the 100 ms budget. The backbone screening uses different pretrained weight recipes and one seed, so these measurements do not isolate architecture alone.
