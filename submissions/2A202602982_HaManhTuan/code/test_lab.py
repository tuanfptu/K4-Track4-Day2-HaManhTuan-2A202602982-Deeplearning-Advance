"""Focused checks for the completed lab code."""
import unittest
import os
import json
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import dataset
import inference
import losses
import model
import run_lab
import package_submission
import train


class LabTests(unittest.TestCase):
    def test_package_requires_completed_supplement_and_excludes_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, submission = Path(tmp) / "output", Path(tmp) / "submission"
            submission.mkdir()
            for name in ("evidence/completed.json", "results.xlsx", "report.md",
                         "curves/B06_seed0.png", "predictions/B06_seed0_val.csv",
                         "runs/B06/seed0/best.pt"):
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(name)
            with self.assertRaises(FileNotFoundError):
                package_submission.package(output, submission)
            (output / "evidence/B06_supplement_completed.json").write_text("{}")
            package_submission.package(output, submission)
            self.assertTrue((submission / "predictions/B06_seed0_val.csv").is_file())
            self.assertFalse((submission / "runs/B06/seed0/best.pt").exists())
            self.assertTrue((submission / "run_metadata/B06/seed0").is_dir())

    def test_original_split_and_image_files(self):
        data = Path(os.environ.get("DEEPWEEDS_DATA", "/workspace/data"))
        if not data.exists():
            self.skipTest("Local dataset not available")
        frames = dataset.load_split(data / "labels")
        result = dataset.check_split(*frames, data / "images")
        self.assertEqual(result["n"], {"train": 10501, "val": 3501, "test": 3507})
        self.assertFalse(any(result["overlap"].values()))

    def test_focal_zero_equals_ce(self):
        z = torch.randn(12, 9)
        y = torch.arange(12) % 9
        self.assertLess(abs(losses.FocalLoss(gamma=0)(z, y).item() - F.cross_entropy(z, y).item()), 1e-6)

    def test_mix_batch_and_probability_aggregation(self):
        x, y = torch.randn(8, 3, 16, 16), torch.arange(8)
        for mode in ("mixup", "cutmix"):
            mixed, (_, _, lam) = losses.mix_batch(x, y, mode=mode)
            self.assertEqual(mixed.shape, x.shape)
            self.assertTrue(0 <= lam <= 1)
        z = np.random.default_rng(0).normal(size=(8, 9))
        for space in ("prob", "logit"):
            p = inference.aggregate_views([z, z], space)
            np.testing.assert_allclose(p.sum(1), 1, atol=1e-7)
        self.assertTrue(inference.fit_temperature(z, y.numpy()) > 0)

    def test_model_groups_cover_trainable_parameters(self):
        net = model.build_model("mobilenetv3_small_050", pretrained=False, init="frozen")
        groups = model.param_groups(net, 1e-4, 1e-3, 0.05)
        self.assertEqual(sum(p.numel() for g in groups for p in g["params"]),
                         sum(p.numel() for p in net.parameters() if p.requires_grad))
        self.assertTrue(any(not p.requires_grad for p in net.parameters()))

    def test_config_override(self):
        values = train.parse_overrides(["seed=2", "amp=false", "ema_decay=0.99", "mix=none"])
        self.assertEqual(values, {"seed": 2, "amp": False, "ema_decay": 0.99, "mix": None})

    def test_conv_bn_fusion_preserves_eval_output(self):
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1),
                                  torch.nn.BatchNorm2d(4)).eval()
        x = torch.randn(2, 3, 8, 8)
        fused = inference.fuse_conv_bn(net)
        self.assertLess((net(x) - fused(x)).abs().max().item(), 1e-5)

    def test_resume_rejects_different_epoch_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = train.Config(out_dir=tmp, epochs=10)
            folder = train.run_dir(cfg)
            folder.mkdir(parents=True)
            (folder / "config.json").write_text(json.dumps(asdict(cfg)))
            (folder / "summary.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "different config"):
                run_lab._load_result(train.Config(out_dir=tmp, epochs=12))

    def test_scheduler_does_not_advance_on_amp_overflow(self):
        net = torch.nn.Linear(3, 9)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        class SkippedStep:
            def __init__(self):
                self.scale_value = 2.0
            def scale(self, loss):
                return loss
            def get_scale(self):
                return self.scale_value
            def step(self, optimizer):
                pass
            def update(self):
                self.scale_value = 1.0
        loader = [(torch.randn(2, 3), torch.tensor([0, 1]), ["a", "b"])]
        train.train_one_epoch(net, loader, torch.nn.CrossEntropyLoss(), optimizer,
                              scheduler, SkippedStep(), train.Config(), torch.device("cpu"))
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.1)

    def test_restore_only_completed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "evidence.zip"
            with zipfile.ZipFile(archive, "w") as z:
                for name in ("runs/B01/seed0/config.json", "runs/B01/seed0/summary.json",
                             "runs/B01/seed0/best.pt", "runs/B02/seed0/best.pt",
                             "curves/B01_seed0.png", "predictions/B01_seed0_val.csv"):
                    z.writestr(name, name)
            destination = Path(tmp) / "out"
            self.assertEqual(run_lab.restore_completed_runs(archive, destination), 5)
            self.assertTrue((destination / "runs/B01/seed0/best.pt").exists())
            self.assertTrue((destination / "curves/B01_seed0.png").exists())
            self.assertFalse((destination / "runs/B02/seed0/best.pt").exists())

    def test_restored_run_is_reused_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "lab_output"
            cfg = train.Config(exp_id="B01", out_dir=str(output / "runs"),
                               pred_dir=str(output / "predictions"), epochs=10)
            archive = Path(tmp) / "evidence.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("runs/B01/seed0/config.json", json.dumps(asdict(cfg)))
                z.writestr("runs/B01/seed0/summary.json", '{"macro_f1_val": 0.785}')
                z.writestr("runs/B01/seed0/best.pt", "checkpoint")
            run_lab.restore_completed_runs(archive, output)
            self.assertEqual(run_lab._load_result(cfg)["macro_f1_val"], 0.785)

    def test_restore_extracted_evidence_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "out"
            folder = source / "runs/B01/seed0"
            folder.mkdir(parents=True)
            for name in ("config.json", "summary.json", "best.pt"):
                (folder / name).write_text(name)
            (source / "curves").mkdir()
            (source / "curves/B01_seed0.png").write_text("curve")
            self.assertEqual(run_lab.restore_completed_tree(source, output), 4)
            self.assertTrue((output / "runs/B01/seed0/best.pt").exists())
            self.assertTrue((output / "curves/B01_seed0.png").exists())


if __name__ == "__main__":
    unittest.main()
