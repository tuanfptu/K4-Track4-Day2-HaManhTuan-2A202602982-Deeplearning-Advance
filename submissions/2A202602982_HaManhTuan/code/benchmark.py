"""Synchronized inference latency measurements."""
import time
import numpy as np
import torch

def bench(fn, warmup=10, iters=100, sync=None):
    if warmup < 10 or iters < 50:
        raise ValueError("Requires >=10 warmup and >=50 measured iterations")
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        if sync:
            sync()
        start = time.perf_counter()
        fn()
        if sync:
            sync()
        samples.append((time.perf_counter() - start) * 1000)
    p50, p95, p99 = np.percentile(samples, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99),
            "mean": float(np.mean(samples)), "n": iters}

def latency_report(model, batch_size, img_size, dtype="fp32", device="cuda", warmup=10, iters=100):
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if dtype not in ("fp32", "amp", "fp16"):
        raise ValueError(dtype)
    net = model.to(dev).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        net = net.half()
        x = x.half()
    def forward():
        with torch.inference_mode(), torch.autocast(device_type=dev.type, enabled=dtype == "amp"):
            net(x)
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else None
    result = bench(forward, warmup, iters, sync)
    return {**result, "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU",
            "dtype": dtype, "batch": batch_size, "img_size": img_size,
            "images_per_s": batch_size / (result["p50"] / 1000),
            "torch": torch.__version__, "preprocessing_included": False}

def tta_latency(model, k_views, **kw):
    if k_views < 1:
        raise ValueError("k_views must be >=1")
    dev = torch.device(kw.get("device", "cuda"))
    net = model.to(dev).eval()
    batch, size = kw.get("batch_size", 1), kw.get("img_size", 224)
    x = torch.randn(batch, 3, size, size, device=dev)
    def forward():
        with torch.inference_mode():
            for view in range(k_views):
                net(x if view == 0 else torch.flip(x, dims=(-1,)))
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else None
    return {**bench(forward, kw.get("warmup", 10), kw.get("iters", 100), sync), "k_views": k_views}
