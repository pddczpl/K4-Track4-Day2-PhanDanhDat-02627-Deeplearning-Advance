"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73, 75; GUIDE.md mục 4.1).

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype, batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không (đây: CHỈ đo forward pass)
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.

    Trả về {"p50", "p95", "p99", "mean", "n"}.
    """
    # Warmup: chạy trước, bỏ kết quả
    for _ in range(warmup):
        fn()
        if sync:
            sync()

    timings_ms = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    timings_ms = np.array(timings_ms)
    return {
        "p50":  float(np.percentile(timings_ms, 50)),
        "p95":  float(np.percentile(timings_ms, 95)),
        "p99":  float(np.percentile(timings_ms, 99)),
        "mean": float(timings_ms.mean()),
        "std":  float(timings_ms.std()),
        "n":    iters,
    }


# ---------------------------------------------------------------------------
def latency_report(model: nn.Module, batch_size: int = 1, img_size: int = 224,
                   dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của model với đầu vào ngẫu nhiên.

    dtype: "fp32" | "amp" (torch.autocast) | "fp16" (model.half())
    Trả về dict có thể ghi vào sheet `Latency` của results.xlsx.
    """
    device_obj = torch.device(device)
    model      = model.to(device_obj).eval()

    # Chuẩn bị đầu vào
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device_obj)

    if dtype == "fp16":
        model = model.half()
        dummy = dummy.half()
    elif dtype == "fp32":
        model = model.float()
        dummy = dummy.float()
    # amp: giữ nguyên fp32 model, dùng autocast bên trong fn

    sync_fn = torch.cuda.synchronize if device_obj.type == "cuda" else None

    @torch.no_grad()
    def _forward_fp32():
        return model(dummy)

    @torch.no_grad()
    def _forward_amp():
        with torch.amp.autocast("cuda"):
            return model(dummy)


    fn = _forward_amp if dtype == "amp" else _forward_fp32

    stats = bench(fn, warmup=warmup, iters=iters, sync=sync_fn)

    # Thông tin GPU
    gpu_name = "N/A"
    if device_obj.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)

    report = {
        "gpu":          gpu_name,
        "dtype":        dtype,
        "batch":        batch_size,
        "img_size":     img_size,
        "torch":        torch.__version__,
        "p50_ms":       stats["p50"],
        "p95_ms":       stats["p95"],
        "p99_ms":       stats["p99"],
        "mean_ms":      stats["mean"],
        "std_ms":       stats["std"],
        "images_per_s": batch_size / (stats["p50"] / 1000),
        "warmup":       warmup,
        "iters":        iters,
    }

    print(f"[latency_report] {gpu_name} | {dtype} | batch={batch_size} | {img_size}px")
    print(f"  p50={stats['p50']:.2f}ms | p95={stats['p95']:.2f}ms | p99={stats['p99']:.2f}ms")
    print(f"  throughput={report['images_per_s']:.1f} img/s")

    if dtype == "fp16":
        model.float()

    return report


# ---------------------------------------------------------------------------
def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1,
                img_size: int = 224, device: str = "cuda",
                warmup: int = 10, iters: int = 100, **kw) -> dict:
    """Đo độ trễ TTA K-view: xấp xỉ K lần forward một view.

    So sánh với k * p50_single để kiểm tra tuyến tính.
    """
    device_obj = torch.device(device)
    model      = model.to(device_obj).float().eval()
    dummy      = torch.randn(batch_size, 3, img_size, img_size, device=device_obj)

    sync_fn = torch.cuda.synchronize if device_obj.type == "cuda" else None

    @torch.no_grad()
    def _tta_forward():
        logits_list = []
        for _ in range(k_views):
            logits_list.append(model(dummy))
        # Aggregate (trung bình)
        return torch.stack(logits_list, 0).mean(0)

    stats = bench(_tta_forward, warmup=warmup, iters=iters, sync=sync_fn)

    print(f"[tta_latency] K={k_views} | p50={stats['p50']:.2f}ms | "
          f"p95={stats['p95']:.2f}ms | p99={stats['p99']:.2f}ms")

    return {
        "k_views": k_views,
        "p50_ms": stats["p50"],
        "p95_ms": stats["p95"],
        "p99_ms": stats["p99"],
        "mean_ms": stats["mean"],
        "images_per_s": batch_size / (stats["p50"] / 1000),
    }


# ---------------------------------------------------------------------------
def run_all_latency(model: nn.Module, img_size: int = 224, device: str = "cuda",
                    batch_sizes: list = None, dtypes: list = None) -> list:
    """Chạy full latency benchmark cho mọi tổ hợp batch_size x dtype.

    Kết quả là list dict để ghi vào sheet Latency của results.xlsx.
    """
    if batch_sizes is None:
        batch_sizes = [1, 32]
    if dtypes is None:
        dtypes = ["fp32", "amp", "fp16"]

    results = []
    for bs in batch_sizes:
        for dt in dtypes:
            try:
                rep = latency_report(model, batch_size=bs, img_size=img_size,
                                     dtype=dt, device=device)
                results.append(rep)
            except Exception as e:
                print(f"[run_all_latency] Lỗi batch={bs} dtype={dt}: {e}")
    return results
