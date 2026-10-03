"""run_inference.py - Bước 3: so sánh các phương pháp suy luận và đo độ trễ.

Chạy sau khi đã có ít nhất 1 backbone được huấn luyện xong (Bước 1).
Dùng trọng số tốt nhất của backbone được chỉ định.

Ví dụ:
    python code/run_inference.py --exp-id T00 --backbone resnet50 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE))

from eval import compute_metrics, save_predictions, NUM_CLASSES
from dataset import load_split, build_transforms, make_loader
from model import build_model
from inference import (
    predict_logits, aggregate_views,
    view_identity, view_hflip, view_vflip, view_rotate90, view_rotate180, view_rotate270,
    views_multicrop, views_multiscale,
    fit_temperature, apply_temperature,
    ensemble_probs, fuse_conv_bn, verify_fuse_conv_bn,
    _softmax,
)
from benchmark import latency_report, tta_latency

IMAGES_DIR = str(ROOT / "data" / "images")
LABELS_DIR = str(ROOT / "data" / "labels")
PRED_DIR   = ROOT / "predictions"
CURVES_DIR = ROOT / "curves"

PRED_DIR.mkdir(exist_ok=True)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def load_model_from_checkpoint(exp_id: str, backbone: str, seed: int, img_size: int = 224,
                                device="cuda") -> torch.nn.Module:
    """Tải model từ checkpoint tốt nhất của một thí nghiệm."""
    ckpt_path = ROOT / "runs" / exp_id / f"seed{seed}" / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")
    
    model = build_model(backbone, pretrained=False, num_classes=NUM_CLASSES)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    print(f"[load_model] Loaded {exp_id}/seed{seed} | val_F1={ckpt.get('val_macro_f1', '?')}")
    return model


def run_inference_experiments(exp_id: str, backbone: str, seed: int = 0,
                               batch_size: int = 128, img_size: int = 224,
                               num_workers: int = 2):
    """Chạy tất cả phương pháp suy luận và đo độ trễ.

    Kết quả lưu vào JSON và predictions/I*.csv để điền vào results.xlsx.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[run_inference] exp_id={exp_id} | backbone={backbone} | seed={seed}")
    print(f"  device={device}")

    # Tải dữ liệu val (không dùng test!)
    train_df, val_df, test_df = load_split(LABELS_DIR, fold=0)
    val_tfm = build_transforms(train=False, img_size=img_size)
    val_loader = make_loader(val_df, IMAGES_DIR, val_tfm,
                             batch_size=batch_size, train=False, num_workers=num_workers)

    # Tải model
    model = load_model_from_checkpoint(exp_id, backbone, seed, img_size, device)

    results = {}

    # ----------------------------------------------------------------
    # I00: 1-view mốc
    # ----------------------------------------------------------------
    print("\n--- I00: 1-view (mốc) ---")
    fnames, y_true, logits_1v = predict_logits(model, val_loader, device)
    probs_1v = _softmax_np(logits_1v)
    m = compute_metrics(y_true, probs_1v.argmax(1), probs_1v)
    results["I00"] = {
        "method": "1-view (mốc)", "macro_f1": m["macro_f1"], "top1": m["top1"],
        "ece": m["ece"], "notes": "resize(256)+centercrop(224)"
    }
    save_predictions(PRED_DIR / f"I00_{exp_id}_seed{seed}_val.csv",
                     fnames, y_true, probs_1v)
    print(f"  F1={m['macro_f1']:.4f} | top1={m['top1']:.4f} | ECE={m['ece']:.4f}")

    # Lưu logit để dùng lại (không cần chạy lại GPU)
    np.save(ROOT / "runs" / exp_id / f"seed{seed}" / "val_logits.npy", logits_1v)
    np.save(ROOT / "runs" / exp_id / f"seed{seed}" / "val_ytrue.npy", y_true)

    # ----------------------------------------------------------------
    # I01: TTA lật ngang (K=2)
    # ----------------------------------------------------------------
    print("\n--- I01: TTA HFlip (K=2) ---")
    _, _, logits_flip = predict_logits(model, val_loader, device, view=view_hflip)
    probs_tta2 = aggregate_views([logits_1v, logits_flip], space="prob")
    m = compute_metrics(y_true, probs_tta2.argmax(1), probs_tta2)
    results["I01"] = {
        "method": "TTA HFlip K=2", "macro_f1": m["macro_f1"], "top1": m["top1"],
        "ece": m["ece"], "notes": "avg_prob(1-view, hflip)"
    }
    print(f"  F1={m['macro_f1']:.4f} | top1={m['top1']:.4f} | ECE={m['ece']:.4f}")

    # ----------------------------------------------------------------
    # I02: TTA 4 phép quay + flip (K=8)
    # ----------------------------------------------------------------
    print("\n--- I02: TTA 8-view (4 quay + flip) ---")
    view_fns = [view_identity, view_hflip, view_vflip,
                view_rotate90, view_rotate180, view_rotate270]
    all_logits_k8 = [logits_1v]
    for vfn in view_fns[1:]:
        _, _, lg = predict_logits(model, val_loader, device, view=vfn)
        all_logits_k8.append(lg)

    probs_tta8_prob  = aggregate_views(all_logits_k8, space="prob")
    m = compute_metrics(y_true, probs_tta8_prob.argmax(1), probs_tta8_prob)
    results["I02"] = {
        "method": f"TTA 6-view (prob)", "macro_f1": m["macro_f1"], "top1": m["top1"],
        "ece": m["ece"], "notes": "6 views: identity,hflip,vflip,rot90/180/270"
    }
    print(f"  F1={m['macro_f1']:.4f} | top1={m['top1']:.4f} | ECE={m['ece']:.4f}")

    # ----------------------------------------------------------------
    # I03: Gộp prob vs logit
    # ----------------------------------------------------------------
    print("\n--- I03: Prob vs Logit aggregation ---")
    probs_tta_logit = aggregate_views(all_logits_k8, space="logit")
    m_logit = compute_metrics(y_true, probs_tta_logit.argmax(1), probs_tta_logit)
    results["I03_logit"] = {
        "method": "TTA 6-view (logit avg)", "macro_f1": m_logit["macro_f1"],
        "top1": m_logit["top1"], "ece": m_logit["ece"]
    }
    print(f"  Logit: F1={m_logit['macro_f1']:.4f} | Prob: F1={m['macro_f1']:.4f}")

    # ----------------------------------------------------------------
    # I04: FixRes – dò độ phân giải kiểm tra
    # ----------------------------------------------------------------
    print("\n--- I04: FixRes – test resolution ---")
    for test_size in [256, 288, 320]:
        val_tfm_fs = build_transforms(train=False, img_size=test_size)
        val_loader_fs = make_loader(val_df, IMAGES_DIR, val_tfm_fs,
                                    batch_size=batch_size, train=False, num_workers=num_workers)
        _, _, logits_fs = predict_logits(model, val_loader_fs, device)
        probs_fs = _softmax_np(logits_fs)
        m = compute_metrics(y_true, probs_fs.argmax(1), probs_fs)
        results[f"I04_{test_size}"] = {
            "method": f"FixRes {test_size}px", "macro_f1": m["macro_f1"],
            "top1": m["top1"], "ece": m["ece"]
        }
        print(f"  {test_size}px: F1={m['macro_f1']:.4f} | top1={m['top1']:.4f}")

    # ----------------------------------------------------------------
    # I07: Temperature Scaling (ECE trước/sau)
    # ----------------------------------------------------------------
    print("\n--- I07: Temperature Scaling ---")
    ece_before = compute_metrics(y_true, probs_1v.argmax(1), probs_1v)["ece"]
    T = fit_temperature(logits_1v, y_true)
    probs_cal = apply_temperature(logits_1v, T)
    m_cal = compute_metrics(y_true, probs_cal.argmax(1), probs_cal)
    ece_after = m_cal["ece"]
    results["I07"] = {
        "method": "Temperature Scaling", "T": T,
        "ece_before": ece_before, "ece_after": ece_after,
        "macro_f1": m_cal["macro_f1"], "top1": m_cal["top1"],
        "notes": "accuracy không thay đổi"
    }
    print(f"  T={T:.4f} | ECE: {ece_before:.4f} → {ece_after:.4f}")
    print(f"  F1={m_cal['macro_f1']:.4f} | top1={m_cal['top1']:.4f}")

    # Lưu temperature để dùng khi test
    (ROOT / "runs" / exp_id / f"seed{seed}" / "temperature.json").write_text(
        json.dumps({"T": T, "ece_val_before": ece_before, "ece_val_after": ece_after}),
        encoding="utf-8"
    )

    # Lưu dự đoán chưa/đã calibrate (để chấm I4)
    save_predictions(PRED_DIR / f"I07uncal_{exp_id}_seed{seed}_val.csv", fnames, y_true, probs_1v)
    save_predictions(PRED_DIR / f"I07cal_{exp_id}_seed{seed}_val.csv",   fnames, y_true, probs_cal)

    # ----------------------------------------------------------------
    # I08: Fused Conv-BN
    # ----------------------------------------------------------------
    print("\n--- I08: Fused Conv-BN ---")
    try:
        diff = verify_fuse_conv_bn(model, img_size)
        print(f"  Max diff trước/sau gộp: {diff:.2e}")
        fused_model = fuse_conv_bn(model).to(device)
        _, _, logits_fused = predict_logits(fused_model, val_loader, device)
        probs_fused = _softmax_np(logits_fused)
        m_fused = compute_metrics(y_true, probs_fused.argmax(1), probs_fused)
        results["I08"] = {
            "method": "Fused Conv-BN", "macro_f1": m_fused["macro_f1"],
            "top1": m_fused["top1"], "max_diff": diff
        }
        print(f"  F1={m_fused['macro_f1']:.4f} | top1={m_fused['top1']:.4f}")
    except Exception as e:
        print(f"  [warn] fuse_conv_bn: {e}")
        results["I08"] = {"method": "Fused Conv-BN", "error": str(e)}

    # ----------------------------------------------------------------
    # Đo độ trễ (GUIDE.md mục 4.1)
    # ----------------------------------------------------------------
    print("\n--- Đo độ trễ ---")
    latency_results = []

    if torch.cuda.is_available():
        for bs in [1, 32]:
            for dt in ["fp32", "amp", "fp16"]:
                try:
                    rep = latency_report(model, batch_size=bs, img_size=img_size,
                                         dtype=dt, device="cuda", warmup=20, iters=200)
                    rep["exp_id"] = exp_id
                    rep["backbone"] = backbone
                    rep["fused_bn"] = False
                    latency_results.append(rep)
                except Exception as e:
                    print(f"  [warn] bs={bs} dtype={dt}: {e}")

        # TTA latency
        for k in [2, 6]:
            try:
                rep_tta = tta_latency(model, k_views=k, batch_size=1,
                                      img_size=img_size, device="cuda",
                                      warmup=20, iters=200)
                rep_tta["exp_id"] = exp_id
                rep_tta["backbone"] = backbone
                rep_tta["dtype"] = "fp32"
                latency_results.append(rep_tta)
            except Exception as e:
                print(f"  [warn] TTA k={k}: {e}")

        # FixRes latency (batch 1, fp32)
        for sz in [256, 288, 320]:
            try:
                rep_sz = latency_report(model, batch_size=1, img_size=sz,
                                        dtype="fp32", device="cuda", warmup=20, iters=200)
                rep_sz["exp_id"] = exp_id
                rep_sz["backbone"] = backbone
                rep_sz["fused_bn"] = False
                latency_results.append(rep_sz)
            except Exception as e:
                print(f"  [warn] size={sz}: {e}")

        # Fused BN latency
        try:
            fused_model_gpu = fuse_conv_bn(model).to("cuda")
            rep_fused = latency_report(fused_model_gpu, batch_size=1, img_size=img_size,
                                        dtype="fp32", device="cuda", warmup=20, iters=200)
            rep_fused["exp_id"] = exp_id
            rep_fused["backbone"] = backbone
            rep_fused["fused_bn"] = True
            latency_results.append(rep_fused)
        except Exception as e:
            print(f"  [warn] fused latency: {e}")

    # ----------------------------------------------------------------
    # Lưu kết quả
    # ----------------------------------------------------------------
    out_json = ROOT / "runs" / exp_id / f"seed{seed}" / "inference_results.json"
    out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    lat_json = ROOT / "runs" / exp_id / f"seed{seed}" / "latency_results.json"
    lat_json.write_text(json.dumps(latency_results, indent=2, default=str), encoding="utf-8")

    print(f"\n[run_inference] Kết quả → {out_json}")
    print(f"[run_inference] Latency  → {lat_json}")

    print("\n=== INFERENCE SUMMARY ===")
    for key, val in results.items():
        if "error" not in val:
            print(f"  {key:<12} | method={val['method']:<30} | "
                  f"F1={val.get('macro_f1', 0):.4f} | "
                  f"top1={val.get('top1', 0):.4f} | "
                  f"ECE={val.get('ece', float('nan')):.4f}")

    return results, latency_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference experiments")
    parser.add_argument("--exp-id",   type=str, default="T00",      help="exp_id của model (ví dụ: T00)")
    parser.add_argument("--backbone", type=str, default="resnet50",  help="Tên backbone")
    parser.add_argument("--seed",     type=int, default=0,           help="Seed")
    parser.add_argument("--img-size", type=int, default=224,         help="Kích thước ảnh")
    parser.add_argument("--batch",    type=int, default=128,         help="Batch size cho inference")
    args = parser.parse_args()

    run_inference_experiments(
        exp_id=args.exp_id,
        backbone=args.backbone,
        seed=args.seed,
        batch_size=args.batch,
        img_size=args.img_size,
    )
