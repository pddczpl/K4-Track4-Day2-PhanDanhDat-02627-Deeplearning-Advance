"""generate_report_charts.py - Tạo biểu đồ bổ trợ cho report.md:
1. curves/f1_vs_latency_backbones.png (Scatter plot F1 vs Latency các backbone)
2. curves/f1_vs_latency_inference.png (Trade-off F1 vs Latency các phương pháp suy luận)
3. curves/confusion_matrix_test.png (Ma trận nhầm lẫn của F01 trên test)
4. curves/calibration_curve.png (Đồ thị hiệu chuẩn độ tin cậy trước & sau Temperature Scaling)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
CURVES_DIR = ROOT / "curves"
CURVES_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT))

from eval import CLASS_NAMES, NUM_CLASSES, read_pred, compute_metrics


# ---------------------------------------------------------------------------
# 1. Scatter F1 vs Latency cho Backbones
# ---------------------------------------------------------------------------
def plot_backbone_tradeoff():
    bb_path = ROOT / "backbone_results.json"
    if not bb_path.exists():
        return
    data = json.loads(bb_path.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(9, 6))
    for item in data:
        exp_id = item["exp_id"]
        name = item["backbone"]
        f1 = item["val_macro_f1"]
        lat = item.get("latency_b1_ms", 5.0)
        params = item.get("params_m", 20.0)

        color = "tab:red" if exp_id == "B03" else ("tab:blue" if exp_id == "T00" else "tab:gray")
        marker = "*" if exp_id == "B03" else "o"
        size = 280 if exp_id == "B03" else 120

        ax.scatter(lat, f1, s=size, c=color, marker=marker, edgecolors="black", linewidths=1.2, zorder=5)
        offset_y = 0.008 if exp_id in ("B03", "B04") else -0.015
        ax.annotate(f"{exp_id}: {name}\n({f1:.3f}, {lat:.1f}ms)",
                    (lat, f1 + offset_y),
                    fontsize=9, weight="bold" if exp_id == "B03" else "normal",
                    ha="center")

    ax.set_title("Đánh đổi Macro-F1 vs Độ trễ suy luận (Batch 1, FP32, RTX 3060)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Độ trễ p50 (ms) - Càng nhỏ càng nhanh", fontsize=11)
    ax.set_ylabel("Validation Macro-F1 - Càng cao càng tốt", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_path = CURVES_DIR / "f1_vs_latency_backbones.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] Saved -> {out_path}")


# ---------------------------------------------------------------------------
# 2. Scatter F1 vs Latency cho Inference
# ---------------------------------------------------------------------------
def plot_inference_tradeoff():
    xl_path = ROOT / "results.xlsx"
    if not xl_path.exists():
        return
    df_inf = pd.read_excel(xl_path, sheet_name="Inference")
    df_valid = df_inf.dropna(subset=["val_macro_f1", "p50_ms"]).copy()

    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in df_valid.iterrows():
        exp_id = row["exp_id"]
        method = row["method"]
        f1 = float(row["val_macro_f1"])
        lat = float(row["p50_ms"])

        color = "tab:green" if exp_id == "I01" else ("tab:purple" if exp_id == "I08" else "tab:blue")
        ax.scatter(lat, f1, s=140, c=color, edgecolors="black", zorder=5)
        ax.annotate(f"{exp_id}\n({lat:.1f}ms)", (lat, f1 + 0.0015), fontsize=8.5, ha="center")

    ax.axvline(100.0, color="red", linestyle="--", linewidth=1.5, label="Ngân sách thời gian thực (100ms)")
    ax.set_title("Đánh đổi Macro-F1 vs Độ trễ qua các phương pháp suy luận (Mốc T00)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Độ trễ p50 (ms)", fontsize=11)
    ax.set_ylabel("Validation Macro-F1", fontsize=11)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_path = CURVES_DIR / "f1_vs_latency_inference.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] Saved -> {out_path}")


# ---------------------------------------------------------------------------
# 3. Confusion Matrix cho F01 trên Test
# ---------------------------------------------------------------------------
def plot_confusion_matrix_test():
    f01_test = ROOT / "predictions" / "F01_seed0_test.csv"
    if not f01_test.exists():
        return
    pred = read_pred(str(f01_test))
    cm = confusion_matrix(pred.y_true, pred.y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                cbar_kws={'label': 'Tỉ lệ dự đoán đúng (Recall theo hàng)'})
    ax.set_title("Ma trận nhầm lẫn mô hình chung kết F01 (ConvNeXt-Tiny) trên Test Fold 0", fontsize=12, fontweight="bold")
    ax.set_xlabel("Nhãn dự đoán (Predicted)", fontsize=11)
    ax.set_ylabel("Nhãn thực tế (Ground Truth)", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_path = CURVES_DIR / "confusion_matrix_test.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] Saved -> {out_path}")


# ---------------------------------------------------------------------------
# 4. Calibration Curve (Reliability Diagram) trước & sau Temperature Scaling
# ---------------------------------------------------------------------------
def compute_calibration_curve(probs, y_true, n_bins=15):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == y_true

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if np.sum(mask) > 0:
            bin_accs.append(np.mean(accuracies[mask]))
            bin_confs.append(np.mean(confidences[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_accs.append(0.0)
            bin_confs.append((bins[i] + bins[i+1])/2)
            bin_counts.append(0)

    return np.array(bin_confs), np.array(bin_accs), np.array(bin_counts)


def plot_calibration():
    uncal_path = ROOT / "predictions" / "F01_uncal_seed0_test.csv"
    cal_path   = ROOT / "predictions" / "F01_seed0_test.csv"
    if not (uncal_path.exists() and cal_path.exists()):
        return

    p_uncal = read_pred(str(uncal_path))
    p_cal   = read_pred(str(cal_path))

    m_uncal = compute_metrics(p_uncal.y_true, p_uncal.y_pred, p_uncal.probs)
    m_cal   = compute_metrics(p_cal.y_true, p_cal.y_pred, p_cal.probs)

    c_uncal, a_uncal, _ = compute_calibration_curve(p_uncal.probs, p_uncal.y_true)
    c_cal,   a_cal,   _ = compute_calibration_curve(p_cal.probs, p_cal.y_true)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Hiệu chuẩn hoàn hảo (y = x)")
    ax.plot(c_uncal, a_uncal, "s-", color="tab:red", label=f"Chưa hiệu chuẩn (ECE = {m_uncal['ece']:.4f})")
    ax.plot(c_cal,   a_cal,   "o-", color="tab:green", label=f"Sau Temperature Scaling (ECE = {m_cal['ece']:.4f})")

    ax.set_title("Biểu đồ hiệu chuẩn độ tin cậy (Reliability Diagram) trên Test", fontsize=12, fontweight="bold")
    ax.set_xlabel("Độ tin cậy trung bình (Mean Confidence)", fontsize=11)
    ax.set_ylabel("Độ chính xác thực tế (Accuracy)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_path = CURVES_DIR / "calibration_curve.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] Saved -> {out_path}")


if __name__ == "__main__":
    plot_backbone_tradeoff()
    plot_inference_tradeoff()
    plot_confusion_matrix_test()
    plot_calibration()
