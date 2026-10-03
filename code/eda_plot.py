"""eda_plot.py - Phân tích phân bố dữ liệu và vẽ biểu đồ EDA.

Lưu ảnh ra curves/eda_class_distribution.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "data" / "labels"
CURVES_DIR = ROOT / "curves"
CURVES_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]

def main():
    train_df = pd.read_csv(LABELS_DIR / "train_subset0.csv")
    val_df   = pd.read_csv(LABELS_DIR / "val_subset0.csv")
    test_df  = pd.read_csv(LABELS_DIR / "test_subset0.csv")
    labels_df = pd.read_csv(LABELS_DIR / "labels.csv")

    # Đếm theo class
    train_counts = [train_df[train_df["Label"] == i].shape[0] for i in range(9)]
    val_counts   = [val_df[val_df["Label"] == i].shape[0] for i in range(9)]
    test_counts  = [test_df[test_df["Label"] == i].shape[0] for i in range(9)]
    total_counts = [labels_df[labels_df["Label"] == i].shape[0] for i in range(9)]

    # In ra bảng đếm
    print("Class Distribution:")
    print(f"{'Class':<16} | {'Label':<5} | {'Train':<6} | {'Val':<6} | {'Test':<6} | {'Total':<6} | {'Paper Table 1':<12}")
    print("-" * 75)
    paper_counts = [1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016, 9106]
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<16} | {i:<5} | {train_counts[i]:<6} | {val_counts[i]:<6} | {test_counts[i]:<6} | {total_counts[i]:<6} | {paper_counts[i]:<12}")

    # Vẽ biểu đồ phân bố lớp
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    r1 = ax.bar(x - width, train_counts, width, label='Train (Fold 0)', color='#2b5c8f')
    r2 = ax.bar(x, val_counts, width, label='Val (Fold 0)', color='#e28743')
    r3 = ax.bar(x + width, test_counts, width, label='Test (Fold 0)', color='#218c74')

    ax.set_ylabel('Số lượng ảnh (Images)')
    ax.set_title('Phân bố lớp trên tập DeepWeeds (Fold 0: 60/20/20)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha='right', fontsize=10)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Hiển thị số lượng trên cột
    for rects in [r1, r2, r3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=90)

    # Thêm ghi chú về mất cân bằng
    ax.text(0.02, 0.95,
            f"Tổng số ảnh: 17,509\nLớp Negatives: 9,106 ({9106/17509*100:.1f}%)\n8 loài cỏ: ~1,000-1,125 ảnh/loài (~6% mỗi loài)",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7f1e3', alpha=0.8, edgecolor='#aaa'))

    fig.tight_layout()
    out_path = CURVES_DIR / "eda_class_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Lưu biểu đồ EDA → {out_path}")

if __name__ == "__main__":
    main()
