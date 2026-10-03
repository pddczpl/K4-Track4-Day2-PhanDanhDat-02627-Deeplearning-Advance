"""run_experiments.py - Script chạy toàn bộ thí nghiệm theo GUIDE.md.

Chạy từ thư mục gốc repo:
    python code/run_experiments.py --step backbone
    python code/run_experiments.py --step training
    python code/run_experiments.py --step final
    python code/run_experiments.py --step all

Các bước suy luận (Bước 3) chạy tự động sau khi có model từ Bước 1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
CODE    = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE))

from train import Config, run

# ===========================================================================
# Cấu hình dường dẫn (chỉnh nếu data nằm chỗ khác)
# ===========================================================================
IMAGES_DIR = str(ROOT / "data" / "images")
LABELS_DIR = str(ROOT / "data" / "labels")
OUT_DIR    = str(ROOT / "runs")
PRED_DIR   = str(ROOT / "predictions")
CURVES_DIR = str(ROOT / "curves")

COMMON = dict(
    images_dir = IMAGES_DIR,
    labels_dir = LABELS_DIR,
    out_dir    = OUT_DIR,
    pred_dir   = PRED_DIR,
    curves_dir = CURVES_DIR,
    num_workers = 2,
    amp         = True,
    fold        = 0,

)

# Công thức nền (T00) - GUIDE.md mục 1.4
BASELINE = dict(
    backbone    = "resnet50",
    init        = "finetune",
    img_size    = 224,
    aug         = "basic",
    sampler     = None,
    mix         = None,
    loss        = "ce",
    label_smoothing = 0.0,
    focal_gamma = 2.0,
    class_weight_beta = None,
    epochs      = 12,
    batch_size  = 64,
    lr_backbone = 1e-4,
    lr_head     = 1e-3,
    weight_decay = 0.05,
    warmup_epochs = 1.0,
    ema_decay   = None,
)


# ===========================================================================
# Bước 1: So sánh Backbone (≥ 5 backbone, 1 seed)
# ===========================================================================
BACKBONE_EXPERIMENTS = [
    # Mốc ResNet (exp_id=T00 cũng là mốc công thức huấn luyện)
    dict(exp_id="T00",  backbone="resnet50",                  seed=0),
    # Backbone 2: ResNeXt
    dict(exp_id="B02",  backbone="resnext50_32x4d",           seed=0),
    # Backbone 3: ConvNeXt-Tiny (CNN hiện đại)
    dict(exp_id="B03",  backbone="convnext_tiny",             seed=0),
    # Backbone 4: DeiT-Small (Transformer)
    dict(exp_id="B04",  backbone="deit_small_patch16_224",    seed=0),
    # Backbone 5: Swin-Tiny (Transformer)
    dict(exp_id="B05",  backbone="swin_tiny_patch4_window7_224", seed=0),
    # Backbone 6: EfficientNet-B0 (mạng nhẹ)
    dict(exp_id="B06",  backbone="efficientnet_b0",           seed=0),
    # Backbone 7: MobileNetV3-Large (nhẹ nhất)
    dict(exp_id="B07",  backbone="mobilenetv3_large_100",     seed=0),
]


# ===========================================================================
# Bước 2: Công thức huấn luyện (≥ 3 trục, mỗi trục ≥ 2 giá trị)
# backbone tốt nhất sẽ được chọn từ Bước 1; mặc định: resnet50
# Đổi BEST_BACKBONE sau khi chạy xong Bước 1!
# ===========================================================================
BEST_BACKBONE = "resnet50"   # <--- CẬP NHẬT sau Bước 1 nếu cần

TRAINING_EXPERIMENTS = [
    # ---------- Trục A: Khởi tạo ----------
    dict(exp_id="T01", backbone=BEST_BACKBONE, init="scratch",  seed=0, epochs=12),
    dict(exp_id="T02", backbone=BEST_BACKBONE, init="frozen",   seed=0, epochs=12),
    # T00 = finetune (mốc đã chạy ở Bước 1)

    # ---------- Trục B: Augmentation ----------
    dict(exp_id="T03", backbone=BEST_BACKBONE, aug="color",   seed=0),
    dict(exp_id="T04", backbone=BEST_BACKBONE, aug="trivial", seed=0),
    dict(exp_id="T05", backbone=BEST_BACKBONE, aug="randaug", seed=0),
    dict(exp_id="T06", backbone=BEST_BACKBONE, mix="mixup",   seed=0),
    dict(exp_id="T07", backbone=BEST_BACKBONE, mix="cutmix",  seed=0),

    # ---------- Trục C: Loss ----------
    dict(exp_id="T08", backbone=BEST_BACKBONE, loss="ls",          label_smoothing=0.1, seed=0),
    dict(exp_id="T09", backbone=BEST_BACKBONE, loss="focal",       focal_gamma=2.0,     seed=0),
    dict(exp_id="T10", backbone=BEST_BACKBONE, loss="ce_weighted", class_weight_beta=0.0, seed=0),

    # ---------- Trục D: Sampler ----------
    dict(exp_id="T11", backbone=BEST_BACKBONE, sampler="balanced", seed=0),

    # ---------- Trục E: LR / Optimizer ----------
    dict(exp_id="T12", backbone=BEST_BACKBONE, lr_backbone=5e-5, lr_head=5e-4, seed=0),
    dict(exp_id="T13", backbone=BEST_BACKBONE, lr_backbone=2e-4, lr_head=2e-3, seed=0),

    # ---------- Trục F: Chính quy hoá – EMA ----------
    dict(exp_id="T14", backbone=BEST_BACKBONE, ema_decay=0.9998, seed=0),

    # ---------- Trục G: Số epoch ----------
    dict(exp_id="T15", backbone=BEST_BACKBONE, epochs=20, seed=0),

    # ---------- Kết hợp tốt nhất (cập nhật sau khi có kết quả ablation) ----------
    # Ví dụ: CutMix + Label Smoothing + EMA
    dict(exp_id="T16", backbone=BEST_BACKBONE, mix="cutmix", loss="ls",
         label_smoothing=0.1, ema_decay=0.9998, seed=0),
    # Ví dụ: RandAugment + Focal + EMA
    dict(exp_id="T17", backbone=BEST_BACKBONE, aug="randaug", loss="focal",
         focal_gamma=2.0, ema_decay=0.9998, seed=0),
]


# ===========================================================================
# Bước 4: Chung kết – cấu hình tốt nhất × 3 seed
# Cập nhật FINAL_CONFIG sau khi xem kết quả val của Bước 1+2!
# ===========================================================================
# Mốc (T00) cũng phải chạy đủ 3 seed để tính mức cải thiện
BASELINE_SEEDS = [0, 1, 2]

# Chung kết - cập nhật cfg theo kết quả Bước 1+2: convnext_tiny là backbone tốt nhất
FINAL_CONFIG = dict(
    backbone    = "convnext_tiny",
    init        = "finetune",
    img_size    = 224,
    aug         = "basic",
    mix         = None,
    loss        = "ls",
    label_smoothing = 0.1,
    ema_decay   = None,
    epochs      = 12,

    batch_size  = 64,
    lr_backbone = 1e-4,
    lr_head     = 1e-3,
    weight_decay = 0.05,
    warmup_epochs = 1.0,
)
FINAL_SEEDS = [0, 1, 2]


# ===========================================================================
# Helpers
# ===========================================================================
def make_cfg(exp_id, seed=0, save_test=False, **overrides) -> Config:
    """Tạo Config với baseline + overrides."""
    params = {**COMMON, **BASELINE, **overrides,
              "exp_id": exp_id, "seed": seed,
              "save_test_predictions": save_test}
    return Config(**params)


def save_summary(results: list, fname: str):
    """Lưu tóm tắt kết quả ra JSON."""
    out = ROOT / fname
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n[save_summary] → {out}")


# ===========================================================================
# Main runner
# ===========================================================================
def run_backbone_experiments():
    print("\n" + "="*70)
    print("BƯỚC 1: SO SÁNH BACKBONE")
    print("="*70)
    results = []
    for exp in BACKBONE_EXPERIMENTS:
        rdir = Path(OUT_DIR) / exp["exp_id"] / f"seed{exp.get('seed', 0)}"
        if (rdir / "result.json").exists():
            print(f"[skip] {exp['exp_id']} đã có kết quả, bỏ qua.")
            try:
                r = json.loads((rdir / "result.json").read_text(encoding="utf-8"))
                results.append(r)
                continue
            except Exception:
                pass
        cfg = make_cfg(**exp)
        try:
            r = run(cfg)
            results.append(r)
        except Exception as e:
            print(f"[ERROR] {exp['exp_id']}: {e}")
            results.append({"exp_id": exp["exp_id"], "error": str(e)})

    save_summary(results, "backbone_results.json")
    print_summary(results, "BACKBONE")
    return results


def run_training_experiments():
    print("\n" + "="*70)
    print("BƯỚC 2: CÔNG THỨC HUẤN LUYỆN")
    print("="*70)
    results = []
    for exp in TRAINING_EXPERIMENTS:
        # T00 đã chạy ở Bước 1 → bỏ qua nếu checkpoint đã tồn tại
        rdir = Path(OUT_DIR) / exp["exp_id"] / f"seed{exp.get('seed', 0)}"
        if (rdir / "result.json").exists():
            print(f"[skip] {exp['exp_id']} đã có kết quả, bỏ qua.")
            try:
                r = json.loads((rdir / "result.json").read_text())
                results.append(r)
            except Exception:
                pass
            continue
        cfg = make_cfg(**exp)
        try:
            r = run(cfg)
            results.append(r)
        except Exception as e:
            print(f"[ERROR] {exp['exp_id']}: {e}")
            results.append({"exp_id": exp["exp_id"], "error": str(e)})
    save_summary(results, "training_results.json")
    print_summary(results, "TRAINING")
    return results


def run_final_experiments():
    print("\n" + "="*70)
    print("BƯỚC 4: CHUNG KẾT (save_test=True)")
    print("="*70)
    results = []

    # Mốc T00 × 3 seed (lần đầu seed 0 đã có, cần thêm seed 1 và 2)
    for seed in BASELINE_SEEDS:
        exp_id = "T00"
        rdir   = Path(OUT_DIR) / exp_id / f"seed{seed}"
        pred_t = Path(PRED_DIR) / f"{exp_id}_seed{seed}_test.csv"
        if pred_t.exists():

            print(f"[skip] {exp_id}_seed{seed}_test.csv đã tồn tại.")
            continue
        if (rdir / "best.pth").exists():
            print(f"[eval only] {exp_id}_seed{seed} đã có checkpoint, đánh giá test...")
            import torch
            from dataset import load_split, build_transforms, make_loader
            from model import build_model
            from losses import build_criterion
            from eval import save_predictions, compute_metrics, NUM_CLASSES
            from train import evaluate, _softmax

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _, _, test_df = load_split(LABELS_DIR, fold=0)
            val_tfm = build_transforms(train=False, img_size=224)
            test_loader = make_loader(test_df, IMAGES_DIR, val_tfm, 128, train=False, num_workers=2)
            model = build_model("resnet50", pretrained=False, num_classes=NUM_CLASSES)
            ckpt = torch.load(rdir / "best.pth", map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            model = model.to(device).eval()
            criterion = build_criterion("ce")
            test_fnames, test_ytrue, test_logits, _ = evaluate(model, test_loader, criterion, device)
            test_probs = _softmax(test_logits)
            save_predictions(pred_t, test_fnames, test_ytrue, test_probs)
            print(f"  Test predictions → {pred_t}")
            r_json = rdir / "result.json"
            if r_json.exists():
                results.append(json.loads(r_json.read_text(encoding="utf-8")))
            continue
        cfg = make_cfg("T00", seed=seed, save_test=True)
        r   = run(cfg)
        results.append(r)


    # Chung kết F01 × 3 seed
    for seed in FINAL_SEEDS:
        exp_id = "F01"
        pred_t = Path(PRED_DIR) / f"{exp_id}_seed{seed}_test.csv"
        if pred_t.exists():
            print(f"[skip] {exp_id}_seed{seed}_test.csv đã tồn tại.")
            continue
        cfg = make_cfg(exp_id, seed=seed, save_test=True, **FINAL_CONFIG)
        r   = run(cfg)
        results.append(r)

    save_summary(results, "final_results.json")
    print_summary(results, "FINAL")
    return results


def print_summary(results, label):
    print(f"\n--- {label} SUMMARY ---")
    for r in results:
        if "error" in r:
            print(f"  {r.get('exp_id', '?')} ERROR: {r['error']}")
        else:
            print(f"  {r.get('exp_id', '?'):<8} | "
                  f"backbone={r.get('backbone','?'):<30} | "
                  f"F1={r.get('val_macro_f1', -1):.4f} | "
                  f"top1={r.get('val_top1', -1):.4f}")


# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepWeeds experiments")
    parser.add_argument("--step", choices=["backbone", "training", "final", "all"],
                        default="all", help="Bước cần chạy")
    parser.add_argument("--best-backbone", type=str, default=None,
                        help="Override backbone tốt nhất (mặc định: resnet50)")
    args = parser.parse_args()

    if args.best_backbone:
        BEST_BACKBONE = args.best_backbone
        # Cập nhật TRAINING_EXPERIMENTS và FINAL_CONFIG
        for exp in TRAINING_EXPERIMENTS:
            if "backbone" not in exp:
                exp["backbone"] = BEST_BACKBONE
        FINAL_CONFIG["backbone"] = BEST_BACKBONE

    t0 = time.time()

    if args.step in ("backbone", "all"):
        run_backbone_experiments()

    if args.step in ("training", "all"):
        run_training_experiments()

    if args.step in ("final", "all"):
        run_final_experiments()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Hoàn thành tất cả thí nghiệm trong {elapsed/60:.1f} phút")
    print(f"{'='*70}")
