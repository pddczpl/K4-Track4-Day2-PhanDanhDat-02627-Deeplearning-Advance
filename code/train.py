"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H).

Chạy từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
"""
from __future__ import annotations

import dataclasses
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.amp as amp


# Thêm thư mục gốc repo (chứa eval.py) vào path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval import save_predictions, compute_metrics, NUM_CLASSES

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from dataset import load_split, check_split, build_transforms, make_loader, NUM_CLASSES as DS_NUM_CLASSES
from model import build_model, param_groups, count_params, count_gmacs, _set_bn_eval
from losses import build_criterion, class_weights, mix_batch, mixed_loss


# ---------------------------------------------------------------------------
@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: Optional[str] = None     # None | balanced
    mix: Optional[str] = None         # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: Optional[float] = None
    # --- tối ưu ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: Optional[float] = None
    amp: bool = True
    num_workers: int = 2

    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết) ---
    save_test_predictions: bool = False
    # --- preload ảnh vào RAM (giảm nghẽn I/O) ---
    preload: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Để tái lập hoàn toàn (chậm hơn ~10-20%)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
def build_optimizer(model: nn.Module, cfg: Config):
    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups, lr=cfg.lr_backbone, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0.

    Cập nhật theo bước (step). LR tăng tuyến tính trong warmup_steps bước đầu,
    sau đó cosine decay về 1e-6 * lr ban đầu trong phần còn lại.
    """
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    total_steps  = cfg.epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        # cosine từ 1.0 xuống 1e-5
        return max(1e-5, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow: dict = {}
        for name, param in model.state_dict().items():
            self.shadow[name] = param.detach().clone()

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if param.is_floating_point():
                    self.shadow[name].lerp_(param.detach(), 1.0 - self.decay)
                else:
                    self.shadow[name].copy_(param.detach())

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        """Áp EMA weights vào model (để đánh giá). Dùng cùng với restore_from."""
        self._backup = {n: p.clone() for n, p in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)

    @torch.no_grad()
    def restore_from(self, model: nn.Module) -> None:
        """Khôi phục trọng số gốc sau apply_to."""
        model.load_state_dict(self._backup, strict=False)
        del self._backup



# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    cfg: Config, device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về dict {train_loss, lr}."""
    model.train()
    if cfg.init == "frozen":
        _set_bn_eval(model)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        targets = labels  # mặc định
        do_mix  = (cfg.mix is not None) and (cfg.mix != "none")

        if do_mix:
            images, targets = mix_batch(images, labels, alpha=cfg.mix_alpha, mode=cfg.mix)

        optimizer.zero_grad()

        if cfg.amp and device.type == "cuda":
            with amp.autocast("cuda"):
                logits = model(images)

                if do_mix:
                    loss = mixed_loss(criterion, logits, targets)
                else:
                    loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            if do_mix:
                loss = mixed_loss(criterion, logits, targets)
            else:
                loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        if ema is not None:
            ema.update(model)

        with torch.no_grad():
            total_loss    += loss.item() * images.size(0)
            if not do_mix:
                total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    train_acc = total_correct / total_samples if total_samples > 0 else 0.0

    # LR hiện tại (nhóm đầu tiên)
    current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else \
                 optimizer.param_groups[0]["lr"]

    return {"train_loss": avg_loss, "train_acc": train_acc, "lr": current_lr}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Chạy model trên một loader ở chế độ eval.

    Trả về (filenames, y_true, logits [N,9], loss).
    """
    model.eval()
    all_filenames, all_ytrue, all_logits = [], [], []
    total_loss = 0.0
    total_samples = 0

    for images, labels, fnames in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with amp.autocast("cuda", enabled=False):  # FP32 tường minh để đảm bảo consistency
            logits = model(images).float()


        loss = criterion(logits, labels)
        total_loss    += loss.item() * images.size(0)
        total_samples += images.size(0)

        all_filenames.extend(fnames)
        all_ytrue.append(labels.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

    y_true  = np.concatenate(all_ytrue)
    logits  = np.concatenate(all_logits)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return all_filenames, y_true, logits, avg_loss


# ---------------------------------------------------------------------------
def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training → curves/<exp_id>_<mota>.png."""
    epochs = list(range(1, len(history) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    ax = axes[0]
    ax.plot(epochs, [h["train_loss"] for h in history], label="Train Loss", marker="o", ms=3)
    ax.plot(epochs, [h["val_loss"]   for h in history], label="Val Loss",   marker="s", ms=3)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Macro-F1
    ax = axes[1]
    ax.plot(epochs, [h["val_macro_f1"] for h in history], label="Val Macro-F1", marker="s", ms=3, color="tab:orange")
    ax.plot(epochs, [h.get("train_acc", 0) for h in history], label="Train Acc", marker="o", ms=3, color="tab:blue", linestyle="--")
    ax.set_title("Macro-F1 Val / Train Acc")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.legend(); ax.grid(True, alpha=0.3)

    # LR
    ax = axes[2]
    ax.plot(epochs, [h["lr"] for h in history], label="LR", color="tab:green")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_curves] Lưu → {path}")


# ---------------------------------------------------------------------------
def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết.

    Trả về dict kết quả tóm tắt.
    """
    # 1. Seed & thư mục
    set_seed(cfg.seed)
    rdir = run_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"[run] exp_id={cfg.exp_id} | backbone={cfg.backbone} | seed={cfg.seed}")
    print(f"  out_dir={rdir}")
    (rdir / "config.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")

    # 2. Dataset
    train_df, val_df, test_df = load_split(cfg.labels_dir, fold=cfg.fold)
    check_info = check_split(train_df, val_df, test_df, cfg.images_dir)
    (rdir / "split_info.json").write_text(
        json.dumps({k: str(v) for k, v in check_info["n"].items()}, indent=2),
        encoding="utf-8"
    )

    train_tfm = build_transforms(train=True,  img_size=cfg.img_size, aug=cfg.aug)
    val_tfm   = build_transforms(train=False, img_size=cfg.img_size)

    train_loader = make_loader(train_df, cfg.images_dir, train_tfm,
                               cfg.batch_size, train=True,
                               sampler=cfg.sampler, num_workers=cfg.num_workers,
                               preload=cfg.preload)
    val_loader   = make_loader(val_df,   cfg.images_dir, val_tfm,
                               cfg.batch_size * 2, train=False,
                               num_workers=cfg.num_workers, preload=cfg.preload)

    test_loader = None
    if cfg.save_test_predictions:
        test_loader = make_loader(test_df, cfg.images_dir, val_tfm,
                                  cfg.batch_size * 2, train=False,
                                  num_workers=cfg.num_workers)

    # 3. Model
    model = build_model(cfg.backbone, pretrained=(cfg.init != "scratch"),
                        num_classes=NUM_CLASSES, drop_rate=cfg.drop_rate,
                        init=cfg.init).to(device)

    params_m  = count_params(model)
    gmacs     = count_gmacs(model, cfg.img_size)
    print(f"  Params={params_m:.2f}M | GMACs={gmacs:.2f}")

    # 4. Loss
    crit_kw: dict = {}
    if cfg.loss in ("ls",):
        crit_kw["smoothing"] = cfg.label_smoothing
    elif cfg.loss == "focal":
        crit_kw["gamma"] = cfg.focal_gamma
    elif cfg.loss == "ce_weighted":
        counts = [train_df[train_df["Label"] == c].shape[0] for c in range(NUM_CLASSES)]
        beta   = cfg.class_weight_beta if cfg.class_weight_beta is not None else 0.0
        w      = class_weights(counts, beta=beta).to(device)
        crit_kw["weight"] = w
        if cfg.label_smoothing > 0:
            crit_kw["smoothing"] = cfg.label_smoothing

    criterion = build_criterion(cfg.loss, **crit_kw)

    # 5. Optimizer / Scheduler / Scaler / EMA
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler    = amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None


    # 6. Training loop
    best_f1      = -1.0
    best_epoch   = -1
    best_ckpt    = rdir / "best.pth"
    history: list[dict] = []

    print(f"\n  epochs={cfg.epochs} | batch={cfg.batch_size} | steps/epoch={steps_per_epoch}")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema
        )

        # Đánh giá val – dùng EMA nếu có
        if ema is not None:
            ema.apply_to(model)

        val_fnames, val_ytrue, val_logits, val_loss = evaluate(model, val_loader, criterion, device)

        if ema is not None:
            ema.restore_from(model)

        # Tính chỉ số val
        val_probs   = _softmax(val_logits)
        val_metrics = compute_metrics(val_ytrue, val_probs.argmax(1), val_probs)
        macro_f1    = val_metrics["macro_f1"]

        epoch_time = time.time() - t0
        row = {
            "epoch": epoch,
            **train_metrics,
            "val_loss": val_loss,
            "val_macro_f1": macro_f1,
            "val_top1": val_metrics["top1"],
            "epoch_time_s": epoch_time,
        }
        history.append(row)

        print(f"  Epoch {epoch:3d}/{cfg.epochs} | "
              f"loss={train_metrics['train_loss']:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_F1={macro_f1:.4f} | "
              f"val_top1={val_metrics['top1']:.4f} | "
              f"t={epoch_time:.1f}s")

        # Chọn checkpoint theo macro-F1 val (hòa → epoch sớm hơn)
        if macro_f1 > best_f1:
            best_f1    = macro_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_state":   ema.shadow if ema else None,
                "optimizer":   optimizer.state_dict(),
                "val_macro_f1": macro_f1,
            }, best_ckpt)

    print(f"\n  Best epoch={best_epoch} | Best val Macro-F1={best_f1:.4f}")

    # 7. Nạp checkpoint tốt nhất và lưu val predictions
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    if ema and ckpt.get("ema_state"):
        ema.shadow = ckpt["ema_state"]
        ema.apply_to(model)

    val_fnames, val_ytrue, val_logits, _ = evaluate(model, val_loader, criterion, device)
    val_probs = _softmax(val_logits)

    save_predictions(pred_path(cfg, "val"), val_fnames, val_ytrue, val_probs)
    print(f"  Val predictions → {pred_path(cfg, 'val')}")

    # 8. (Chỉ Bước 4) Đánh giá test một lần
    if cfg.save_test_predictions and test_loader is not None:
        test_fnames, test_ytrue, test_logits, _ = evaluate(model, test_loader, criterion, device)
        test_probs = _softmax(test_logits)

        if cfg.exp_id.startswith("F"):
            from inference import fit_temperature, apply_temperature
            uncal_path = Path(cfg.pred_dir) / f"{cfg.exp_id}_uncal_seed{cfg.seed}_test.csv"
            save_predictions(uncal_path, test_fnames, test_ytrue, test_probs)
            print(f"  Uncalibrated test predictions → {uncal_path}")

            # Khớp T trên VAL (tuân thủ S2, S4: KHÔNG khớp trên test)
            T = fit_temperature(val_logits, val_ytrue)
            test_probs_cal = apply_temperature(test_logits, T)
            save_predictions(pred_path(cfg, "test"), test_fnames, test_ytrue, test_probs_cal)
            print(f"  Calibrated test predictions (T={T:.4f}) → {pred_path(cfg, 'test')}")
            (rdir / "temperature.json").write_text(json.dumps({"T": T}), encoding="utf-8")
        else:
            save_predictions(pred_path(cfg, "test"), test_fnames, test_ytrue, test_probs)
            print(f"  Test predictions → {pred_path(cfg, 'test')}")

    
    if ema and ckpt.get("ema_state"):
        ema.restore_from(model)

    # 9. Lưu history, vẽ curve
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(rdir / "history.csv", index=False)

    curves_dir = Path(cfg.curves_dir)
    curves_dir.mkdir(parents=True, exist_ok=True)
    curve_path = curves_dir / f"{cfg.exp_id}_{cfg.backbone.replace('/', '_')}.png"
    plot_curves(history, curve_path, title=f"{cfg.exp_id} | {cfg.backbone} | seed={cfg.seed}")

    # 10. Đo độ trễ sơ bộ batch 1 (RUBRIC mục B)
    lat_b1 = None
    if device.type == "cuda":
        try:
            from benchmark import latency_report
            lat_rep = latency_report(model, batch_size=1, img_size=cfg.img_size, dtype="fp32", device="cuda", warmup=10, iters=50)
            lat_b1 = round(lat_rep["p50_ms"], 2)
            (rdir / "latency_results.json").write_text(json.dumps([lat_rep], indent=2), encoding="utf-8")
        except Exception as e:
            print(f"  [warn] Đo độ trễ sơ bộ lỗi: {e}")

    # 11. Trả kết quả tóm tắt
    val_final = compute_metrics(val_ytrue, val_probs.argmax(1), val_probs)
    result = {
        "exp_id":        cfg.exp_id,
        "backbone":      cfg.backbone,
        "seed":          cfg.seed,
        "best_epoch":    best_epoch,
        "val_macro_f1":  float(val_final["macro_f1"]),
        "val_top1":      float(val_final["top1"]),
        "val_ece":       float(val_final["ece"]),
        "params_m":      params_m,
        "gmacs":         gmacs,
        "epoch_time_s":  float(hist_df["epoch_time_s"].mean()),
        "latency_b1_ms": lat_b1,
    }

    (rdir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[run] DONE: val_macro_f1={result['val_macro_f1']:.4f} | val_top1={result['val_top1']:.4f}")
    return result


# ---------------------------------------------------------------------------
def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


# ---------------------------------------------------------------------------
def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal'] thành dict, ép kiểu theo field của Config."""
    fields = {f.name: f for f in dataclasses.fields(Config)}
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override phải ở dạng KEY=VALUE, nhận: '{pair}'")
        key, val = pair.split("=", 1)
        if key not in fields:
            raise KeyError(f"Config không có field '{key}'. Các field hợp lệ: {list(fields.keys())}")
        ftype = fields[key].type
        # Ép kiểu đơn giản
        if val.lower() == "none":
            out[key] = None
        elif ftype in ("bool", "Optional[bool]") or ftype is bool:
            out[key] = val.lower() in ("true", "1", "yes")
        elif ftype in ("int",):
            out[key] = int(val)
        elif ftype in ("float",):
            out[key] = float(val)
        elif ftype in ("Optional[float]",):
            out[key] = float(val)
        elif ftype in ("Optional[str]",):
            out[key] = None if val.lower() == "none" else str(val)
        else:
            out[key] = str(val)
    return out


def main() -> None:
    """Điểm vào: python train.py --set exp_id=B01 backbone=resnet50 seed=0"""
    import argparse
    parser = argparse.ArgumentParser(description="DeepWeeds trainer")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override Config fields, ví dụ: exp_id=B01 backbone=resnet50")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    result = run(cfg)
    print("\n=== Kết quả ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
