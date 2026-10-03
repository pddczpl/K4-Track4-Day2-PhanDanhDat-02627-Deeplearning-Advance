"""build_results.py - Tạo results.xlsx từ các file JSON sau khi chạy thí nghiệm.

Chạy sau khi hoàn thành Bước 1, 2, 3, 4:
    python code/build_results.py

Output: results.xlsx với các sheet: Backbones, Training, Inference, Final, PerClass, Latency, Summary
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT  = Path(__file__).resolve().parent.parent
CODE  = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import compute_metrics, read_pred, NUM_CLASSES, CLASS_NAMES, expand


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_result(exp_id: str, seed: int):
    rdir = ROOT / "runs" / exp_id / f"seed{seed}"
    return load_json(rdir / "result.json")


def collect_backbone_results():
    """Sheet: Backbones"""
    rows = []
    backbone_exps = [
        ("T00", "resnet50"),
        ("B02", "resnext50_32x4d"),
        ("B03", "convnext_tiny"),
        ("B04", "deit_small_patch16_224"),
        ("B05", "swin_tiny_patch4_window7_224"),
        ("B06", "efficientnet_b0"),
        ("B07", "mobilenetv3_large_100"),
    ]
    for exp_id, backbone in backbone_exps:
        r = read_result(exp_id, 0)
        if not r:
            print(f"[warn] {exp_id} chưa có result.json")
            rows.append({
                "exp_id": exp_id, "backbone": backbone, "seed": 0,
                "val_macro_f1": "?", "val_top1": "?",
            })
            continue

        # Đọc latency nếu có
        lat_path = ROOT / "runs" / exp_id / "seed0" / "latency_results.json"
        lat = load_json(lat_path)
        p50_batch1 = "?"
        if lat:
            fp32_batch1 = [x for x in lat if isinstance(x, dict) and
                           x.get("batch") == 1 and x.get("dtype") == "fp32"]
            if fp32_batch1:
                p50_batch1 = round(fp32_batch1[0]["p50_ms"], 2)

        rows.append({
            "exp_id":         exp_id,
            "backbone":       backbone,
            "tag_weights":    "pretrained_imagenet",
            "params_M":       round(r.get("params_m", -1), 2),
            "gmacs":          round(r.get("gmacs", -1), 2),
            "img_size":       224,
            "epochs":         r.get("best_epoch", "?"),
            "seed":           0,
            "val_macro_f1":   round(r.get("val_macro_f1", 0), 4),
            "val_top1":       round(r.get("val_top1", 0), 4),
            "ece_val":        round(r.get("val_ece", 0), 4),
            "epoch_time_s":   round(r.get("epoch_time_s", -1), 1),
            "latency_b1_ms":  p50_batch1,
            "notes":          "",
        })
    return pd.DataFrame(rows)


def collect_training_results():
    """Sheet: Training"""
    TRAINING_EXPS = {
        "T00": "mốc (baseline)",
        "T01": "A: scratch",
        "T02": "A: frozen backbone",
        "T03": "B: aug=color",
        "T04": "B: aug=trivial",
        "T05": "B: aug=randaug",
        "T06": "B: mix=mixup",
        "T07": "B: mix=cutmix",
        "T08": "C: loss=ls (ε=0.1)",
        "T09": "C: loss=focal (γ=2)",
        "T10": "C: loss=ce_weighted",
        "T11": "D: sampler=balanced",
        "T12": "E: lr×0.5",
        "T13": "E: lr×2",
        "T14": "F: EMA(d=0.9998)",
        "T15": "G: epochs=20",
        "T16": "Combo: cutmix+ls+ema",
        "T17": "Combo: randaug+focal+ema",
    }
    rows = []
    base_f1 = None

    for exp_id, desc in TRAINING_EXPS.items():
        r = read_result(exp_id, 0)
        if exp_id == "T00" and r:
            base_f1 = r.get("val_macro_f1", 0)

        f1 = r.get("val_macro_f1", float("nan")) if r else float("nan")
        top1 = r.get("val_top1", float("nan")) if r else float("nan")
        delta = round(f1 - base_f1, 4) if (base_f1 is not None and not np.isnan(f1)) else "?"

        f1_chinee = "?"
        f1_snake = "?"
        pred_val = ROOT / "predictions" / f"{exp_id}_seed0_val.csv"
        if pred_val.exists():
            try:
                pv = read_pred(str(pred_val))
                mv = compute_metrics(pv.y_true, pv.y_pred, pv.probs)
                f1_chinee = round(float(mv["f1"][0]), 4)
                f1_snake  = round(float(mv["f1"][7]), 4)
            except Exception:
                pass

        lat_path = ROOT / "runs" / exp_id / "seed0" / "latency_results.json"
        lat = load_json(lat_path)
        p50_b1 = "?"
        if lat:
            fp32_b1 = [x for x in lat if isinstance(x, dict) and
                       x.get("batch") == 1 and x.get("dtype") == "fp32"]
            if fp32_b1:
                p50_b1 = round(fp32_b1[0]["p50_ms"], 2)

        rows.append({
            "exp_id":          exp_id,
            "backbone":        r.get("backbone", "resnet50") if r else "resnet50",
            "truc_thay_doi":   desc,
            "khac_T00":        desc,
            "seed":            0,
            "val_macro_f1":    round(f1, 4) if not np.isnan(f1) else "?",
            "val_top1":        round(top1, 4) if not np.isnan(top1) else "?",
            "ece_val":         round(r.get("val_ece", 0), 4) if r else "?",
            "delta_vs_T00":    delta,
            "f1_chinee_apple": f1_chinee,
            "f1_snake_weed":   f1_snake,
            "latency_b1_ms":   p50_b1,
            "notes":           "",
        })

    return pd.DataFrame(rows)


def collect_inference_results():
    """Sheet: Inference"""
    rows = []
    # Đọc từ inference_results.json của thí nghiệm mốc
    for inf_json in sorted(ROOT.glob("runs/*/seed*/inference_results.json")):
        data = load_json(inf_json)
        exp_id = inf_json.parent.parent.name
        seed   = int(inf_json.parent.name.replace("seed", ""))
        lat_path = inf_json.parent / "latency_results.json"
        lat_list = load_json(lat_path) if lat_path.exists() else []

        p50_base = None
        for x in lat_list:
            if isinstance(x, dict) and x.get("batch") == 1 and x.get("dtype") == "fp32" and not x.get("fused_bn") and "k_views" not in x and x.get("img_size", 224) == 224:
                p50_base = x.get("p50_ms")
                break

        for key, val in data.items():
            if not isinstance(val, dict):
                continue

            lat_item = None
            if key in ("I00", "I07"):
                for x in lat_list:
                    if isinstance(x, dict) and x.get("batch") == 1 and x.get("dtype") == "fp32" and not x.get("fused_bn") and "k_views" not in x and x.get("img_size", 224) == 224:
                        lat_item = x
                        break
            elif key == "I01":
                for x in lat_list:
                    if isinstance(x, dict) and x.get("k_views") == 2:
                        lat_item = x
                        break
            elif key in ("I02", "I03", "I03_logit"):
                for x in lat_list:
                    if isinstance(x, dict) and x.get("k_views") == 6:
                        lat_item = x
                        break
            elif key == "I08":
                for x in lat_list:
                    if isinstance(x, dict) and x.get("fused_bn") is True:
                        lat_item = x
                        break
            elif key.startswith("I04_"):
                try:
                    sz = int(key.split("_")[1])
                    for x in lat_list:
                        if isinstance(x, dict) and x.get("img_size") == sz and x.get("batch") == 1 and x.get("dtype") == "fp32" and not x.get("fused_bn"):
                            lat_item = x
                            break
                except Exception:
                    pass

            p50 = round(lat_item["p50_ms"], 2) if lat_item and "p50_ms" in lat_item else "?"
            p95 = round(lat_item["p95_ms"], 2) if lat_item and "p95_ms" in lat_item else "?"
            p99 = round(lat_item["p99_ms"], 2) if lat_item and "p99_ms" in lat_item else "?"
            thp = round(lat_item["images_per_s"], 1) if lat_item and "images_per_s" in lat_item else (
                round(1000.0 / lat_item["p50_ms"], 1) if lat_item and "p50_ms" in lat_item and lat_item["p50_ms"] > 0 else "?"
            )
            cost_rel = round(lat_item["p50_ms"] / p50_base, 2) if lat_item and p50_base and "p50_ms" in lat_item else (1.0 if key == "I00" else "?")

            k_views = 1
            if key == "I01":
                k_views = 2
            elif key in ("I02", "I03", "I03_logit"):
                k_views = 6

            rows.append({
                "exp_id":       key,
                "method":       val.get("method", key),
                "model_ckpt":   f"{exp_id}/seed{seed}",
                "K_views":      k_views,
                "val_macro_f1": round(val.get("macro_f1", float("nan")), 4),
                "val_top1":     round(val.get("top1", float("nan")), 4),
                "ece_val":      round(val.get("ece", float("nan")), 4),
                "p50_ms":       p50,
                "p95_ms":       p95,
                "p99_ms":       p99,
                "throughput_img_s": thp,
                "cost_rel_I00": cost_rel,
                "notes":        val.get("notes", ""),
            })

    if not rows:
        # Placeholder nếu chưa chạy
        placeholder_methods = [
            ("I00", "1-view (mốc)"),
            ("I01", "TTA HFlip K=2"),
            ("I02", "TTA 6-view (prob)"),
            ("I03", "TTA 6-view (logit avg)"),
            ("I04_256", "FixRes 256px"),
            ("I04_288", "FixRes 288px"),
            ("I04_320", "FixRes 320px"),
            ("I07", "Temperature Scaling"),
            ("I08", "Fused Conv-BN"),
        ]
        for exp_id, method in placeholder_methods:
            rows.append({
                "exp_id": exp_id, "method": method,
                "model_ckpt": "T00/seed0", "K_views": "?",
                "val_macro_f1": "?", "val_top1": "?", "ece_val": "?",
                "p50_ms": "?", "p95_ms": "?", "p99_ms": "?",
                "throughput_img_s": "?", "cost_rel_I00": "?", "notes": "",
            })
    return pd.DataFrame(rows)


def collect_final_results():
    """Sheet: Final – chung kết qua seed"""
    exp_ids = ["T00", "F01"]
    rows = []
    for exp_id in exp_ids:
        seed_results = []
        for seed in [0, 1, 2]:
            pred_path = ROOT / "predictions" / f"{exp_id}_seed{seed}_test.csv"
            if not pred_path.exists():
                continue
            r_res = read_result(exp_id, seed)
            val_f1 = round(r_res.get("val_macro_f1", 0), 4) if r_res else "?"
            try:
                pred = read_pred(str(pred_path))
                m    = compute_metrics(pred.y_true, pred.y_pred, pred.probs)
                seed_results.append({
                    "exp_id": exp_id, "seed": seed,
                    "val_macro_f1": val_f1,
                    "test_macro_f1": round(m["macro_f1"], 4),
                    "test_top1": round(m["top1"], 4),
                    "test_ece": round(m["ece"], 4),
                    "chinee_apple_recall": round(float(m["recall"][0]), 4),
                    "snake_weed_recall": round(float(m["recall"][7]), 4),
                })
                rows.append(seed_results[-1])
            except Exception as e:
                print(f"[warn] {pred_path}: {e}")
                rows.append({
                    "exp_id": exp_id, "seed": seed,
                    "val_macro_f1": val_f1, "test_macro_f1": "?",
                    "test_top1": "?", "test_ece": "?",
                    "chinee_apple_recall": "?", "snake_weed_recall": "?",
                })

        # Dòng tổng hợp mean ± std
        if len(seed_results) >= 2:
            f1s   = [r["test_macro_f1"] for r in seed_results if isinstance(r["test_macro_f1"], (int, float))]
            top1s = [r["test_top1"] for r in seed_results if isinstance(r["test_top1"], (int, float))]
            eces  = [r["test_ece"] for r in seed_results if isinstance(r["test_ece"], (int, float))]
            rows.append({
                "exp_id":   f"{exp_id}_MEAN±STD",
                "seed":     f"{len(seed_results)} seeds",
                "val_macro_f1": "-",
                "test_macro_f1": f"{np.mean(f1s):.4f} ± {np.std(f1s, ddof=1):.4f}",
                "test_top1": f"{np.mean(top1s):.4f} ± {np.std(top1s, ddof=1):.4f}",
                "test_ece": f"{np.mean(eces):.4f} ± {np.std(eces, ddof=1):.4f}" if eces else "?",
                "chinee_apple_recall": "-",
                "snake_weed_recall": "-",
            })


    if not rows:
        rows.append({
            "exp_id": "F01", "seed": "0,1,2",
            "val_macro_f1": "?", "test_macro_f1": "?",
            "test_top1": "?", "test_ece": "?",
            "chinee_apple_recall": "?", "snake_weed_recall": "?",
            "notes": "Chạy Bước 4 để điền"
        })
    return pd.DataFrame(rows)


def collect_per_class_results():
    """Sheet: PerClass"""
    rows = []
    # Thử đọc từ T00 (mốc) và F01 (chung kết)
    for (exp_id, seed, label) in [("T00", 0, "Mốc T00"), ("F01", 0, "Chung kết F01")]:
        test_pred = ROOT / "predictions" / f"{exp_id}_seed{seed}_test.csv"
        if not test_pred.exists():
            for cls_name in CLASS_NAMES:
                rows.append({
                    "class": cls_name, "label": label,
                    "n_test": "?", "precision": "?",
                    "recall": "?", "f1": "?",
                })
            continue
        pred = read_pred(str(test_pred))
        m = compute_metrics(pred.y_true, pred.y_pred, pred.probs)
        for i, cls_name in enumerate(CLASS_NAMES):
            rows.append({
                "class": cls_name, "label": label,
                "n_test": int(m["support"][i]),
                "precision": round(float(m["precision"][i]), 4),
                "recall": round(float(m["recall"][i]), 4),
                "f1": round(float(m["f1"][i]), 4),
            })

    # Thêm dòng trung bình qua các seed của F01
    f01_seeds = []
    for s in [0, 1, 2]:
        p = ROOT / "predictions" / f"F01_seed{s}_test.csv"
        if p.exists():
            try:
                pred = read_pred(str(p))
                f01_seeds.append(compute_metrics(pred.y_true, pred.y_pred, pred.probs))
            except Exception:
                pass
    if len(f01_seeds) >= 2:
        for i, cls_name in enumerate(CLASS_NAMES):
            prec_mean = np.mean([float(m["precision"][i]) for m in f01_seeds])
            rec_mean = np.mean([float(m["recall"][i]) for m in f01_seeds])
            f1_mean = np.mean([float(m["f1"][i]) for m in f01_seeds])
            rows.append({
                "class": cls_name, "label": f"F01 (mean {len(f01_seeds)} seeds)",
                "n_test": int(f01_seeds[0]["support"][i]),
                "precision": round(prec_mean, 4),
                "recall": round(rec_mean, 4),
                "f1": round(f1_mean, 4),
            })

    return pd.DataFrame(rows)



def collect_latency_results():
    """Sheet: Latency"""
    rows = []
    for lat_json in sorted(ROOT.glob("runs/*/seed*/latency_results.json")):
        data = load_json(lat_json)
        if isinstance(data, list):
            for rep in data:
                if isinstance(rep, dict):
                    rows.append(rep)

    if not rows:
        rows.append({
            "gpu": "RTX 3060 12GB",
            "dtype": "fp32", "batch": 1, "img_size": 224,
            "p50_ms": "?", "p95_ms": "?", "p99_ms": "?",
            "images_per_s": "?", "fused_bn": False,
            "notes": "Chạy run_inference.py để điền",
        })
    return pd.DataFrame(rows)


def build_summary(df_backbone: pd.DataFrame, df_training: pd.DataFrame,
                  df_inference: pd.DataFrame) -> pd.DataFrame:
    """Sheet: Summary – top 10 cấu hình theo Macro-F1 val."""
    rows = []
    for df in [df_backbone, df_training, df_inference]:
        if "val_macro_f1" in df.columns and "exp_id" in df.columns:
            for _, row in df.iterrows():
                rows.append({
                    "exp_id":       row.get("exp_id", "?"),
                    "description":  row.get("truc_thay_doi", row.get("method", row.get("backbone", "?"))),
                    "val_macro_f1": row.get("val_macro_f1", "?"),
                    "val_top1":     row.get("val_top1", "?"),
                    "ece_val":      row.get("ece_val", row.get("val_ece", "?")),
                    "latency_p50_ms": row.get("latency_b1_ms", row.get("p50_ms", "?")),
                })

    df = pd.DataFrame(rows).drop_duplicates("exp_id")
    # Cố gắng sắp xếp theo F1 (bỏ qua nếu cột là "?")
    try:
        df["_f1_num"] = pd.to_numeric(df["val_macro_f1"], errors="coerce")
        df = df.sort_values("_f1_num", ascending=False).drop(columns=["_f1_num"])
    except Exception:
        pass
    return df.head(20)


def write_xlsx(out_path: Path):
    print(f"\n[build_results] Đọc kết quả thí nghiệm...")
    df_backbone  = collect_backbone_results()
    df_training  = collect_training_results()
    df_inference = collect_inference_results()
    df_final     = collect_final_results()
    df_perclass  = collect_per_class_results()
    df_latency   = collect_latency_results()
    df_summary   = build_summary(df_backbone, df_training, df_inference)

    print(f"[build_results] Ghi → {out_path}")
    with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
        df_backbone.to_excel( writer, sheet_name="Backbones", index=False)
        df_training.to_excel( writer, sheet_name="Training",  index=False)
        df_inference.to_excel(writer, sheet_name="Inference", index=False)
        df_final.to_excel(    writer, sheet_name="Final",     index=False)
        df_perclass.to_excel( writer, sheet_name="PerClass",  index=False)
        df_latency.to_excel(  writer, sheet_name="Latency",   index=False)
        df_summary.to_excel(  writer, sheet_name="Summary",   index=False)

    print(f"[build_results] ✓ results.xlsx tạo thành công")
    print(f"  Các ô '?' cần điền số liệu thật sau khi chạy thí nghiệm.")


if __name__ == "__main__":
    write_xlsx(ROOT / "results.xlsx")
