"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict]
    count_params(model) -> float (triệu)
    count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

# Gợi ý backbone (GUIDE.md mục 2.1). GHI RÕ tag trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50":       "resnet50",                          # mốc ResNet
    "resnext50":      "resnext50_32x4d",                   # ResNeXt
    "convnext_tiny":  "convnext_tiny",                     # ConvNeXt hiện đại
    "deit_small":     "deit_small_patch16_224",            # ViT/DeiT transformer
    "swin_tiny":      "swin_tiny_patch4_window7_224",      # Swin transformer
    "efficientnet_b0": "efficientnet_b0",                  # nhẹ
    "mobilenetv3":    "mobilenetv3_large_100",             # nhẹ nhất
}


# ---------------------------------------------------------------------------
def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp.

    `init`:
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ (mặc định)
    """
    if init == "scratch":
        pretrained_flag = False
    else:
        pretrained_flag = pretrained  # True

    # Resolve tên alias
    model_name = SUGGESTED_BACKBONES.get(name, name)

    model = timm.create_model(
        model_name,
        pretrained=pretrained_flag,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    # Ghi lại tag trọng số thực sự được tải
    if hasattr(model, "pretrained_cfg"):
        tag = model.pretrained_cfg.get("tag", "unknown")
        source = model.pretrained_cfg.get("url", model.pretrained_cfg.get("hf_hub_id", "?"))
        print(f"[build_model] {model_name} | init={init} | tag={tag}")
        print(f"  pretrained_cfg url/hf_hub: {str(source)[:80]}")
    else:
        print(f"[build_model] {model_name} | init={init}")

    if init == "frozen":
        freeze_backbone(model)

    return model


# ---------------------------------------------------------------------------
def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ head (classifier).

    BatchNorm trong backbone đóng băng phải ở chế độ eval() – xem ghi chú
    trong train.py: sau model.train(), cần gọi lại _set_bn_eval(model).
    """
    # Lấy tham số của head
    try:
        head = model.get_classifier()
    except AttributeError:
        # fallback: tìm head theo tên thông dụng
        head = getattr(model, "fc", None) or getattr(model, "head", None) or \
               getattr(model, "classifier", None)

    head_ids = {id(p) for p in (head.parameters() if head is not None else [])}

    frozen = 0
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad_(False)
            frozen += 1

    print(f"[freeze_backbone] Đóng băng {frozen} tensor (head vẫn train)")


def _set_bn_eval(model: nn.Module) -> None:
    """Đặt BatchNorm2d/1d trong phần backbone đóng băng ở chế độ eval.

    Gọi sau model.train() trong train loop khi init='frozen'.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            # Nếu không có tham số trainable thì đặt eval
            if all(not p.requires_grad for p in module.parameters()):
                module.eval()


# ---------------------------------------------------------------------------
def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1 : lr = lr_backbone, weight_decay = weight_decay
    - norm và bias (ndim ≤ 1): lr = lr_backbone, weight_decay = 0
    - head mới : lr = lr_head (thường gấp 10×), weight_decay = weight_decay

    Bỏ qua tham số requires_grad == False.
    """
    try:
        head = model.get_classifier()
    except AttributeError:
        head = getattr(model, "fc", None) or getattr(model, "head", None) or \
               getattr(model, "classifier", None)

    head_ids = {id(p) for p in (head.parameters() if head is not None else [])}

    backbone_decay, backbone_no_decay, head_params = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in head_ids:
            head_params.append(param)
        elif param.ndim <= 1 or name.endswith(".bias"):
            # norm layers, bias: no weight decay
            backbone_no_decay.append(param)
        else:
            backbone_decay.append(param)

    groups = []
    if backbone_decay:
        groups.append({"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if backbone_no_decay:
        groups.append({"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})

    n_params = sum(p.numel() for g in groups for p in g["params"])
    print(f"[param_groups] {len(groups)} nhóm, {n_params/1e6:.2f}M tham số trainable")
    return groups


# ---------------------------------------------------------------------------
def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size.

    Dùng thư viện fvcore (ưu tiên) hoặc thop làm fallback.
    Số có thể lệch vài phần trăm giữa các công cụ.
    """
    device = next(model.parameters()).device
    dummy  = torch.zeros(1, 3, img_size, img_size).to(device)
    model.eval()

    # Thử fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        macs  = flops.total() / 2  # MACs = FLOPs / 2
        return macs / 1e9
    except ImportError:
        pass

    # Thử thop
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return macs / 1e9
    except ImportError:
        pass

    # Thử ptflops
    try:
        from ptflops import get_model_complexity_info
        macs_str, _ = get_model_complexity_info(
            model, (3, img_size, img_size), as_strings=False, print_per_layer_stat=False, verbose=False
        )
        return macs_str / 1e9
    except ImportError:
        pass

    print("[count_gmacs] Không tìm thấy fvcore/thop/ptflops. Trả về -1.")
    return -1.0
