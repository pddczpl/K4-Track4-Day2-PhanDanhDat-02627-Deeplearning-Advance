"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
T khớp trên VAL rồi áp dụng sang test.

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model, hoặc None.
    Trả về (filenames, y_true, logits) dưới dạng numpy.
    """
    model.eval()
    all_filenames, all_ytrue, all_logits = [], [], []

    for images, labels, fnames in loader:
        images = images.to(device, non_blocking=True)
        if view is not None:
            images = view(images)
        logits = model(images).float().cpu().numpy()
        all_filenames.extend(fnames)
        all_ytrue.append(labels.numpy())
        all_logits.append(logits)

    y_true  = np.concatenate(all_ytrue)
    logits  = np.concatenate(all_logits)
    return all_filenames, y_true, logits


# ---------------------------------------------------------------------------
# ---- Các hàm view cho TTA ----

def view_identity(x: torch.Tensor) -> torch.Tensor:
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch (N, C, H, W) – slide trang 75."""
    return torch.flip(x, dims=[3])


def view_vflip(x: torch.Tensor) -> torch.Tensor:
    """Lật dọc (ảnh cỏ dại → có thể hợp lệ do góc chụp từ trên)."""
    return torch.flip(x, dims=[2])


def view_rotate90(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=1, dims=[2, 3])


def view_rotate180(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=2, dims=[2, 3])


def view_rotate270(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=3, dims=[2, 3])


def views_multicrop(x: torch.Tensor, crop: int = 224):
    """5-crop: 4 góc + trung tâm (và có thể thêm lật ngang). Trả về list batch."""
    N, C, H, W = x.shape
    crops = []
    # center
    ch = (H - crop) // 2
    cw = (W - crop) // 2
    positions = [
        (0, 0),                      # top-left
        (0, W - crop),               # top-right
        (H - crop, 0),               # bottom-left
        (H - crop, W - crop),        # bottom-right
        (ch, cw),                    # center
    ]
    for r, c in positions:
        crops.append(x[:, :, r:r+crop, c:c+crop])
    return crops


def views_multiscale(x: torch.Tensor, sizes: list = None):
    """Resize batch về từng kích thước trong `sizes`. Trả về list batch.

    Lưu ý: CNN với global pooling chấp nhận kích thước khác; ViT cần cẩn thận.
    """
    if sizes is None:
        sizes = [224, 256, 288, 320]
    views = []
    for s in sizes:
        resized = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        views.append(resized)
    return views


# ---------------------------------------------------------------------------
def aggregate_views(logits_per_view: list, space: str = "prob") -> np.ndarray:
    """Gộp K lượt TTA thành một dự đoán.

    - space="prob"  : trung bình softmax của từng view  (ổn định hơn)
    - space="logit" : trung bình logit rồi softmax      (giảm peak probability)
    """
    stack = np.stack(logits_per_view, axis=0)  # (K, N, C)
    if space == "prob":
        probs_per_view = _softmax(stack)        # (K, N, C)
        probs = probs_per_view.mean(axis=0)     # (N, C)
    elif space == "logit":
        avg_logit = stack.mean(axis=0)          # (N, C)
        probs = _softmax(avg_logit)             # (N, C)
    else:
        raise ValueError(f"space='{space}' không hợp lệ. Chọn: prob, logit")
    return probs


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------------------
def ensemble_probs(list_of_probs: list) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình.

    Mỗi phần tử là mảng probs (N, 9). Phải cùng thứ tự file.
    """
    stack = np.stack(list_of_probs, axis=0)  # (M, N, C)
    return stack.mean(axis=0)                # (N, C)


# ---------------------------------------------------------------------------
def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T).

    Dùng LBFGS trên log(T) để ổn định số. KHÔNG khớp T trên test.
    """
    import torch
    import torch.optim as optim

    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    labels_t = torch.tensor(val_labels, dtype=torch.long)

    # Khởi tạo log_T = 0 (tương đương T = 1)
    log_T = torch.nn.Parameter(torch.zeros(1))
    opt   = optim.LBFGS([log_T], max_iter=50, tolerance_grad=1e-9, tolerance_change=1e-9,
                         history_size=10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        T    = log_T.exp()
        loss = F.cross_entropy(logits_t / T, labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    T_val = float(log_T.exp().item())
    print(f"[fit_temperature] T = {T_val:.6f}")
    return T_val


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về softmax(logits / T)."""
    return _softmax(logits / T)


# ---------------------------------------------------------------------------
def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm2d vào tích chập liền trước (chính xác lúc suy luận).

        w' = gamma * w / sqrt(var + eps)
        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    Không áp dụng với ViT/Swin/ConvNeXt (dùng LayerNorm thay BN).
    Sau khi gộp: đầu ra trước/sau lệch nhau < 1e-5.
    """
    import copy
    model = copy.deepcopy(model)
    model.eval()

    fused_count = 0

    def _fuse(parent: nn.Module):
        nonlocal fused_count
        children = list(parent.named_children())
        i = 0
        while i < len(children):
            name, module = children[i]
            if (i + 1 < len(children) and
                    isinstance(module, nn.Conv2d) and
                    isinstance(children[i + 1][1], nn.BatchNorm2d)):
                bn = children[i + 1][1]
                # Tính trọng số mới
                scale = bn.weight / (bn.running_var + bn.eps).sqrt()
                conv_w = module.weight * scale.reshape(-1, 1, 1, 1)
                bias   = (module.bias if module.bias is not None
                          else torch.zeros(module.out_channels, device=module.weight.device))
                conv_b = bn.bias + scale * (bias - bn.running_mean)

                # Thay conv mới có bias
                new_conv = nn.Conv2d(
                    module.in_channels, module.out_channels, module.kernel_size,
                    stride=module.stride, padding=module.padding, dilation=module.dilation,
                    groups=module.groups, bias=True,
                )
                new_conv.weight = nn.Parameter(conv_w)
                new_conv.bias   = nn.Parameter(conv_b)
                setattr(parent, name, new_conv)

                # Thay BN bằng Identity
                setattr(parent, children[i + 1][0], nn.Identity())
                fused_count += 1
                i += 2
            else:
                _fuse(module)
                i += 1

    _fuse(model)

    if fused_count == 0:
        print("[fuse_conv_bn] Không tìm thấy cặp Conv-BN. Kiến trúc này có thể dùng LayerNorm (ViT/Swin/ConvNeXt).")
    else:
        print(f"[fuse_conv_bn] Gộp {fused_count} cặp Conv-BN.")

    return model


# ---------------------------------------------------------------------------
def verify_fuse_conv_bn(model: nn.Module, img_size: int = 224) -> float:
    """Kiểm tra sai số tối đa trước/sau gộp BN. Trả về max abs diff."""
    model = model.cpu().eval()
    dummy = torch.randn(2, 3, img_size, img_size)
    with torch.no_grad():
        out_orig = model(dummy)
    fused = fuse_conv_bn(model)
    with torch.no_grad():
        out_fused = fused(dummy)
    diff = (out_orig - out_fused).abs().max().item()
    print(f"[verify_fuse_conv_bn] Max abs diff = {diff:.2e}")
    return diff
