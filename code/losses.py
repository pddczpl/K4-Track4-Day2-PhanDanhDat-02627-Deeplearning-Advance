"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Giao diện:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`.

    kind:
      - "ce"          : CrossEntropyLoss (mặc định)
      - "ls"          : Label smoothing CE
      - "focal"       : Focal Loss
      - "ce_weighted" : CrossEntropyLoss có trọng số lớp (kw phải có weight=tensor)
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        smoothing = kw.get("smoothing", 0.1)
        return LabelSmoothingCE(smoothing=smoothing)
    elif kind == "focal":
        gamma = kw.get("gamma", 2.0)
        alpha = kw.get("alpha", None)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get("weight", None)
        smoothing = kw.get("smoothing", 0.0)
        if smoothing > 0:
            return LabelSmoothingCE(smoothing=smoothing, weight=weight)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"kind='{kind}' không hợp lệ. Chọn: ce, ls, focal, ce_weighted")


# ---------------------------------------------------------------------------
class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K.

    eps = 0 cho đúng CE tiêu chuẩn.
    """

    def __init__(self, smoothing: float = 0.1, weight=None):
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "smoothing phải trong [0, 1)"
        self.smoothing = smoothing
        self.register_buffer("weight", weight)  # weight có thể là None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.smoothing == 0.0:
            return F.cross_entropy(logits, target, weight=self.weight)
        # Dùng built-in của PyTorch (đã tối ưu)
        return F.cross_entropy(logits, target, weight=self.weight, label_smoothing=self.smoothing)


# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal Loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    gamma = 0 → CrossEntropy thông thường (đã kiểm tra).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # log_softmax + nll để ổn định số
        log_probs = F.log_softmax(logits, dim=1)          # (N, C)
        ce = F.nll_loss(log_probs, target, reduction="none")  # (N,)

        if self.gamma == 0.0:
            # gamma=0 → CE thông thường
            if self.alpha is not None:
                ce = self.alpha[target] * ce
            return ce.mean()

        # p_t = exp(log_p_t)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # (N,)
        pt = log_pt.exp()

        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            focal_weight = focal_weight * self.alpha[target]

        loss = focal_weight * ce
        return loss.mean()


# ---------------------------------------------------------------------------
def _verify_focal():
    """Unit test: FocalLoss(gamma=0) ≡ CrossEntropyLoss (sai số < 1e-6)."""
    torch.manual_seed(0)
    logits = torch.randn(16, 9)
    target = torch.randint(0, 9, (16,))
    ce_val  = nn.CrossEntropyLoss()(logits, target)
    fl_val  = FocalLoss(gamma=0.0)(logits, target)
    diff = abs(float(ce_val) - float(fl_val))
    assert diff < 1e-6, f"FocalLoss(γ=0) ≠ CE: diff={diff:.2e}"
    print(f"[_verify_focal] OK – diff={diff:.2e}")


# ---------------------------------------------------------------------------
def class_weights(counts, beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0 : w_c = 1 / n_c, chuẩn hoá về trung bình 1
    - beta > 0 : class-balanced: w_c = (1 - beta) / (1 - beta^n_c)
    """
    counts = np.asarray(counts, dtype=np.float64)
    if beta == 0.0:
        w = 1.0 / counts
        w = w / w.mean()
    else:
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
        w = w / w.sum() * len(counts)  # chuẩn hoá tổng = C
    return torch.tensor(w, dtype=torch.float32)


# ---------------------------------------------------------------------------
def mix_batch(x: torch.Tensor, y: torch.Tensor,
              alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha)
    - mode="mixup"  : x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix" : cắt hộp chữ nhật, điều chỉnh lam theo diện tích thực
    Trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm]
    """
    assert mode in ("mixup", "cutmix"), f"mode='{mode}' không hợp lệ"
    batch_size = x.size(0)

    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(batch_size, device=x.device)

    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    else:  # cutmix
        x_mix, lam = _cutmix(x, perm, lam)

    return x_mix, (y, y[perm], lam)


def _cutmix(x: torch.Tensor, perm: torch.Tensor, lam: float):
    """Tạo ảnh CutMix; trả về (x_mix, lam_điều_chỉnh)."""
    H, W = x.shape[-2], x.shape[-1]
    # Diện tích hộp theo lam
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    # Tâm ngẫu nhiên
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]

    # lam điều chỉnh theo diện tích thực
    lam_adj = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    return x_mix, lam_adj


def mixed_loss(criterion, logits: torch.Tensor, targets) -> torch.Tensor:
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    targets = (y_a, y_b, lam) từ mix_batch.
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _verify_focal()
    print("losses.py: tất cả unit test passed.")
