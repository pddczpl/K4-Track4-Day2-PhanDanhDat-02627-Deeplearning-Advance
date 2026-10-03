"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Giao diện:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import v2  # noqa: F401  (fallback below)

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv.

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_dir = Path(labels_dir)
    train_df = pd.read_csv(labels_dir / f"train_subset{fold}.csv")
    val_df   = pd.read_csv(labels_dir / f"val_subset{fold}.csv")
    test_df  = pd.read_csv(labels_dir / f"test_subset{fold}.csv")
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1).

    Trả về dict với số liệu để dán vào báo cáo.
    """
    images_dir = Path(images_dir)
    info = {"n": {}, "per_class": {}, "overlap": {}, "files_exist": True}

    # 1. Số ảnh mỗi tập
    info["n"]["train"] = len(train_df)
    info["n"]["val"]   = len(val_df)
    info["n"]["test"]  = len(test_df)
    total = info["n"]["train"] + info["n"]["val"] + info["n"]["test"]
    info["n"]["total"] = total

    print(f"[check_split] train={info['n']['train']}, val={info['n']['val']}, "
          f"test={info['n']['test']}, total={total}")

    # Tỉ lệ
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        pct = len(df) / total * 100
        print(f"  {split}: {len(df)} ảnh ({pct:.1f}%)")

    # 2. Kiểm tra giao rỗng
    train_files = set(train_df["Filename"])
    val_files   = set(val_df["Filename"])
    test_files  = set(test_df["Filename"])

    tv_overlap = train_files & val_files
    tt_overlap = train_files & test_files
    vt_overlap = val_files   & test_files

    info["overlap"]["train_val"]  = len(tv_overlap)
    info["overlap"]["train_test"] = len(tt_overlap)
    info["overlap"]["val_test"]   = len(vt_overlap)

    assert len(tv_overlap)  == 0, f"train∩val không rỗng: {len(tv_overlap)} ảnh"
    assert len(tt_overlap)  == 0, f"train∩test không rỗng: {len(tt_overlap)} ảnh"
    assert len(vt_overlap)  == 0, f"val∩test không rỗng: {len(vt_overlap)} ảnh"
    print(f"[check_split] Giao giữa các tập: train∩val=0, train∩test=0, val∩test=0 ✓")

    # 3. Hợp = 17.509
    all_files = train_files | val_files | test_files
    assert len(all_files) == 17_509, f"Hợp ba tập = {len(all_files)}, kỳ vọng 17.509"
    print(f"[check_split] Hợp ba tập = {len(all_files)} ✓")

    # 4. Mọi file tồn tại
    missing = []
    for df in [train_df, val_df, test_df]:
        for fname in df["Filename"]:
            if not (images_dir / fname).exists():
                missing.append(fname)

    if missing:
        info["files_exist"] = False
        print(f"[check_split] ⚠ Thiếu {len(missing)} file ảnh trong {images_dir}")
        print(f"  Ví dụ: {missing[:5]}")
    else:
        print(f"[check_split] Mọi file ảnh tồn tại trong {images_dir} ✓")

    # 5. Số ảnh mỗi lớp
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        per_class = df.groupby("Label").size().to_dict()
        info["per_class"][split] = per_class

    return info


# ---------------------------------------------------------------------------
def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation.

    Các giá trị `aug`:
      - "basic"   : RandomResizedCrop + HorizontalFlip (công thức nền)
      - "color"   : basic + ColorJitter + GaussianBlur
      - "trivial" : TrivialAugmentWide (tự chọn phép biến đổi)
      - "randaug" : RandAugment (N=2, M=9)

    Train: augmentation ngẫu nhiên + ToTensor + Normalize.
    Val/test: Resize(256) + CenterCrop(img_size) + ToTensor + Normalize. KHÔNG augment.
    """
    mean = IMAGENET_MEAN
    std  = IMAGENET_STD

    if not train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # Augmentation theo từng mức
    if aug == "basic":
        aug_list = [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
    elif aug == "color":
        aug_list = [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ]
    elif aug == "trivial":
        aug_list = [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
        ]
    elif aug == "randaug":
        aug_list = [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
        ]
    else:
        raise ValueError(f"aug='{aug}' không hợp lệ. Chọn: basic, color, trivial, randaug")

    return transforms.Compose([
        *aug_list,
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ---------------------------------------------------------------------------
class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (ảnh đã transform, nhãn int, tên file str).
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None,
                 preload: bool = False):
        self.df = df.reset_index(drop=True)
        self.filenames = self.df["Filename"].astype(str).tolist()
        self.labels = self.df["Label"].astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.transform  = transform
        self.preload    = preload
        self._cache: dict = {}

        if preload:
            print(f"[DeepWeedsDataset] Preloading {len(self.filenames)} ảnh vào RAM...")
            for i, fname in enumerate(self.filenames):
                img = Image.open(self.images_dir / fname).convert("RGB")
                self._cache[i] = img
            print("[DeepWeedsDataset] Preload xong.")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        fname = self.filenames[i]
        label = self.labels[i]

        if self.preload and i in self._cache:
            img = self._cache[i]
        else:
            img = Image.open(self.images_dir / fname).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label, fname



# ---------------------------------------------------------------------------
def _worker_init_fn(worker_id: int):
    """Cố định seed cho mỗi DataLoader worker để tái lập."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                preload: bool = False):
    """Tạo DataLoader.

    - train=True: shuffle (hoặc dùng sampler); train=False: không shuffle, giữ thứ tự
    - sampler=None | "balanced": WeightedRandomSampler với trọng số 1/(số ảnh lớp)
    - drop_last=True khi train nếu batch cuối quá nhỏ
    - pin_memory=True để nạp lên GPU nhanh hơn
    """
    dataset = DeepWeedsDataset(df, images_dir, transform, preload=preload)

    if train:
        if sampler == "balanced":
            counts = df["Label"].value_counts().sort_index()
            class_weights = 1.0 / counts.values.astype(float)
            sample_weights = class_weights[df["Label"].values]
            sampler_obj = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(df),
                replacement=True,
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler_obj,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=_worker_init_fn,
                persistent_workers=num_workers > 0,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=_worker_init_fn,
                persistent_workers=num_workers > 0,
            )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=_worker_init_fn,
            persistent_workers=num_workers > 0,
        )
