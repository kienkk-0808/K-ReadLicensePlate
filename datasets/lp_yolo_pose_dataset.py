"""Dataset loader (YOLO object detection chuẩn) cho SCRFD_MBF.

Format label: class cx cy w h (normalized 0-1). Gộp mọi class gốc về 1 class "plate".
"""

import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.augmentations import train_augment

CLASS_NAMES = ["plate"]


def letterbox(img: np.ndarray, new_size: int = 640, pad_value: int = 114):
    """-> canvas vuông (new_size), scale, (pad_x, pad_y)."""
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_size, new_size, 3), pad_value, dtype=img.dtype)
    pad_x = (new_size - new_w) // 2
    pad_y = (new_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def parse_label_file(label_path: Path) -> List[dict]:
    """-> list[{class_id, cx, cy, w, h}] (normalized 0-1)."""
    objects = []
    if not label_path.exists():
        return objects

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            values = list(map(float, parts))
            class_id = int(values[0])
            cx, cy, w, h = values[1:5]
            objects.append({"class_id": class_id, "cx": cx, "cy": cy, "w": w, "h": h})
    return objects


class LicensePlateYoloPoseDataset(Dataset):
    def __init__(self, root: str, split: str = "train", img_size: int = 640,
                 augment: bool = None, augment_hyp: dict = None):
        self.root = Path(root)
        self.img_dir = self.root / split / "images"
        self.label_dir = self.root / split / "labels"
        self.img_size = img_size
        self.split = split
        self.augment = (split == "train") if augment is None else augment
        self.augment_hyp = augment_hyp or {}

        self.img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = self.img_dir / img_name
        label_path = self.label_dir / (Path(img_name).stem + ".txt")

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        objects = parse_label_file(label_path)
        num_obj = len(objects)

        if num_obj:
            cx = np.array([o["cx"] for o in objects]) * orig_w
            cy = np.array([o["cy"] for o in objects]) * orig_h
            bw = np.array([o["w"] for o in objects]) * orig_w
            bh = np.array([o["h"] for o in objects]) * orig_h
            boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
            boxes = boxes.astype(np.float32)
            labels = np.zeros(num_obj, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        if self.augment:
            img, boxes, keep = train_augment(img, boxes, **self.augment_hyp)
            if labels.shape[0]:
                labels = labels[keep]

        canvas, scale, (pad_x, pad_y) = letterbox(img, self.img_size)

        if boxes.shape[0]:
            boxes = boxes * scale
            boxes[:, [0, 2]] += pad_x
            boxes[:, [1, 3]] += pad_y

        img_tensor = torch.from_numpy(np.ascontiguousarray(canvas)).permute(2, 0, 1).float() / 255.0

        if boxes.shape[0]:
            boxes_t = torch.from_numpy(boxes)
            labels_t = torch.from_numpy(labels)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "img_size": self.img_size,
            "orig_size": (orig_w, orig_h),
            "scale": scale,
            "pad": (pad_x, pad_y),
            "file_name": img_name,
        }
        return img_tensor, target


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets


if __name__ == "__main__":
    ds = LicensePlateYoloPoseDataset(root="dataset", split="train", img_size=640)
    print(f"So anh train: {len(ds)} | augment: {ds.augment}")

    img, target = ds[0]
    print("Image tensor:", img.shape)
    print("Boxes:", target["boxes"].shape)
    print("Labels:", target["labels"], [CLASS_NAMES[i] for i in target["labels"].tolist()])

    ds_val = LicensePlateYoloPoseDataset(root="dataset", split="valid", img_size=640)
    print(f"So anh valid: {len(ds_val)} | augment: {ds_val.augment}")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    imgs, targets = next(iter(loader))
    print("Batch imgs:", imgs.shape, "| num targets:", len(targets))
