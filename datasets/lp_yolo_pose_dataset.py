"""Dataset loader YOLO-Pose (Roboflow) cho SCRFD_MBF.

Format label: class cx cy w h  x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4 (normalized).
Thứ tự 4 keypoint không cố định TL/TR/BR/BL -> chuẩn hoá bằng sort_corners().
Gộp mọi class gốc (plate-1-line/plate-2-line) về 1 class "plate".
"""

import os
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.scrfd_mbf import sort_corners, NUM_KPS  # noqa: E402
from datasets.augmentations import train_augment  # noqa: E402

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


def parse_label_file(label_path: Path, num_kps: int = NUM_KPS) -> List[dict]:
    """-> list[{class_id, cx, cy, w, h, kps: [(x,y,v), ...]}] (normalized 0-1)."""
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
            kps_flat = values[5:5 + num_kps * 3]
            kps = [
                (kps_flat[i * 3], kps_flat[i * 3 + 1], kps_flat[i * 3 + 2])
                for i in range(num_kps)
            ]
            objects.append({
                "class_id": class_id, "cx": cx, "cy": cy, "w": w, "h": h, "kps": kps,
            })
    return objects


class LicensePlateYoloPoseDataset(Dataset):
    def __init__(self, root: str, split: str = "train", img_size: int = 640,
                 num_kps: int = NUM_KPS, augment: bool = None,
                 augment_hyp: dict = None):
        self.root = Path(root)
        self.img_dir = self.root / split / "images"
        self.label_dir = self.root / split / "labels"
        self.img_size = img_size
        self.num_kps = num_kps
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

        objects = parse_label_file(label_path, self.num_kps)
        num_obj = len(objects)

        if num_obj:
            cx = np.array([o["cx"] for o in objects]) * orig_w
            cy = np.array([o["cy"] for o in objects]) * orig_h
            bw = np.array([o["w"] for o in objects]) * orig_w
            bh = np.array([o["h"] for o in objects]) * orig_h
            boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
            boxes = boxes.astype(np.float32)

            labels = np.zeros(num_obj, dtype=np.int64)

            kps = np.array([
                [(kx * orig_w, ky * orig_h) for (kx, ky, kv) in o["kps"]]
                for o in objects
            ], dtype=np.float32)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            kps = np.zeros((0, self.num_kps, 2), dtype=np.float32)

        if self.augment:
            img, boxes, kps, keep = train_augment(img, boxes, kps, **self.augment_hyp)
            if labels.shape[0]:
                labels = labels[keep]

        canvas, scale, (pad_x, pad_y) = letterbox(img, self.img_size)

        if boxes.shape[0]:
            boxes = boxes * scale
            boxes[:, [0, 2]] += pad_x
            boxes[:, [1, 3]] += pad_y

            kps = kps * scale
            kps[..., 0] += pad_x
            kps[..., 1] += pad_y

        img_tensor = torch.from_numpy(np.ascontiguousarray(canvas)).permute(2, 0, 1).float() / 255.0

        if boxes.shape[0]:
            boxes_t = torch.from_numpy(boxes)
            labels_t = torch.from_numpy(labels)
            kps_t = sort_corners(torch.from_numpy(kps))
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)
            kps_t = torch.zeros((0, self.num_kps, 2), dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "kps": kps_t,
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
    print("Kps:", target["kps"].shape)

    ds_val = LicensePlateYoloPoseDataset(root="dataset", split="valid", img_size=640)
    print(f"So anh valid: {len(ds_val)} | augment: {ds_val.augment}")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    imgs, targets = next(iter(loader))
    print("Batch imgs:", imgs.shape, "| num targets:", len(targets))
