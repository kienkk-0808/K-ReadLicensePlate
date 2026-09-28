"""Tiện ích dùng chung giữa model, assigner và loss: sinh anchor point theo lưới
và "làm phẳng" (flatten) output đa cấp (P3/P4/P5) về 1 tensor duy nhất mỗi ảnh.
"""

from typing import List, Tuple

import torch

from models.scrfd_mbf import STRIDES, NUM_KPS


def generate_points(img_size: int, strides: Tuple[int, ...] = STRIDES, device=None):
    """Sinh toạ độ tâm anchor point cho toàn bộ level, theo đúng công thức decode()
    trong models/scrfd_mbf.py: px = (j+0.5)*stride, py = (i+0.5)*stride.

    Trả về:
        points: (total_points, 2) — toạ độ (px, py) pixel trong ảnh img_size x img_size
        strides_per_point: (total_points,) — stride tương ứng của từng điểm
        num_points_per_level: List[int]
    """
    all_points = []
    all_strides = []
    num_points_per_level = []

    for stride in strides:
        size = img_size // stride
        yv, xv = torch.meshgrid(
            torch.arange(size, device=device),
            torch.arange(size, device=device),
            indexing="ij",
        )
        px = (xv.reshape(-1).float() + 0.5) * stride
        py = (yv.reshape(-1).float() + 0.5) * stride
        pts = torch.stack([px, py], dim=-1)  # (size*size, 2)

        all_points.append(pts)
        all_strides.append(torch.full((size * size,), float(stride), device=device))
        num_points_per_level.append(size * size)

    points = torch.cat(all_points, dim=0)
    strides_per_point = torch.cat(all_strides, dim=0)
    return points, strides_per_point, num_points_per_level


def flatten_head_outputs(outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """outputs: list[(cls_score, bbox_dist, kps_offset)] mỗi phần tử shape (B,C,H,W),
    theo thứ tự P3,P4,P5 (giống forward() của SCRFD_MBF).

    Trả về (mỗi tensor đã gộp theo thứ tự level P3->P4->P5, giống generate_points):
        cls_logits: (B, total_points, num_classes)
        bbox_dist:  (B, total_points, 4)
        kps_offset: (B, total_points, NUM_KPS*2)
    """
    cls_list, bbox_list, kps_list = [], [], []
    for cls_score, bbox_dist, kps_offset in outputs:
        b, num_classes, h, w = cls_score.shape
        cls_list.append(cls_score.permute(0, 2, 3, 1).reshape(b, h * w, num_classes))
        bbox_list.append(bbox_dist.permute(0, 2, 3, 1).reshape(b, h * w, 4))
        kps_list.append(kps_offset.permute(0, 2, 3, 1).reshape(b, h * w, NUM_KPS * 2))

    cls_logits = torch.cat(cls_list, dim=1)
    bbox_dist = torch.cat(bbox_list, dim=1)
    kps_offset = torch.cat(kps_list, dim=1)
    return cls_logits, bbox_dist, kps_offset


def decode_points(points: torch.Tensor, bbox_dist: torch.Tensor, kps_offset: torch.Tensor):
    """Decode (l,t,r,b) và (dx,dy)x4 tại từng điểm -> box xyxy + kps tuyệt đối.

    points: (N, 2) | bbox_dist: (..., N, 4) | kps_offset: (..., N, NUM_KPS*2)
    Hỗ trợ broadcast theo batch (bbox_dist/kps_offset có thêm chiều batch ở đầu).
    """
    px, py = points[..., 0], points[..., 1]

    x1 = px - bbox_dist[..., 0]
    y1 = py - bbox_dist[..., 1]
    x2 = px + bbox_dist[..., 2]
    y2 = py + bbox_dist[..., 3]
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    kps = kps_offset.reshape(*kps_offset.shape[:-1], NUM_KPS, 2)
    kps = kps + torch.stack([px, py], dim=-1).unsqueeze(-2)
    return boxes, kps
