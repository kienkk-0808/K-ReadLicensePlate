"""Anchor points + flatten/decode dùng chung giữa model, assigner và loss."""

from typing import List, Tuple

import torch

from models.scrfd_mbf import STRIDES


def generate_points(img_size: int, strides: Tuple[int, ...] = STRIDES, device=None):
    """-> points(total,2), strides_per_point(total,), num_points_per_level."""
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
        pts = torch.stack([px, py], dim=-1)

        all_points.append(pts)
        all_strides.append(torch.full((size * size,), float(stride), device=device))
        num_points_per_level.append(size * size)

    points = torch.cat(all_points, dim=0)
    strides_per_point = torch.cat(all_strides, dim=0)
    return points, strides_per_point, num_points_per_level


def flatten_head_outputs(outputs: List[Tuple[torch.Tensor, torch.Tensor]]):
    """list[(cls,bbox) per level, (B,C,H,W)] -> (cls_logits, bbox_dist) gộp (B,N,C)."""
    cls_list, bbox_list = [], []
    for cls_score, bbox_dist in outputs:
        b, num_classes, h, w = cls_score.shape
        cls_list.append(cls_score.permute(0, 2, 3, 1).reshape(b, h * w, num_classes))
        bbox_list.append(bbox_dist.permute(0, 2, 3, 1).reshape(b, h * w, 4))

    cls_logits = torch.cat(cls_list, dim=1)
    bbox_dist = torch.cat(bbox_list, dim=1)
    return cls_logits, bbox_dist


def decode_points(points: torch.Tensor, bbox_dist: torch.Tensor):
    """points(N,2) + bbox_dist(...,N,4) -> boxes xyxy."""
    px, py = points[..., 0], points[..., 1]

    x1 = px - bbox_dist[..., 0]
    y1 = py - bbox_dist[..., 1]
    x2 = px + bbox_dist[..., 2]
    y2 = py + bbox_dist[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)
