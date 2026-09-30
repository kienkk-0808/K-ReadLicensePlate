"""Anchor points + flatten/decode dùng chung giữa model, assigner và loss."""

from typing import List, Tuple

import torch

from models.scrfd_mbf import STRIDES, dfl_to_distance


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
    """list[(cls,bbox_logits) per level, (B,C,H,W)] -> (cls_logits, bbox_logits) gộp (B,N,C)."""
    cls_list, bbox_list = [], []
    for cls_score, bbox_logits in outputs:
        b, num_classes, h, w = cls_score.shape
        bbox_c = bbox_logits.shape[1]
        cls_list.append(cls_score.permute(0, 2, 3, 1).reshape(b, h * w, num_classes))
        bbox_list.append(bbox_logits.permute(0, 2, 3, 1).reshape(b, h * w, bbox_c))

    cls_logits = torch.cat(cls_list, dim=1)
    bbox_logits = torch.cat(bbox_list, dim=1)
    return cls_logits, bbox_logits


def decode_points(points: torch.Tensor, strides_per_point: torch.Tensor,
                   bbox_logits: torch.Tensor, reg_max: int):
    """points(N,2), strides_per_point(N,), bbox_logits(...,N,4*(reg_max+1)) -> boxes xyxy."""
    dist_units = dfl_to_distance(bbox_logits, reg_max)
    dist_pixels = dist_units * strides_per_point.unsqueeze(-1)

    px, py = points[..., 0], points[..., 1]
    x1 = px - dist_pixels[..., 0]
    y1 = py - dist_pixels[..., 1]
    x2 = px + dist_pixels[..., 2]
    y2 = py + dist_pixels[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)
