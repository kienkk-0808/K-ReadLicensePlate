"""ATSS assigner cho anchor-free point-based detector (SCRFD-style)."""

from typing import List

import torch


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter + 1e-7
    return inter / union


@torch.no_grad()
def atss_assign(
    points: torch.Tensor,
    strides_per_point: torch.Tensor,
    num_points_per_level: List[int],
    gt_boxes: torch.Tensor,
    topk: int = 9,
    anchor_scale: float = 8.0,
) -> torch.Tensor:
    """-> assigned_gt_inds (N,): 0 = background, k>0 = gt index (k-1)."""
    num_points = points.shape[0]
    num_gt = gt_boxes.shape[0]
    device = points.device

    if num_gt == 0:
        return torch.zeros((num_points,), dtype=torch.long, device=device)

    half = strides_per_point * anchor_scale / 2.0
    pseudo_anchors = torch.stack([
        points[:, 0] - half, points[:, 1] - half,
        points[:, 0] + half, points[:, 1] + half,
    ], dim=-1)

    ious = _box_iou(pseudo_anchors, gt_boxes)

    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gt_centers = torch.stack([gt_cx, gt_cy], dim=-1)
    distances = torch.cdist(points, gt_centers)

    candidate_idxs = []
    start = 0
    for num_p in num_points_per_level:
        end = start + num_p
        dist_level = distances[start:end]
        k = min(topk, num_p)
        _, topk_idx = dist_level.topk(k, dim=0, largest=False)
        candidate_idxs.append(topk_idx + start)
        start = end
    candidate_idxs = torch.cat(candidate_idxs, dim=0)

    candidate_ious = torch.gather(ious, 0, candidate_idxs)
    iou_thr = candidate_ious.mean(dim=0) + candidate_ious.std(dim=0)
    is_pos = candidate_ious >= iou_thr[None, :]

    candidate_points = points[candidate_idxs]
    l = candidate_points[..., 0] - gt_boxes[None, :, 0]
    t = candidate_points[..., 1] - gt_boxes[None, :, 1]
    r = gt_boxes[None, :, 2] - candidate_points[..., 0]
    b = gt_boxes[None, :, 3] - candidate_points[..., 1]
    is_in_gt = torch.stack([l, t, r, b], dim=-1).min(dim=-1).values > 0.01

    valid = is_pos & is_in_gt

    pos_mask_full = torch.zeros((num_points, num_gt), dtype=torch.bool, device=device)
    k_dim = candidate_idxs.shape[0]
    gt_idx_grid = torch.arange(num_gt, device=device).unsqueeze(0).expand(k_dim, num_gt)
    sel_point_idx = candidate_idxs[valid]
    sel_gt_idx = gt_idx_grid[valid]
    pos_mask_full[sel_point_idx, sel_gt_idx] = True

    overlaps_masked = ious.clone()
    overlaps_masked[~pos_mask_full] = -1.0
    max_overlaps, argmax_gt = overlaps_masked.max(dim=1)

    assigned_gt_inds = torch.zeros((num_points,), dtype=torch.long, device=device)
    valid_rows = max_overlaps > -0.5
    assigned_gt_inds[valid_rows] = argmax_gt[valid_rows] + 1
    return assigned_gt_inds
