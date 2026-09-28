"""Loss cho SCRFD_MBF:
- Cls: Sigmoid Focal Loss (hard label 0/1), giống RetinaNet/ATSS/SCRFD gốc.
- Bbox: DIoU Loss, chỉ tính tại vị trí positive.
- Kps: Smooth-L1, chuẩn hoá theo đường chéo box GT, chỉ tính tại vị trí positive.

Ghi chú so với docs/architecture/SCRFD-MBF-LP-4KPS.md: doc mô tả phương án dùng
Quality Focal Loss (soft target = IoU) cho nhánh cls. Ở bản implement này dùng
Focal Loss với target cứng (0/1) để đơn giản hoá vòng lặp train (không cần decode
pred box ngay trong loss để tính IoU động mỗi step) — đây là lựa chọn tiêu chuẩn
của SCRFD/ATSS gốc (mmdetection ATSSHead mặc định dùng FocalLoss), vẫn đúng tinh
thần thiết kế, chỉ khác ở việc soft/hard target.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.atss_assigner import atss_assign
from models.scrfd_utils import generate_points, flatten_head_outputs, decode_points


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                        alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """logits, targets: cùng shape (..., num_classes). Trả về loss chưa reduce (sum-ready)."""
    prob = logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss


def bbox_diou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """pred, target: (N, 4) xyxy. Trả về (N,) loss = 1 - DIoU."""
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)

    pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    target_area = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)

    inter_x1 = torch.max(px1, tx1)
    inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2)
    inter_y2 = torch.min(py2, ty2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    union = pred_area + target_area - inter_area + eps
    iou = inter_area / union

    enclose_x1 = torch.min(px1, tx1)
    enclose_y1 = torch.min(py1, ty1)
    enclose_x2 = torch.max(px2, tx2)
    enclose_y2 = torch.max(py2, ty2)
    c2 = (enclose_x2 - enclose_x1).pow(2) + (enclose_y2 - enclose_y1).pow(2) + eps

    p_cx, p_cy = (px1 + px2) / 2, (py1 + py2) / 2
    t_cx, t_cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    rho2 = (p_cx - t_cx).pow(2) + (p_cy - t_cy).pow(2)

    diou = iou - rho2 / c2
    return 1 - diou


class SCRFDLoss(nn.Module):
    def __init__(self, img_size: int = 640, num_classes: int = 2,
                 lambda_cls: float = 1.0, lambda_bbox: float = 1.0, lambda_kps: float = 2.0,
                 topk: int = 9, anchor_scale: float = 8.0):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_bbox = lambda_bbox
        self.lambda_kps = lambda_kps
        self.topk = topk
        self.anchor_scale = anchor_scale

        points, strides_per_point, num_points_per_level = generate_points(img_size)
        self.register_buffer("points", points)
        self.register_buffer("strides_per_point", strides_per_point)
        self.num_points_per_level = num_points_per_level

    def forward(self, outputs, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        cls_logits, bbox_dist, kps_offset = flatten_head_outputs(outputs)
        # (B, N, num_classes), (B, N, 4), (B, N, 8)
        batch_size = cls_logits.shape[0]
        device = cls_logits.device

        pred_boxes, pred_kps = decode_points(self.points, bbox_dist, kps_offset)
        # pred_boxes: (B, N, 4) | pred_kps: (B, N, 4, 2)

        total_cls_loss = cls_logits.new_zeros(())
        total_bbox_loss = cls_logits.new_zeros(())
        total_kps_loss = cls_logits.new_zeros(())
        total_pos = 0

        for i in range(batch_size):
            gt_boxes = targets[i]["boxes"].to(device)
            gt_labels = targets[i]["labels"].to(device)
            gt_kps = targets[i]["kps"].to(device)

            cls_target = cls_logits.new_zeros((cls_logits.shape[1], self.num_classes))

            if gt_boxes.shape[0] > 0:
                assigned_gt_inds = atss_assign(
                    self.points, self.strides_per_point, self.num_points_per_level,
                    gt_boxes, topk=self.topk, anchor_scale=self.anchor_scale,
                )
                pos_mask = assigned_gt_inds > 0
                num_pos = int(pos_mask.sum().item())

                if num_pos > 0:
                    pos_gt_idx = assigned_gt_inds[pos_mask] - 1
                    cls_target[pos_mask, gt_labels[pos_gt_idx]] = 1.0

                    matched_boxes = gt_boxes[pos_gt_idx]
                    matched_kps = gt_kps[pos_gt_idx]

                    pred_boxes_pos = pred_boxes[i][pos_mask]
                    bbox_loss = bbox_diou_loss(pred_boxes_pos, matched_boxes).sum()

                    diag = torch.sqrt(
                        (matched_boxes[:, 2] - matched_boxes[:, 0]).pow(2)
                        + (matched_boxes[:, 3] - matched_boxes[:, 1]).pow(2)
                    ).clamp(min=1.0)
                    pred_kps_pos = pred_kps[i][pos_mask]  # (num_pos, 4, 2)
                    kps_diff = (pred_kps_pos - matched_kps) / diag.view(-1, 1, 1)
                    kps_loss = F.smooth_l1_loss(
                        kps_diff, torch.zeros_like(kps_diff), reduction="sum"
                    )

                    total_bbox_loss = total_bbox_loss + bbox_loss
                    total_kps_loss = total_kps_loss + kps_loss
                    total_pos += num_pos

            cls_loss = sigmoid_focal_loss(cls_logits[i], cls_target).sum()
            total_cls_loss = total_cls_loss + cls_loss

        norm = max(total_pos, 1)
        cls_loss = total_cls_loss / norm
        bbox_loss = total_bbox_loss / norm
        kps_loss = total_kps_loss / norm

        loss = self.lambda_cls * cls_loss + self.lambda_bbox * bbox_loss + self.lambda_kps * kps_loss

        return {
            "loss": loss,
            "cls_loss": cls_loss.detach(),
            "bbox_loss": bbox_loss.detach(),
            "kps_loss": kps_loss.detach(),
            "num_pos": torch.tensor(float(total_pos)),
        }
