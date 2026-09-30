"""Loss cho SCRFD_MBF (bbox-only): cls = Sigmoid Focal Loss, bbox = CIoU + DFL."""

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.atss_assigner import atss_assign
from models.scrfd_utils import generate_points, flatten_head_outputs, decode_points


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                        alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    prob = logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss


def bbox_ciou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """pred, target: (N,4) xyxy. -> (N,) loss = 1 - CIoU."""
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)

    pred_w = (px2 - px1).clamp(min=eps)
    pred_h = (py2 - py1).clamp(min=eps)
    target_w = (tx2 - tx1).clamp(min=eps)
    target_h = (ty2 - ty1).clamp(min=eps)

    pred_area = pred_w * pred_h
    target_area = target_w * target_h

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

    v = (4 / math.pi ** 2) * (torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha * v
    return 1 - ciou


def dfl_loss(pred_logits: torch.Tensor, target: torch.Tensor, reg_max: int) -> torch.Tensor:
    """pred_logits: (N,4,reg_max+1) raw. target: (N,4) liên tục, đơn vị stride.

    Target không nguyên -> phân bổ cross-entropy có trọng số cho 2 bin nguyên liền kề
    (chuẩn DFL, GFocal/YOLOv8) thay vì chỉ 1 bin, giữ được thông tin sub-pixel.
    """
    target = target.clamp(0, reg_max - 0.01)
    left = target.long()
    right = left + 1

    weight_left = right.float() - target
    weight_right = target - left.float()

    log_prob = F.log_softmax(pred_logits, dim=-1)
    loss_left = -log_prob.gather(-1, left.unsqueeze(-1)).squeeze(-1) * weight_left
    loss_right = -log_prob.gather(-1, right.clamp(max=reg_max).unsqueeze(-1)).squeeze(-1) * weight_right
    return loss_left + loss_right


class SCRFDLoss(nn.Module):
    def __init__(self, img_size: int = 640, num_classes: int = 1,
                 lambda_cls: float = 1.0, lambda_bbox: float = 1.0, lambda_dfl: float = 0.25,
                 reg_max: int = 16, topk: int = 9, anchor_scale: float = 8.0):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_bbox = lambda_bbox
        self.lambda_dfl = lambda_dfl
        self.reg_max = reg_max
        self.topk = topk
        self.anchor_scale = anchor_scale

        points, strides_per_point, num_points_per_level = generate_points(img_size)
        self.register_buffer("points", points)
        self.register_buffer("strides_per_point", strides_per_point)
        self.num_points_per_level = num_points_per_level

    def forward(self, outputs, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        cls_logits, bbox_logits = flatten_head_outputs(outputs)
        batch_size = cls_logits.shape[0]
        device = cls_logits.device

        pred_boxes = decode_points(self.points, self.strides_per_point, bbox_logits, self.reg_max)

        total_cls_loss = cls_logits.new_zeros(())
        total_bbox_loss = cls_logits.new_zeros(())
        total_dfl_loss = cls_logits.new_zeros(())
        total_pos = 0

        for i in range(batch_size):
            gt_boxes = targets[i]["boxes"].to(device)
            gt_labels = targets[i]["labels"].to(device)

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
                    pred_boxes_pos = pred_boxes[i][pos_mask]
                    bbox_loss = bbox_ciou_loss(pred_boxes_pos, matched_boxes).sum()

                    pos_points = self.points[pos_mask]
                    pos_strides = self.strides_per_point[pos_mask]
                    target_l = (pos_points[:, 0] - matched_boxes[:, 0]) / pos_strides
                    target_t = (pos_points[:, 1] - matched_boxes[:, 1]) / pos_strides
                    target_r = (matched_boxes[:, 2] - pos_points[:, 0]) / pos_strides
                    target_b = (matched_boxes[:, 3] - pos_points[:, 1]) / pos_strides
                    target_dist = torch.stack([target_l, target_t, target_r, target_b], dim=-1)

                    bbox_logits_pos = bbox_logits[i][pos_mask].reshape(num_pos, 4, self.reg_max + 1)
                    dfl = dfl_loss(bbox_logits_pos, target_dist, self.reg_max).sum()

                    total_bbox_loss = total_bbox_loss + bbox_loss
                    total_dfl_loss = total_dfl_loss + dfl
                    total_pos += num_pos

            cls_loss = sigmoid_focal_loss(cls_logits[i], cls_target).sum()
            total_cls_loss = total_cls_loss + cls_loss

        norm = max(total_pos, 1)
        cls_loss = total_cls_loss / norm
        bbox_loss = total_bbox_loss / norm
        dfl = total_dfl_loss / norm

        loss = self.lambda_cls * cls_loss + self.lambda_bbox * bbox_loss + self.lambda_dfl * dfl

        return {
            "loss": loss,
            "cls_loss": cls_loss.detach(),
            "bbox_loss": bbox_loss.detach(),
            "dfl_loss": dfl.detach(),
            "num_pos": torch.tensor(float(total_pos)),
        }
