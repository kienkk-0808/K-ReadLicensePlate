"""Đánh giá detection thật: mAP@0.5 và mAP@0.75 (bbox-only)."""

from typing import Dict, List

import numpy as np
import torch
from torchvision.ops import batched_nms, box_iou


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """AP kiểu VOC2012 (all-point interpolation)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _compute_map(gt_records, pred_records, num_classes, iou_thr) -> float:
    ap_per_class = {}
    for c in range(num_classes):
        gts = gt_records[c]
        preds = sorted(pred_records[c], key=lambda x: -x["score"])
        num_gt = len(gts)

        if num_gt == 0:
            ap_per_class[c] = None
            continue

        matched = [False] * num_gt
        gts_by_image: Dict[int, List[int]] = {}
        for gi, g in enumerate(gts):
            gts_by_image.setdefault(g["image_idx"], []).append(gi)

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        for pi, p in enumerate(preds):
            candidate_idx = gts_by_image.get(p["image_idx"], [])
            best_iou, best_gi = 0.0, -1
            if candidate_idx:
                cand_boxes = torch.tensor(
                    np.stack([gts[gi]["box"] for gi in candidate_idx]), dtype=torch.float32
                )
                pred_box = torch.tensor(p["box"], dtype=torch.float32).unsqueeze(0)
                ious = box_iou(pred_box, cand_boxes).squeeze(0).numpy()
                best_local = int(np.argmax(ious))
                best_iou = float(ious[best_local])
                best_gi = candidate_idx[best_local]

            if best_iou >= iou_thr and best_gi >= 0 and not matched[best_gi]:
                tp[pi] = 1
                matched[best_gi] = True
            else:
                fp[pi] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / num_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        ap_per_class[c] = _voc_ap(recall, precision) if len(preds) else 0.0

    valid_aps = [v for v in ap_per_class.values() if v is not None]
    return float(np.mean(valid_aps)) if valid_aps else 0.0


@torch.no_grad()
def evaluate_metrics(
    model, val_loader, device,
    score_thr: float = 0.05, nms_iou: float = 0.5, map_iou: float = 0.5,
    num_classes: int = 1,
) -> Dict[str, float]:
    """-> {"mAP50", "mAP75"}. Càng cao càng tốt."""
    model.eval()

    gt_records: Dict[int, List[dict]] = {c: [] for c in range(num_classes)}
    pred_records: Dict[int, List[dict]] = {c: [] for c in range(num_classes)}

    img_idx = 0
    for imgs, targets in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        decoded = model.decode(outputs, score_thr=score_thr)

        for i, (boxes, scores, labels) in enumerate(decoded):
            if boxes.shape[0] > 0:
                keep = batched_nms(boxes, scores, labels, nms_iou)
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            for b, s, l in zip(boxes, scores, labels):
                pred_records[int(l.item())].append({
                    "score": float(s.item()), "box": b.cpu().numpy(), "image_idx": img_idx,
                })

            gt_boxes = targets[i]["boxes"].numpy()
            gt_labels = targets[i]["labels"].numpy()
            for b, l in zip(gt_boxes, gt_labels):
                gt_records[int(l)].append({"image_idx": img_idx, "box": b})

            img_idx += 1

    map50 = _compute_map(gt_records, pred_records, num_classes, map_iou)
    map75 = _compute_map(gt_records, pred_records, num_classes, 0.75)

    model.train()
    return {"mAP50": map50, "mAP75": map75}
