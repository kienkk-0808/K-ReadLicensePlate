"""Đánh giá detection thật trong lúc train: mAP@0.5 (box) + NME keypoint (trên các
match true-positive), thay vì chỉ dựa vào val loss để chọn checkpoint tốt nhất —
loss thấp không chắc đã tương ứng chất lượng phát hiện tốt (đặc biệt giai đoạn đầu
train khi cls loss có thể giảm nhanh dù bbox/kps còn kém).
"""

from typing import Dict, List

import numpy as np
import torch
from torchvision.ops import batched_nms, box_iou

from models.scrfd_mbf import NUM_KPS


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """AP kiểu VOC2012 (all-point interpolation)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@torch.no_grad()
def evaluate_metrics(
    model, val_loader, device,
    score_thr: float = 0.05, nms_iou: float = 0.5, map_iou: float = 0.5,
    num_classes: int = 2,
) -> Dict[str, float]:
    """Chạy model trên toàn bộ val_loader, trả về:
        {"mAP50": ..., "kps_nme": ..., "AP50_class_<i>": ...}
    mAP50 càng CAO càng tốt, kps_nme càng THẤP càng tốt.
    """
    model.eval()

    # gt_records[class_id] = list of {"image_idx", "box"(4,), "kps"(4,2)}
    gt_records: Dict[int, List[dict]] = {c: [] for c in range(num_classes)}
    # pred_records[class_id] = list of {"score", "box"(4,), "kps"(4,2), "image_idx"}
    pred_records: Dict[int, List[dict]] = {c: [] for c in range(num_classes)}

    img_idx = 0
    for imgs, targets in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        decoded = model.decode(outputs, score_thr=score_thr)

        for i, (boxes, scores, labels, kps) in enumerate(decoded):
            if boxes.shape[0] > 0:
                keep = batched_nms(boxes, scores, labels, nms_iou)
                boxes, scores, labels, kps = boxes[keep], scores[keep], labels[keep], kps[keep]

            for b, s, l, k in zip(boxes, scores, labels, kps):
                pred_records[int(l.item())].append({
                    "score": float(s.item()), "box": b.cpu().numpy(),
                    "kps": k.cpu().numpy(), "image_idx": img_idx,
                })

            gt_boxes = targets[i]["boxes"].numpy()
            gt_labels = targets[i]["labels"].numpy()
            gt_kps = targets[i]["kps"].numpy()
            for b, l, k in zip(gt_boxes, gt_labels, gt_kps):
                gt_records[int(l)].append({"image_idx": img_idx, "box": b, "kps": k})

            img_idx += 1

    ap_per_class = {}
    all_tp_kps_err = []

    for c in range(num_classes):
        gts = gt_records[c]
        preds = sorted(pred_records[c], key=lambda x: -x["score"])
        num_gt = len(gts)

        if num_gt == 0:
            ap_per_class[c] = None  # không có GT lớp này trong tập valid -> bỏ qua khi macro-average
            continue

        matched = [False] * num_gt
        # gom GT theo image để tra cứu nhanh
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

            if best_iou >= map_iou and best_gi >= 0 and not matched[best_gi]:
                tp[pi] = 1
                matched[best_gi] = True

                gt_box = gts[best_gi]["box"]
                gt_kps = gts[best_gi]["kps"]
                diag = np.sqrt((gt_box[2] - gt_box[0]) ** 2 + (gt_box[3] - gt_box[1]) ** 2)
                diag = max(diag, 1.0)
                err = np.linalg.norm(p["kps"] - gt_kps, axis=-1).mean() / diag
                all_tp_kps_err.append(err)
            else:
                fp[pi] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / num_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        ap_per_class[c] = _voc_ap(recall, precision) if len(preds) else 0.0

    valid_aps = [v for v in ap_per_class.values() if v is not None]
    map50 = float(np.mean(valid_aps)) if valid_aps else 0.0
    kps_nme = float(np.mean(all_tp_kps_err)) if all_tp_kps_err else float("inf")

    model.train()

    result = {"mAP50": map50, "kps_nme": kps_nme}
    for c, ap in ap_per_class.items():
        result[f"AP50_class_{c}"] = ap if ap is not None else float("nan")
    return result
