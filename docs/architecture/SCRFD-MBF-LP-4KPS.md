# Thiết kế model SCRFD-MBF cho phát hiện biển số (bbox-only)

> **Lịch sử:** bản đầu có thêm nhánh 4 keypoint góc biển số (để warp-perspective trước OCR).
> Đã bỏ nhánh keypoint để đơn giản hoá bài toán và dồn toàn bộ ngân sách compute cho
> cls/bbox, tối đa hoá độ chính xác phát hiện box. Tài liệu này mô tả bản hiện tại.

## 1. Mục tiêu

Phát hiện biển số xe trong ảnh, trả về bounding box (x1,y1,x2,y2) + confidence score.
Kiến trúc dựa trên **SCRFD** (InsightFace), backbone thay bằng **MBF (MobileFaceNet-style)**
để nhẹ, phù hợp edge/embedded.

## 2. Tổng quan kiến trúc

```
Input (RGB, vd 640x640)
        │
        ▼
┌───────────────────┐
│   MBF Backbone     │  → C3 (stride 8), C4 (stride 16), C5 (stride 32)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   PAFPN Neck       │  → P3, P4, P5 (cùng fpn_channels)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Detection Head     │  (shared weights giữa 3 level, anchor-free, kiểu FCOS/SCRFD)
│  ├─ Cls branch      │  → objectness/score biển số (num_classes kênh, sigmoid)
│  └─ Bbox branch     │  → khoảng cách 4 phía (l, t, r, b) tới anchor point
└───────────────────┘
        │
        ▼
   Decode theo stride của từng level + NMS
```

## 3. Backbone — MBF (MobileFaceNet-style)

7 block (stem+dw_stem đạt stride 4 ngay từ đầu, 3 stage InvertedResidual), ~0.26 GMacs.
`width_mult`: 0.5 cho bản edge nhẹ hơn, 1.0 mặc định.

| Stage | Output stride | Block | Channels (t=expand) |
|---|---|---|---|
| Stem + DW-stem | 4 | Conv3x3 s2 + DW Conv3x3 s2 | c=16 |
| Stage1 | 8 (C3) | 2 block | t=2, c=32 |
| Stage2 | 16 (C4) | 3 block | t=2, c=64 |
| Stage3 | 32 (C5) | 2 block | t=4, c=64 |

## 4. Neck — PAFPN

Top-down + bottom-up, lateral conv 1x1 đưa C3/C4/C5 về `fpn_channels` (mặc định 48).

## 5. Head — Anchor-free (dùng chung trọng số 3 level)

```
Input Pk (fpn_channels)
   │
   ├─ Cls stem: [Conv3x3 + GN + ReLU] × stacked_convs → Conv3x3 → num_classes (sigmoid)
   │
   └─ Bbox stem: [Conv3x3 + GN + ReLU] × stacked_convs → Conv3x3 → 4 (l,t,r,b) × scale[k] × stride[k]
```

`stacked_convs` mặc định 2 (sau khi bỏ nhánh kps, ngân sách compute dư ra được dồn
cho cls/bbox để tăng độ chính xác — tổng ~2.48 GFLOPs @640x640, 0.344M params).

## 6. Anchor-free encoding/decoding (FCOS-style)

Điểm anchor tại `(i,j)` level stride `s`: `px=(j+0.5)*s, py=(i+0.5)*s`.

```
x1 = px - l*s   y1 = py - t*s   x2 = px + r*s   y2 = py + b*s
```

Label assignment: **ATSS** (top-k theo khoảng cách tâm + ngưỡng IoU thích ứng).

## 7. Loss function

| Nhánh | Loss |
|---|---|
| Cls | Sigmoid Focal Loss (hard label) |
| Bbox | **CIoU** (DIoU + số hạng phạt lệch tỉ lệ khung — chặt hơn DIoU cho độ chính xác box) |

```
L = λ_cls * L_cls + λ_bbox * L_bbox
```
Mặc định `λ_cls=1.0, λ_bbox=1.0` (`--lambda-cls/--lambda-bbox`).

## 8. Đánh giá & chọn checkpoint

`score = (mAP@0.5 + mAP@0.75) / 2` — thưởng box khít (mAP75 đòi hỏi IoU cao), không
chỉ IoU lỏng 0.5. Elitist: revert về best.pt sau `--patience` epoch liên tiếp không
cải thiện (chống nhiễu tập valid nhỏ).

## 9. Hậu xử lý

1. Threshold cls score.
2. Decode bbox theo mục 6.
3. NMS (torchvision.ops.batched_nms, per-class).

## 10. File liên quan

- Code: [`models/scrfd_mbf.py`](../../models/scrfd_mbf.py), [`models/losses.py`](../../models/losses.py)
