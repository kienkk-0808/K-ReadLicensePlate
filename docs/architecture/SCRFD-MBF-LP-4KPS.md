# Thiết kế model SCRFD-MBF cho phát hiện biển số (4 keypoint góc)

## 1. Mục tiêu

Phát hiện biển số xe trong ảnh, với mỗi biển số trả về:
- Bounding box (x1, y1, x2, y2)
- Confidence score
- 4 điểm keypoint tương ứng 4 góc biển số: **top-left, top-right, bottom-right, bottom-left** (theo thứ tự cố định để phục vụ warp-perspective trước khi đưa vào OCR đọc biển số).

Kiến trúc dựa trên **SCRFD** (Sample and Computation Redistribution for Efficient Face Detection, InsightFace) — vốn thiết kế cho face detection + 5 keypoint (mắt/mũi/miệng) — được điều chỉnh: backbone thay bằng **MBF (MobileFaceNet-style)** để nhẹ, phù hợp chạy edge/embedded (camera AI, Jetson, RK3588...), và đầu keypoint đổi từ 5 điểm → **4 điểm góc biển số**.

## 2. Tổng quan kiến trúc

```
Input (RGB, ví dụ 640x640 hoặc 320x320 cho edge)
        │
        ▼
┌───────────────────┐
│   MBF Backbone     │  → xuất 3 feature map đa tỉ lệ: C3 (stride 8), C4 (stride 16), C5 (stride 32)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   PAFPN Neck       │  → hợp nhất top-down + bottom-up → P3, P4, P5 (cùng số kênh, ví dụ 64/96)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Detection Head     │  (shared weights giữa 3 level, anchor-free, theo kiểu FCOS/SCRFD)
│  ├─ Cls branch      │  → objectness/score biển số (1 kênh, sigmoid)
│  ├─ Bbox branch     │  → khoảng cách 4 phía (l, t, r, b) tới điểm anchor point
│  └─ Kps branch      │  → 4 điểm góc, offset (dx, dy) × 4 = 8 kênh
└───────────────────┘
        │
        ▼
   Decode theo stride của từng level + NMS
```

## 3. Backbone — MBF (MobileFaceNet-style)

Backbone gốc MobileFaceNet dùng cho face recognition (embedding), ở đây được "cắt cụt" (bỏ GDConv + FC embedding cuối) để dùng làm feature extractor đa tỉ lệ cho detection — giữ nguyên khối cơ bản: **Depthwise Separable Conv + Inverted Residual (MobileNetV2-style) + PReLU**.

### Cấu trúc các stage

> **Cập nhật (sau khi đo FLOPs thực tế):** bản thiết kế đầu tiên (24 block, downsample chậm) nặng **16.7 GFLOPs @640x640** — gấp ~33 lần SCRFD_500M gốc (~500 MFLOPs), khiến inference CPU mất ~400ms/ảnh dù backbone cùng "họ" MobileFaceNet. Nguyên nhân: downsample quá chậm (giữ stride 2 quá lâu trước khi vào block), quá nhiều block (24) và expand ratio cao (4x) áp dụng ở feature map độ phân giải lớn. Bảng dưới là bản đã tinh gọn lại theo đúng ngân sách "500M-1G class".

| Stage | Input stride | Output stride | Block | Số block | Channels (t=expand, c=out) |
|---|---|---|---|---|---|
| Stem | 1 | 2 | Conv3x3 s2 + PReLU | 1 | c=16 |
| DW-stem | 2 | 4 | Depthwise Conv3x3 s2 | 1 | c=16 (depthwise, groups=16) |
| Stage1 | 4 | 8 | Inverted Residual (bottleneck) | 2 | t=2, c=32 → **C3 (stride 8)** |
| Stage2 | 8 | 16 | Inverted Residual | 3 | t=2, c=64 → **C4 (stride 16)** |
| Stage3 | 16 | 32 | Inverted Residual | 2 | t=4, c=64 → **C5 (stride 32)** |

- Downsample đạt stride 4 chỉ sau 2 layer đầu (stem + dw-stem đều stride 2) — giảm compute sớm, đúng tinh thần backbone "500M-class" (khác bản đầu giữ stride 2 qua nhiều layer).
- Tổng chỉ còn **7 block** (thay vì 24), expand ratio thấp (2x) ở 2 stage đầu — chỉ dùng 4x ở stage cuối (C5, feature map đã nhỏ 20x20 nên chi phí thấp).
- Activation: **PReLU** toàn bộ (đặc trưng của MobileFaceNet, khác ReLU6 của MobileNetV2 gốc).
- BatchNorm sau mỗi conv (trước activation).
- 3 output feature map cần lấy ra cho neck: **C3 (stride 8), C4 (stride 16), C5 (stride 32)**.

### Biến thể độ nặng (width multiplier) — đo FLOPs thực tế bằng `thop`

| Biến thể | Width multiplier | fpn_channels | Tổng FLOPs @640x640 | Use case |
|---|---|---|---|---|
| MBF-0.5 | 0.5x | 32 | ~0.94 GFLOPs | Camera AI edge, NPU giới hạn (RK3568/RV1126) |
| MBF-1.0 | 1.0x (mặc định) | 32 | ~1.29 GFLOPs | Server/Jetson, cần độ chính xác cao hơn |

Bản đầu (16.7 GFLOPs) chạy ~400ms/ảnh trên CPU thường; bản đã tinh gọn (1.29 GFLOPs) chạy **~26ms/ảnh** cùng điều kiện — nhanh hơn ~15 lần, đúng bằng tỉ lệ giảm FLOPs.

## 4. Neck — PAFPN (Path Aggregation FPN)

Giống SCRFD gốc: top-down (semantic từ C5 xuống C3) + bottom-up (chi tiết từ C3 lên C5), giúp feature map nhỏ (P3, dùng để phát hiện biển số nhỏ/xa) vẫn có ngữ nghĩa tốt.

- Lateral conv 1x1: đưa C3/C4/C5 về cùng số kênh `fpn_channels` (mặc định **32** — đã giảm từ 64 ban đầu, vì head áp dụng conv dày đặc trên P3 80x80 nên chi phí tăng theo bình phương số kênh).
- Top-down: upsample (nearest x2) + add.
- Bottom-up: downsample conv3x3 stride2 + add.
- Mỗi P-level qua thêm 1 conv3x3 để làm mượt (smooth conv).

Output: **P3 (stride 8), P4 (stride 16), P5 (stride 32)** — mỗi cái `fpn_channels` kênh.

## 5. Head — Anchor-free Detection Head (dùng chung trọng số 3 level)

Theo đúng tinh thần SCRFD: **stacked conv head dùng chung (shared) giữa các level**, chỉ có scale factor riêng cho bbox regression mỗi level (learnable scalar, giống FCOS).

> **Cập nhật (sau khi đo FLOPs thực tế):** `stacked_convs=2` ban đầu khiến head chiếm tới **73% tổng compute** (1.93/2.65 GMacs) — vì mỗi conv3x3 64→64 áp trên P3 (80x80) lặp lại 2 lần cho CẢ 3 nhánh (cls/bbox/kps) riêng biệt. Đã giảm xuống `stacked_convs=1` + `fpn_channels=32` (mặc định mới) để cân bằng lại ngân sách FLOPs với backbone.

### Cấu trúc mỗi nhánh (áp dụng độc lập trên P3/P4/P5, cùng bộ trọng số)

```
Input Pk (fpn_channels=32)
   │
   ├─ Cls stem: [Conv3x3 + GN + ReLU] x 1  → Conv3x3 → num_classes kênh (sigmoid)
   │
   ├─ Bbox stem: [Conv3x3 + GN + ReLU] x 1  → Conv3x3 → 4 kênh (l, t, r, b), nhân scale[k] học được, sau đó x stride[k]
   │
   └─ Kps stem: [Conv3x3 + GN + ReLU] x 1   → Conv3x3 → 8 kênh (dx1,dy1,dx2,dy2,dx3,dy3,dx4,dy4)
                                                 nhân scale_kps[k] học được, sau đó x stride[k]
```

- `stacked_convs` cấu hình được qua `SCRFDHead(stacked_convs=...)` — tăng lên 2 nếu cần thêm khả năng biểu diễn và chấp nhận đánh đổi FLOPs (đã đo: mỗi +1 stacked_conv ở fpn_channels=32 cộng thêm ~0.2 GMacs).
- GN (GroupNorm) thay vì BN trong head — chuẩn thực hành để ổn định khi batch nhỏ lúc fine-tune.

### Số kênh output mỗi level (tại 1 vị trí anchor point)

| Nhánh | Kênh | Ý nghĩa |
|---|---|---|
| Cls | 1 | objectness score biển số (không cần multi-class nếu chỉ detect 1 loại "license plate") |
| Bbox | 4 | (l, t, r, b) — khoảng cách từ điểm tới 4 cạnh box, đơn vị pixel sau khi nhân stride |
| Kps | 8 | (dx, dy) × 4 góc — offset từ điểm anchor tới từng góc, đơn vị pixel sau khi nhân stride |

## 6. Anchor-free encoding / decoding (theo kiểu FCOS/SCRFD)

Với mỗi level k có stride `s_k ∈ {8, 16, 32}`, tại mỗi vị trí lưới `(i, j)` trên feature map, điểm anchor tương ứng trong ảnh gốc:

```
px = (j + 0.5) * s_k
py = (i + 0.5) * s_k
```

**Decode bbox:**
```
x1 = px - l * s_k
y1 = py - t * s_k
x2 = px + r * s_k
y2 = py + b * s_k
```

**Decode keypoint (góc thứ m, m=0..3):**
```
kp_x[m] = px + dx[m] * s_k
kp_y[m] = py + dy[m] * s_k
```

**Gán nhãn khi train (label assignment):** dùng **ATSS (Adaptive Training Sample Selection)** như SCRFD gốc — chọn top-k anchor point gần tâm GT box nhất theo IoU với box thô, threshold = mean + std IoU của tập candidate. Ưu điểm: không cần tune anchor thủ công, cân bằng số sample dương giữa các level tốt hơn so với gán theo range diện tích cố định.

## 7. Loss function

| Nhánh | Loss | Ghi chú |
|---|---|---|
| Cls | **Quality Focal Loss (QFL)** hoặc Focal Loss thường | target = IoU giữa box dự đoán và GT (soft label) nếu dùng QFL |
| Bbox | **DIoU Loss** (hoặc GIoU) | tính trên box đã decode, chỉ tính tại vị trí dương (positive sample) |
| Kps | **Smooth-L1 (weighted)** | chuẩn hoá theo kích thước box GT: `loss = SmoothL1((kp_pred - kp_gt) / box_diag)` để ổn định gradient giữa biển số to/nhỏ |

Tổng loss:
```
L = λ_cls * L_cls + λ_bbox * L_bbox + λ_kps * L_kps
```
Mặc định: `λ_cls=1.0, λ_bbox=1.0, λ_kps=2.0` (cấu hình qua `--lambda-cls/--lambda-bbox/--lambda-kps` trong `train.py`).

> **Cập nhật (sau khi quan sát training thật):** bản đầu dùng `λ_kps=0.5` — checkpoint 12 epoch đạt mAP@0.5=0.85 (box tốt) nhưng `kps_nme=0.13` (keypoint vẽ ra lệch rõ so với góc biển thật). mAP@0.5 chỉ đo IoU box, không phản ánh gì về keypoint, nên nhánh box có thể "báo cáo tốt" trong khi nhánh keypoint vẫn chưa hội tụ. Tăng `λ_kps` lên 2.0 (gấp 4 lần) để ép gradient nhánh keypoint mạnh hơn, tương xứng với việc regression 4 điểm góc vốn là bài toán khó hơn (nhạy với nhiễu/xoay) so với chỉ regress 4 khoảng cách cạnh box.

## 8. Input / Output tensor shape (ví dụ input 640x640)

| Level | Feature map size | Cls | Bbox | Kps |
|---|---|---|---|---|
| P3 (stride 8) | 80x80 | 80x80x1 | 80x80x4 | 80x80x8 |
| P4 (stride 16) | 40x40 | 40x40x1 | 40x40x4 | 40x40x8 |
| P5 (stride 32) | 20x20 | 20x20x1 | 20x20x4 | 20x20x8 |

Tổng số anchor point: 80x80 + 40x40 + 20x20 = 8400.

## 9. Hậu xử lý (post-process)

1. Threshold cls score (ví dụ > 0.3).
2. Decode bbox + kps theo công thức mục 6.
3. NMS (IoU threshold ~0.4) trên bbox, giữ lại kps tương ứng với box thắng.
4. Sắp xếp lại 4 điểm góc theo đúng thứ tự TL→TR→BR→BL (sort theo góc so với centroid) để đảm bảo tính nhất quán trước khi warp-perspective, phòng trường hợp model học lệch thứ tự ở biển số bị nghiêng/xoay mạnh.

## 10. Lý do chọn thiết kế này

- **MBF làm backbone**: giữ nguyên tinh thần SCRFD gốc (thiết kế cho thiết bị hạn chế tài nguyên), tận dụng block depthwise-separable + PReLU đã được kiểm chứng nhẹ và hiệu quả cho bài toán detect vật thể nhỏ/vừa như biển số — phù hợp deploy trên camera AI/edge của dòng sản phẩm iParking.
- **4 keypoint thay vì 5**: biển số là hình chữ nhật phẳng, chỉ cần 4 góc để warp-perspective chuẩn hoá ảnh trước khi OCR — không cần điểm giữa như face landmark.
- **Anchor-free + ATSS**: giảm số hyperparameter (không cần tune anchor ratio/scale theo tỉ lệ khung biển số dài/hẹp khác nhau giữa các quốc gia/loại xe).
- **Shared head giữa các level**: giảm tham số, tăng tốc độ suy luận — quan trọng khi chạy real-time trên nhiều luồng camera.

## 11. File liên quan

- Code khung PyTorch: [`models/scrfd_mbf.py`](../../models/scrfd_mbf.py)
