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

| Stage | Input stride | Output stride | Block | Số block | Channels (t=expand, c=out) |
|---|---|---|---|---|---|
| Stem | 1 | 2 | Conv3x3 s2 + PReLU | 1 | c=32 |
| DW-stem | 2 | 2 | Depthwise Conv3x3 (residual sau) | 1 | c=32 (depthwise, groups=32) |
| Stage1 | 2 | 4 | Inverted Residual (bottleneck) | 4 | t=2, c=64 |
| Stage2 | 4 | 8 | Inverted Residual | 6 | t=4, c=64 → **C3 (stride 8)** |
| Stage3 | 8 | 16 | Inverted Residual | 8 | t=4, c=128 → **C4 (stride 16)** |
| Stage4 | 16 | 32 | Inverted Residual | 6 | t=4, c=128 → **C5 (stride 32)** |
| Conv-final | 32 | 32 | Conv1x1 + PReLU | 1 | c=256 (chỉ dùng nếu cần thêm ngữ nghĩa cho C5) |

- Activation: **PReLU** toàn bộ (đặc trưng của MobileFaceNet, khác ReLU6 của MobileNetV2 gốc).
- BatchNorm sau mỗi conv (trước activation).
- Không dùng SE-block để giữ nhẹ (có thể bật SE ở Stage3/4 nếu cần tăng độ chính xác, đánh đổi FLOPs).
- 3 output feature map cần lấy ra cho neck: **C3 (stride 8), C4 (stride 16), C5 (stride 32)**.

### Biến thể độ nặng (width multiplier)

| Biến thể | Width multiplier | Use case |
|---|---|---|
| MBF-0.5 | 0.5x | Camera AI edge, NPU giới hạn (RK3568/RV1126) |
| MBF-1.0 | 1.0x (mặc định, bảng trên) | Server/Jetson, cần độ chính xác cao hơn |

## 4. Neck — PAFPN (Path Aggregation FPN)

Giống SCRFD gốc: top-down (semantic từ C5 xuống C3) + bottom-up (chi tiết từ C3 lên C5), giúp feature map nhỏ (P3, dùng để phát hiện biển số nhỏ/xa) vẫn có ngữ nghĩa tốt.

- Lateral conv 1x1: đưa C3/C4/C5 về cùng số kênh `fpn_channels` (khuyến nghị 64 cho bản nhẹ, 96 cho bản chuẩn).
- Top-down: upsample (nearest x2) + add.
- Bottom-up: downsample conv3x3 stride2 + add.
- Mỗi P-level qua thêm 1 conv3x3 để làm mượt (smooth conv).

Output: **P3 (stride 8), P4 (stride 16), P5 (stride 32)** — mỗi cái `fpn_channels` kênh.

## 5. Head — Anchor-free Detection Head (dùng chung trọng số 3 level)

Theo đúng tinh thần SCRFD: **stacked conv head dùng chung (shared) giữa các level**, chỉ có scale factor riêng cho bbox regression mỗi level (learnable scalar, giống FCOS).

### Cấu trúc mỗi nhánh (áp dụng độc lập trên P3/P4/P5, cùng bộ trọng số)

```
Input Pk (fpn_channels)
   │
   ├─ Cls stem: [Conv3x3 + GN + ReLU] x 2  → Conv3x3 → 1 kênh (objectness biển số, sigmoid)
   │
   ├─ Bbox stem: [Conv3x3 + GN + ReLU] x 2  → Conv3x3 → 4 kênh (l, t, r, b), nhân scale[k] học được, sau đó x stride[k]
   │
   └─ Kps stem: [Conv3x3 + GN + ReLU] x 2   → Conv3x3 → 8 kênh (dx1,dy1,dx2,dy2,dx3,dy3,dx4,dy4)
                                                 nhân scale_kps[k] học được, sau đó x stride[k]
```

- **Cls stem** và **Bbox stem** có thể share 2 conv đầu (giảm tham số), tách nhánh ở conv cuối — tuỳ ngân sách tham số. Kps stem tách riêng hoàn toàn vì task khác biệt (regression điểm, không phải cạnh box).
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
Khuyến nghị khởi điểm: `λ_cls=1.0, λ_bbox=1.0, λ_kps=0.5` (giảm dần theo epoch nếu keypoint dataset ít hơn bbox dataset).

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
