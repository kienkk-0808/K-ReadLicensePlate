"""
SCRFD-MBF: SCRFD detector với backbone MobileFaceNet-style (MBF), đầu ra 4 keypoint
góc biển số (thay vì 5 keypoint khuôn mặt của SCRFD gốc).

Kiến trúc: xem docs/architecture/SCRFD-MBF-LP-4KPS.md
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_KPS = 4  # 4 góc biển số: top-left, top-right, bottom-right, bottom-left
STRIDES = (8, 16, 32)


def conv_bn_prelu(in_c, out_c, kernel=3, stride=1, groups=1):
    padding = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_c),
        nn.PReLU(out_c),
    )


class InvertedResidual(nn.Module):
    """Bottleneck MobileNetV2-style dùng PReLU (đặc trưng MobileFaceNet)."""

    def __init__(self, in_c, out_c, stride, expand_ratio):
        super().__init__()
        assert stride in (1, 2)
        hidden = int(round(in_c * expand_ratio))
        self.use_residual = stride == 1 and in_c == out_c

        layers = []
        if expand_ratio != 1:
            layers.append(conv_bn_prelu(in_c, hidden, kernel=1))
        layers.append(conv_bn_prelu(hidden, hidden, kernel=3, stride=stride, groups=hidden))
        layers.append(nn.Conv2d(hidden, out_c, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_c))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class MBFBackbone(nn.Module):
    """MobileFaceNet-style backbone, cắt bỏ GDConv/embedding, xuất C3/C4/C5.

    width_mult: 0.5 cho bản edge/NPU nhẹ, 1.0 cho bản chuẩn.
    """

    def __init__(self, width_mult: float = 1.0):
        super().__init__()

        def c(ch):
            return max(8, int(round(ch * width_mult / 8) * 8))

        self.stem = conv_bn_prelu(3, c(32), kernel=3, stride=2)
        self.dw_stem = conv_bn_prelu(c(32), c(32), kernel=3, stride=1, groups=c(32))

        self.stage1 = self._make_stage(c(32), c(64), stride=2, expand=2, n=4)   # -> stride 4
        self.stage2 = self._make_stage(c(64), c(64), stride=2, expand=4, n=6)   # -> stride 8  (C3)
        self.stage3 = self._make_stage(c(64), c(128), stride=2, expand=4, n=8)  # -> stride 16 (C4)
        self.stage4 = self._make_stage(c(128), c(128), stride=2, expand=4, n=6) # -> stride 32 (C5)

        self.out_channels = (c(64), c(128), c(128))
        self._init_weights()

    @staticmethod
    def _make_stage(in_c, out_c, stride, expand, n):
        layers = [InvertedResidual(in_c, out_c, stride, expand)]
        for _ in range(n - 1):
            layers.append(InvertedResidual(out_c, out_c, 1, expand))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.dw_stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c3, c4, c5


class PAFPN(nn.Module):
    """Path Aggregation FPN: top-down + bottom-up, theo tinh thần SCRFD."""

    def __init__(self, in_channels: Tuple[int, int, int], out_channels: int = 64):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        self.smooth_td = nn.ModuleList([
            conv_bn_prelu(out_channels, out_channels, kernel=3) for _ in range(2)
        ])
        self.downsample = nn.ModuleList([
            conv_bn_prelu(out_channels, out_channels, kernel=3, stride=2) for _ in range(2)
        ])
        self.smooth_bu = nn.ModuleList([
            conv_bn_prelu(out_channels, out_channels, kernel=3) for _ in range(2)
        ])

    def forward(self, feats):
        c3, c4, c5 = feats
        p3 = self.lateral[0](c3)
        p4 = self.lateral[1](c4)
        p5 = self.lateral[2](c5)

        # top-down
        p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p4 = self.smooth_td[0](p4)
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p3 = self.smooth_td[1](p3)

        # bottom-up
        p4 = p4 + self.downsample[0](p3)
        p4 = self.smooth_bu[0](p4)
        p5 = p5 + self.downsample[1](p4)
        p5 = self.smooth_bu[1](p5)

        return p3, p4, p5


class ScaleExp(nn.Module):
    """Learnable per-level scale, dùng cho bbox/kps regression (kiểu FCOS)."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


class SCRFDHead(nn.Module):
    """Head anchor-free dùng chung trọng số giữa các level P3/P4/P5."""

    def __init__(self, in_channels: int = 64, stacked_convs: int = 2, num_groups: int = 8,
                 num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes

        def make_stem():
            layers = []
            for _ in range(stacked_convs):
                layers += [
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
                    nn.GroupNorm(num_groups, in_channels),
                    nn.ReLU(inplace=True),
                ]
            return nn.Sequential(*layers)

        self.cls_stem = make_stem()
        self.bbox_stem = make_stem()
        self.kps_stem = make_stem()

        self.cls_pred = nn.Conv2d(in_channels, num_classes, 3, 1, 1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, 1, 1)
        self.kps_pred = nn.Conv2d(in_channels, NUM_KPS * 2, 3, 1, 1)

        self.bbox_scales = nn.ModuleList([ScaleExp(1.0) for _ in STRIDES])
        self.kps_scales = nn.ModuleList([ScaleExp(1.0) for _ in STRIDES])

        self._init_weights()

    def _init_weights(self):
        for m in [self.cls_pred, self.bbox_pred, self.kps_pred]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
        # bias khởi tạo cho cls để ổn định focal loss lúc đầu train (prior prob ~0.01)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)

    def forward_single(self, feat, level_idx: int):
        stride = STRIDES[level_idx]

        cls_feat = self.cls_stem(feat)
        cls_score = self.cls_pred(cls_feat)  # (B, 1, H, W), logit

        bbox_feat = self.bbox_stem(feat)
        bbox_dist = self.bbox_pred(bbox_feat)  # (B, 4, H, W) -> (l, t, r, b)
        bbox_dist = F.relu(self.bbox_scales[level_idx](bbox_dist)) * stride

        kps_feat = self.kps_stem(feat)
        kps_offset = self.kps_pred(kps_feat)  # (B, 8, H, W)
        kps_offset = self.kps_scales[level_idx](kps_offset) * stride

        return cls_score, bbox_dist, kps_offset

    def forward(self, feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        outputs = []
        for i, feat in enumerate(feats):
            outputs.append(self.forward_single(feat, i))
        return outputs  # list[(cls, bbox, kps)] theo thứ tự P3, P4, P5


class SCRFD_MBF(nn.Module):
    """Model tổng hợp: MBF backbone -> PAFPN -> SCRFD head (4 keypoint góc biển số)."""

    def __init__(self, width_mult: float = 1.0, fpn_channels: int = 64, num_classes: int = 2):
        super().__init__()
        self.backbone = MBFBackbone(width_mult=width_mult)
        self.neck = PAFPN(self.backbone.out_channels, out_channels=fpn_channels)
        self.head = SCRFDHead(in_channels=fpn_channels, num_classes=num_classes)
        self.strides = STRIDES
        self.num_classes = num_classes

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.head(feats)  # list[(cls_score, bbox_dist, kps_offset)] mỗi level

    @torch.no_grad()
    def decode(self, outputs, score_thr: float = 0.3):
        """Decode raw head outputs -> list[(boxes, scores, labels, kps)] theo batch.

        boxes: (N, 4) x1,y1,x2,y2 | scores: (N,) | labels: (N,) class id (argmax)
        kps: (N, 4, 2)
        Lưu ý: chưa NMS — áp dụng NMS (vd torchvision.ops.nms, per-class) sau bước này.
        """
        batch_size = outputs[0][0].shape[0]
        results = [[] for _ in range(batch_size)]

        for level_idx, (cls_score, bbox_dist, kps_offset) in enumerate(outputs):
            stride = self.strides[level_idx]
            b, num_classes, h, w = cls_score.shape

            cls_prob = cls_score.sigmoid().permute(0, 2, 3, 1).reshape(b, h * w, num_classes)
            scores, labels = cls_prob.max(dim=-1)  # (b, h*w) mỗi

            yv, xv = torch.meshgrid(
                torch.arange(h, device=cls_score.device),
                torch.arange(w, device=cls_score.device),
                indexing="ij",
            )
            px = (xv.reshape(-1).float() + 0.5) * stride
            py = (yv.reshape(-1).float() + 0.5) * stride

            bbox_dist = bbox_dist.permute(0, 2, 3, 1).reshape(b, h * w, 4)
            kps_offset = kps_offset.permute(0, 2, 3, 1).reshape(b, h * w, NUM_KPS * 2)

            x1 = px - bbox_dist[..., 0]
            y1 = py - bbox_dist[..., 1]
            x2 = px + bbox_dist[..., 2]
            y2 = py + bbox_dist[..., 3]
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (b, h*w, 4)

            kps = kps_offset.reshape(b, h * w, NUM_KPS, 2)
            kps = kps + torch.stack([px, py], dim=-1).reshape(1, h * w, 1, 2)

            for bi in range(b):
                mask = scores[bi] > score_thr
                if mask.any():
                    results[bi].append((
                        boxes[bi][mask], scores[bi][mask], labels[bi][mask], kps[bi][mask]
                    ))

        final = []
        for r in results:
            if not r:
                final.append((
                    torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long),
                    torch.zeros((0, NUM_KPS, 2))
                ))
                continue
            boxes = torch.cat([x[0] for x in r], dim=0)
            scores = torch.cat([x[1] for x in r], dim=0)
            labels = torch.cat([x[2] for x in r], dim=0)
            kps = torch.cat([x[3] for x in r], dim=0)
            final.append((boxes, scores, labels, kps))
        return final


def sort_corners(kps: torch.Tensor) -> torch.Tensor:
    """Sắp xếp lại 4 điểm góc theo thứ tự TL, TR, BR, BL dựa trên centroid + góc.

    kps: (N, 4, 2)
    """
    centroid = kps.mean(dim=1, keepdim=True)  # (N, 1, 2)
    vec = kps - centroid
    angles = torch.atan2(vec[..., 1], vec[..., 0])  # (N, 4)
    order = torch.argsort(angles, dim=1)
    sorted_kps = torch.gather(
        kps, 1, order.unsqueeze(-1).expand(-1, -1, 2)
    )
    return sorted_kps


if __name__ == "__main__":
    model = SCRFD_MBF(width_mult=1.0, fpn_channels=64, num_classes=2)
    dummy = torch.randn(1, 3, 640, 640)
    outs = model(dummy)
    for i, (cls_s, bbox_d, kps_o) in enumerate(outs):
        print(f"Level {i} (stride {STRIDES[i]}): cls={tuple(cls_s.shape)} "
              f"bbox={tuple(bbox_d.shape)} kps={tuple(kps_o.shape)}")

    decoded = model.decode(outs, score_thr=0.0)
    boxes, scores, labels, kps = decoded[0]
    print("Decoded boxes shape:", boxes.shape, "labels shape:", labels.shape)
