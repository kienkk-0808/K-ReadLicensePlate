"""SCRFD-MBF: backbone MobileFaceNet-style + head anchor-free (bbox only).

Kiến trúc: xem docs/architecture/SCRFD-MBF-LP-4KPS.md
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDES = (8, 16, 32)


def conv_bn_prelu(in_c, out_c, kernel=3, stride=1, groups=1):
    padding = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_c),
        nn.PReLU(out_c),
    )


class InvertedResidual(nn.Module):
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
    def __init__(self, width_mult: float = 1.0):
        super().__init__()

        def c(ch):
            return max(8, int(round(ch * width_mult / 8) * 8))

        self.stem = conv_bn_prelu(3, c(16), kernel=3, stride=2)
        self.dw_stem = conv_bn_prelu(c(16), c(16), kernel=3, stride=2, groups=c(16))

        self.stage1 = self._make_stage(c(16), c(32), stride=2, expand=2, n=2)
        self.stage2 = self._make_stage(c(32), c(64), stride=2, expand=2, n=3)
        self.stage3 = self._make_stage(c(64), c(64), stride=2, expand=4, n=2)

        self.out_channels = (c(32), c(64), c(64))
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
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        return c3, c4, c5


class PAFPN(nn.Module):
    def __init__(self, in_channels: Tuple[int, int, int], out_channels: int = 48):
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

        p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p4 = self.smooth_td[0](p4)
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p3 = self.smooth_td[1](p3)

        p4 = p4 + self.downsample[0](p3)
        p4 = self.smooth_bu[0](p4)
        p5 = p5 + self.downsample[1](p4)
        p5 = self.smooth_bu[1](p5)

        return p3, p4, p5


def dfl_to_distance(bbox_logits: torch.Tensor, reg_max: int) -> torch.Tensor:
    """bbox_logits(...,4*(reg_max+1)) raw -> (...,4) khoảng cách kỳ vọng, đơn vị stride.

    DFL (Distribution Focal Loss, YOLOv8/GFocal): thay vì hồi quy 1 số thực cho mỗi
    cạnh (l,t,r,b), dự đoán 1 phân phối rời rạc trên reg_max+1 bin rồi lấy kỳ vọng
    (expected value) — cho độ chính xác sub-pixel tốt hơn hồi quy trực tiếp.
    """
    shape = bbox_logits.shape[:-1]
    x = bbox_logits.reshape(*shape, 4, reg_max + 1)
    prob = F.softmax(x, dim=-1)
    bins = torch.arange(reg_max + 1, dtype=prob.dtype, device=prob.device)
    return (prob * bins).sum(dim=-1)


class SCRFDHead(nn.Module):
    def __init__(self, in_channels: int = 48, stacked_convs: int = 2, num_groups: int = 8,
                 num_classes: int = 1, reg_max: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

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

        self.cls_pred = nn.Conv2d(in_channels, num_classes, 3, 1, 1)
        self.bbox_pred = nn.Conv2d(in_channels, 4 * (reg_max + 1), 3, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.cls_pred, self.bbox_pred]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
        prior_prob = 0.01
        nn.init.constant_(self.cls_pred.bias, -math.log((1 - prior_prob) / prior_prob))

    def forward_single(self, feat):
        cls_score = self.cls_pred(self.cls_stem(feat))
        bbox_logits = self.bbox_pred(self.bbox_stem(feat))
        return cls_score, bbox_logits

    def forward(self, feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        return [self.forward_single(feat) for feat in feats]


class SCRFD_MBF(nn.Module):
    def __init__(self, width_mult: float = 1.0, fpn_channels: int = 48, num_classes: int = 1,
                 stacked_convs: int = 2, reg_max: int = 16):
        super().__init__()
        self.backbone = MBFBackbone(width_mult=width_mult)
        self.neck = PAFPN(self.backbone.out_channels, out_channels=fpn_channels)
        self.head = SCRFDHead(in_channels=fpn_channels, num_classes=num_classes,
                               stacked_convs=stacked_convs, reg_max=reg_max)
        self.strides = STRIDES
        self.num_classes = num_classes
        self.reg_max = reg_max

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.head(feats)

    @torch.no_grad()
    def decode(self, outputs, score_thr: float = 0.3):
        """-> list[(boxes[N,4], scores[N], labels[N])] theo batch. Chưa NMS."""
        batch_size = outputs[0][0].shape[0]
        results = [[] for _ in range(batch_size)]

        for level_idx, (cls_score, bbox_logits) in enumerate(outputs):
            stride = self.strides[level_idx]
            b, num_classes, h, w = cls_score.shape

            cls_prob = cls_score.sigmoid().permute(0, 2, 3, 1).reshape(b, h * w, num_classes)
            scores, labels = cls_prob.max(dim=-1)

            yv, xv = torch.meshgrid(
                torch.arange(h, device=cls_score.device),
                torch.arange(w, device=cls_score.device),
                indexing="ij",
            )
            px = (xv.reshape(-1).float() + 0.5) * stride
            py = (yv.reshape(-1).float() + 0.5) * stride

            bbox_logits = bbox_logits.permute(0, 2, 3, 1).reshape(b, h * w, 4 * (self.reg_max + 1))
            bbox_dist = dfl_to_distance(bbox_logits, self.reg_max) * stride

            x1 = px - bbox_dist[..., 0]
            y1 = py - bbox_dist[..., 1]
            x2 = px + bbox_dist[..., 2]
            y2 = py + bbox_dist[..., 3]
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)

            for bi in range(b):
                mask = scores[bi] > score_thr
                if mask.any():
                    results[bi].append((boxes[bi][mask], scores[bi][mask], labels[bi][mask]))

        final = []
        for r in results:
            if not r:
                final.append((
                    torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)
                ))
                continue
            boxes = torch.cat([x[0] for x in r], dim=0)
            scores = torch.cat([x[1] for x in r], dim=0)
            labels = torch.cat([x[2] for x in r], dim=0)
            final.append((boxes, scores, labels))
        return final


if __name__ == "__main__":
    model = SCRFD_MBF(width_mult=1.0, fpn_channels=48, num_classes=1, stacked_convs=2)
    dummy = torch.randn(1, 3, 640, 640)
    outs = model(dummy)
    for i, (cls_s, bbox_d) in enumerate(outs):
        print(f"Level {i} (stride {STRIDES[i]}): cls={tuple(cls_s.shape)} bbox={tuple(bbox_d.shape)}")

    decoded = model.decode(outs, score_thr=0.0)
    boxes, scores, labels = decoded[0]
    print("Decoded boxes shape:", boxes.shape, "labels shape:", labels.shape)
