"""Augmentation cho ảnh biển số, áp dụng TRƯỚC letterbox."""

import random
from typing import Tuple

import cv2
import numpy as np


def augment_hsv(img: np.ndarray, hgain: float = 0.015, sgain: float = 0.7,
                 vgain: float = 0.4) -> np.ndarray:
    if not (hgain or sgain or vgain):
        return img
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    dtype = img.dtype

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((
        cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)
    ))
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


def random_flip_lr(img: np.ndarray, boxes: np.ndarray,
                    p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() >= p:
        return img, boxes

    w = img.shape[1]
    img = np.ascontiguousarray(img[:, ::-1])

    if boxes.shape[0]:
        x1 = boxes[:, 0].copy()
        boxes[:, 0] = w - boxes[:, 2]
        boxes[:, 2] = w - x1
    return img, boxes


def _transform_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    shape = pts.shape
    flat = pts.reshape(-1, 2).astype(np.float32)
    ones = np.ones((flat.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([flat, ones], axis=1)
    out = (M @ homo.T).T[:, :2]
    return out.reshape(shape)


def random_affine(
    img: np.ndarray, boxes: np.ndarray,
    degrees: float = 10.0, scale: Tuple[float, float] = (0.75, 1.25),
    translate: float = 0.10, border_value=(114, 114, 114),
    min_area_ratio: float = 0.4, min_size: float = 4.0,
):
    """-> img, boxes, keep_mask (loại object bị cắt mất phần lớn sau transform)."""
    h, w = img.shape[:2]

    center = np.eye(3, dtype=np.float32)
    center[0, 2] = -w / 2
    center[1, 2] = -h / 2

    angle = random.uniform(-degrees, degrees)
    scale_f = random.uniform(*scale)
    R = np.eye(3, dtype=np.float32)
    R[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale_f)

    T = np.eye(3, dtype=np.float32)
    T[0, 2] = random.uniform(-translate, translate) * w
    T[1, 2] = random.uniform(-translate, translate) * h

    back = np.eye(3, dtype=np.float32)
    back[0, 2] = w / 2
    back[1, 2] = h / 2

    M = back @ T @ R @ center

    img_out = cv2.warpAffine(img, M[:2], dsize=(w, h), borderValue=border_value)

    n = boxes.shape[0]
    if n == 0:
        return img_out, boxes, np.ones((0,), dtype=bool)

    corners = np.zeros((n, 4, 2), dtype=np.float32)
    corners[:, 0] = boxes[:, [0, 1]]
    corners[:, 1] = boxes[:, [2, 1]]
    corners[:, 2] = boxes[:, [2, 3]]
    corners[:, 3] = boxes[:, [0, 3]]

    corners_t = _transform_points(corners, M)
    new_boxes = np.concatenate([corners_t.min(axis=1), corners_t.max(axis=1)], axis=1)

    new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clip(0, w)
    new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clip(0, h)

    orig_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    new_w = new_boxes[:, 2] - new_boxes[:, 0]
    new_h = new_boxes[:, 3] - new_boxes[:, 1]
    new_area = new_w * new_h

    keep = (
        (new_area / (orig_area + 1e-6) > min_area_ratio)
        & (new_w > min_size) & (new_h > min_size)
    )
    return img_out, new_boxes, keep


def random_jpeg_compression(img: np.ndarray, p: float = 0.3,
                             quality_range: Tuple[int, int] = (25, 70)) -> np.ndarray:
    if random.random() >= p:
        return img
    quality = random.randint(*quality_range)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def random_gaussian_noise(img: np.ndarray, p: float = 0.3,
                           sigma_range: Tuple[float, float] = (3.0, 15.0)) -> np.ndarray:
    if random.random() >= p:
        return img
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(img.dtype)


def random_motion_blur(img: np.ndarray, p: float = 0.25,
                        kernel_range: Tuple[int, int] = (3, 9)) -> np.ndarray:
    if random.random() >= p:
        return img
    k = random.randrange(kernel_range[0], kernel_range[1] + 1, 2)
    angle = random.uniform(0, 180)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def random_gamma(img: np.ndarray, p: float = 0.4,
                  gamma_range: Tuple[float, float] = (0.5, 1.8)) -> np.ndarray:
    if random.random() >= p:
        return img
    gamma = random.uniform(*gamma_range)
    inv_gamma = 1.0 / gamma
    table = ((np.arange(0, 256) / 255.0) ** inv_gamma * 255).astype(np.uint8)
    return cv2.LUT(img, table)


def random_downscale_upscale(img: np.ndarray, p: float = 0.25,
                              scale_range: Tuple[float, float] = (0.35, 0.7)) -> np.ndarray:
    if random.random() >= p:
        return img
    h, w = img.shape[:2]
    s = random.uniform(*scale_range)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def train_augment(
    img: np.ndarray, boxes: np.ndarray,
    flip_p: float = 0.5, degrees: float = 10.0, scale: Tuple[float, float] = (0.75, 1.25),
    translate: float = 0.10, hsv: Tuple[float, float, float] = (0.02, 0.8, 0.6),
    domain_robust: bool = True,
):
    """domain_robust=True: thêm JPEG/nhiễu/mờ/gamma/resize để mô phỏng camera thật."""
    img, boxes = random_flip_lr(img, boxes, p=flip_p)
    img, boxes, keep = random_affine(
        img, boxes, degrees=degrees, scale=scale, translate=translate,
    )
    img = augment_hsv(img, *hsv)

    if domain_robust:
        img = random_gamma(img)
        img = random_motion_blur(img)
        img = random_downscale_upscale(img)
        img = random_gaussian_noise(img)
        img = random_jpeg_compression(img)

    if boxes.shape[0]:
        boxes = boxes[keep]

    return img, boxes, keep
