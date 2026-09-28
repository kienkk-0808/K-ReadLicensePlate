"""Chạy thử model (checkpoint .pt hoặc .onnx) trên 1 ảnh thật, vẽ box + 4 keypoint
góc biển số lên ảnh gốc và lưu kết quả ra file.

Ví dụ chạy:
    # Dùng checkpoint PyTorch
    python infer.py --image dataset/valid/images/<ten_anh>.jpg \
        --checkpoint runs/scrfd_mbf_fast/best.pt --output out.jpg

    # Dùng model đã export ONNX (khuyến nghị để test đúng cái sẽ deploy)
    python infer.py --image dataset/valid/images/<ten_anh>.jpg \
        --onnx runs/scrfd_mbf_fast/best.onnx --output out.jpg
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

from datasets.lp_yolo_pose_dataset import letterbox, CLASS_NAMES
from models.scrfd_mbf import SCRFD_MBF
from models.scrfd_utils import generate_points, flatten_head_outputs, decode_points

BOX_COLOR = (0, 200, 0)
CORNER_COLORS = [
    (0, 0, 255),    # index 0 - đỏ
    (0, 255, 0),    # index 1 - xanh lá
    (255, 0, 0),    # index 2 - xanh dương
    (0, 255, 255),  # index 3 - vàng
]


def parse_args():
    p = argparse.ArgumentParser(description="Chạy thử SCRFD_MBF trên 1 ảnh")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="", help="Đường dẫn .pt (PyTorch)")
    p.add_argument("--onnx", type=str, default="", help="Đường dẫn .onnx (onnxruntime)")
    p.add_argument("--output", type=str, default="", help="Mặc định: <image>_pred.jpg")
    p.add_argument("--img-size", type=int, default=0, help="0 = lấy theo checkpoint/mặc định 640")
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--warmup", type=int, default=2,
                    help="Số lần chạy 'đánh thức' model trước khi đo thời gian (không tính vào log)")
    p.add_argument("--iters", type=int, default=5,
                    help="Số lần lặp lại inference để đo thời gian xử lý trung bình")
    return p.parse_args()


def load_torch_model(checkpoint_path: str, img_size_override: int):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args = ckpt.get("args", {})
    img_size = img_size_override or args.get("img_size", 640)

    model = SCRFD_MBF(
        width_mult=args.get("width_mult", 1.0),
        fpn_channels=args.get("fpn_channels", 32),
        num_classes=args.get("num_classes", 1),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, img_size


def run_torch_inference(model, img_size, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        cls_logits, bbox_dist, kps_offset = flatten_head_outputs(outputs)
        scores = cls_logits.sigmoid()
        points, _, _ = generate_points(img_size)
        boxes, kps = decode_points(points, bbox_dist, kps_offset)
    return scores[0], boxes[0], kps[0]  # bỏ chiều batch (batch=1)


def load_onnx_session(onnx_path: str):
    import onnxruntime as ort
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def run_onnx_inference(sess, img_tensor: torch.Tensor):
    scores, boxes, kps = sess.run(None, {"image": img_tensor.numpy()})
    return torch.from_numpy(scores[0]), torch.from_numpy(boxes[0]), torch.from_numpy(kps[0])


def postprocess(scores, boxes, kps, score_thr, nms_iou):
    """scores: (N, num_classes) | boxes: (N,4) | kps: (N,4,2) -> sau threshold + NMS."""
    max_scores, labels = scores.max(dim=-1)
    keep_mask = max_scores > score_thr
    boxes, max_scores, labels, kps = boxes[keep_mask], max_scores[keep_mask], labels[keep_mask], kps[keep_mask]

    if boxes.shape[0] == 0:
        return boxes, max_scores, labels, kps

    keep = batched_nms(boxes, max_scores, labels, nms_iou)
    return boxes[keep], max_scores[keep], labels[keep], kps[keep]


def unletterbox_points(pts: np.ndarray, scale: float, pad: tuple) -> np.ndarray:
    pad_x, pad_y = pad
    out = pts.copy()
    out[..., 0] = (out[..., 0] - pad_x) / scale
    out[..., 1] = (out[..., 1] - pad_y) / scale
    return out


def draw_predictions(img_bgr, boxes, scores, labels, kps):
    for box, score, label, kp in zip(boxes, scores, labels, kps):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), BOX_COLOR, 2)

        text = f"{CLASS_NAMES[int(label)]} {score:.2f}"
        cv2.putText(img_bgr, text, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, BOX_COLOR, 2, cv2.LINE_AA)

        pts = kp.astype(int)
        for i in range(4):
            cv2.circle(img_bgr, tuple(pts[i]), 4, CORNER_COLORS[i], -1)
            cv2.line(img_bgr, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (255, 255, 255), 1)

    return img_bgr


def main():
    args = parse_args()
    if not args.checkpoint and not args.onnx:
        raise ValueError("Cần truyền --checkpoint hoặc --onnx")

    img_path = Path(args.image)
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_size = args.img_size or 640
    if args.checkpoint:
        model, img_size = load_torch_model(args.checkpoint, args.img_size)
    else:
        sess = load_onnx_session(args.onnx)
        img_size = args.img_size or img_size

    canvas, scale, pad = letterbox(img_rgb, img_size)
    img_tensor = torch.from_numpy(canvas).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    def infer_once():
        if args.checkpoint:
            return run_torch_inference(model, img_size, img_tensor)
        return run_onnx_inference(sess, img_tensor)

    # "Đánh thức" model: chạy vài lần đầu không tính thời gian (cudnn autotune,
    # onnxruntime lazy init phiên đầu, cache warm...) để số đo sau đó phản ánh
    # đúng tốc độ inference ổn định, không lẫn overhead khởi động.
    for _ in range(max(args.warmup, 0)):
        infer_once()
    print(f"[infer] đã đánh thức model ({args.warmup} lần chạy khởi động, không tính thời gian)")

    latencies_ms = []
    scores = boxes = kps = None
    for _ in range(max(args.iters, 1)):
        t0 = time.perf_counter()
        scores, boxes, kps = infer_once()
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(latencies_ms) / len(latencies_ms)
    print(
        f"[infer] thời gian xử lý: avg={avg_ms:.2f}ms "
        f"min={min(latencies_ms):.2f}ms max={max(latencies_ms):.2f}ms "
        f"(trung bình {len(latencies_ms)} lần, ảnh {img_size}x{img_size})"
    )

    boxes, scores, labels, kps = postprocess(scores, boxes, kps, args.score_thr, args.nms_iou)
    print(f"[infer] phát hiện {boxes.shape[0]} biển số (score > {args.score_thr})")

    boxes_np = unletterbox_points(boxes.numpy().reshape(-1, 2, 2), scale, pad).reshape(-1, 4)
    kps_np = unletterbox_points(kps.numpy(), scale, pad)

    for i, (b, s, l) in enumerate(zip(boxes_np, scores.numpy(), labels.numpy())):
        print(f"  #{i}: {CLASS_NAMES[int(l)]} score={s:.3f} box={b.round(1).tolist()}")

    result_img = draw_predictions(img_bgr.copy(), boxes_np, scores.numpy(), labels.numpy(), kps_np)

    output_path = Path(args.output) if args.output else img_path.with_name(img_path.stem + "_pred.jpg")
    cv2.imwrite(str(output_path), result_img)
    print(f"[infer] đã lưu kết quả: {output_path}")


if __name__ == "__main__":
    main()
