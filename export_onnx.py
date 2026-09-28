"""Export SCRFD_MBF (checkpoint .pt từ train.py) sang ONNX để triển khai edge/embedded.

Model wrapper bọc sẵn bước decode (sigmoid cls + quy đổi l,t,r,b / dx,dy về pixel
tuyệt đối) vào trong đồ thị ONNX — runtime triển khai (C++/NPU SDK) chỉ cần
threshold theo score rồi NMS, không phải cài lại công thức decode.

Ví dụ chạy:
    python export_onnx.py --checkpoint runs/scrfd_mbf_fast/best.pt --output runs/scrfd_mbf_fast/model.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.scrfd_mbf import SCRFD_MBF
from models.scrfd_utils import generate_points, flatten_head_outputs, decode_points


class SCRFDONNXWrapper(nn.Module):
    """Input: ảnh (1,3,H,W) float32, giá trị [0,1] (RGB, đã letterbox sẵn ở tiền xử lý).
    Output:
        scores: (1, N, num_classes) — đã sigmoid, N = tổng anchor point 3 level.
        boxes:  (1, N, 4) — x1,y1,x2,y2 pixel, cùng hệ toạ độ với ảnh input (H,W).
        kps:    (1, N, 4, 2) — 4 góc biển số, pixel, cùng hệ toạ độ ảnh input.
    """

    def __init__(self, model: SCRFD_MBF, img_size: int):
        super().__init__()
        self.model = model
        points, _, _ = generate_points(img_size)
        self.register_buffer("points", points)

    def forward(self, x):
        outputs = self.model(x)
        cls_logits, bbox_dist, kps_offset = flatten_head_outputs(outputs)
        scores = cls_logits.sigmoid()
        boxes, kps = decode_points(self.points, bbox_dist, kps_offset)
        return scores, boxes, kps


def parse_args():
    p = argparse.ArgumentParser(description="Export SCRFD_MBF sang ONNX")
    p.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn best.pt/last.pt")
    p.add_argument("--output", type=str, default="", help="Đường dẫn .onnx xuất ra (mặc định cùng thư mục checkpoint)")
    p.add_argument("--img-size", type=int, default=0, help="0 = lấy theo args lưu trong checkpoint")
    p.add_argument("--width-mult", type=float, default=0.0, help="0 = lấy theo checkpoint")
    p.add_argument("--fpn-channels", type=int, default=0, help="0 = lấy theo checkpoint")
    p.add_argument("--num-classes", type=int, default=0, help="0 = lấy theo checkpoint")
    p.add_argument("--opset", type=int, default=18,
                    help="Op Resize (upsample trong PAFPN) yêu cầu tối thiểu opset 18")
    p.add_argument("--dynamic-batch", action="store_true", help="Cho phép batch size động (mặc định cố định batch=1)")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    img_size = args.img_size or ckpt_args.get("img_size", 640)
    width_mult = args.width_mult or ckpt_args.get("width_mult", 1.0)
    fpn_channels = args.fpn_channels or ckpt_args.get("fpn_channels", 32)
    num_classes = args.num_classes or ckpt_args.get("num_classes", 1)

    print(f"[config] img_size={img_size} width_mult={width_mult} "
          f"fpn_channels={fpn_channels} num_classes={num_classes}")
    if "det_metrics" in ckpt:
        print(f"[checkpoint] epoch={ckpt.get('epoch')} det_metrics={ckpt['det_metrics']}")

    model = SCRFD_MBF(width_mult=width_mult, fpn_channels=fpn_channels, num_classes=num_classes)
    model.load_state_dict(ckpt["model"])
    model.eval()

    wrapper = SCRFDONNXWrapper(model, img_size=img_size)
    wrapper.eval()

    dummy = torch.randn(1, 3, img_size, img_size)

    output_path = Path(args.output) if args.output else Path(args.checkpoint).with_suffix(".onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "image": {0: "batch"},
            "scores": {0: "batch"},
            "boxes": {0: "batch"},
            "kps": {0: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, str(output_path),
            input_names=["image"], output_names=["scores", "boxes", "kps"],
            opset_version=args.opset, dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    _merge_external_data_into_single_file(output_path)
    print(f"[export] đã ghi {output_path} (1 file duy nhất, không tách weight riêng)")

    verify(wrapper, output_path, dummy)


def _merge_external_data_into_single_file(output_path: Path):
    """torch.onnx.export (dynamo exporter) mặc định tách weight lớn ra file
    '<name>.onnx.data' riêng (external data). Gộp lại thành đúng 1 file .onnx để
    dễ đóng gói/deploy — nạp lại toàn bộ weight vào bộ nhớ rồi ghi đè ở dạng embed.
    """
    import onnx

    data_file = output_path.with_name(output_path.name + ".data")
    model = onnx.load(str(output_path), load_external_data=True)
    onnx.save(model, str(output_path), save_as_external_data=False)
    if data_file.exists():
        data_file.unlink()


def verify(wrapper: nn.Module, onnx_path: Path, dummy: torch.Tensor):
    """Kiểm tra đồ thị hợp lệ (onnx.checker) + so khớp số học với PyTorch (onnxruntime)."""
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("[verify] onnx.checker.check_model: OK")

    with torch.no_grad():
        torch_out = wrapper(dummy)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"image": dummy.numpy()})

    names = ["scores", "boxes", "kps"]
    all_close = True
    for name, t_out, o_out in zip(names, torch_out, ort_out):
        t_np = t_out.numpy()
        max_diff = np.abs(t_np - o_out).max()
        ok = np.allclose(t_np, o_out, atol=1e-4, rtol=1e-3)
        all_close = all_close and ok
        print(f"[verify] {name}: shape={o_out.shape} max_diff={max_diff:.6f} match={ok}")

    if all_close:
        print("[verify] PyTorch vs ONNXRuntime khớp số học -> export thành công.")
    else:
        print("[verify] CẢNH BÁO: sai lệch vượt ngưỡng, cần kiểm tra lại.")


if __name__ == "__main__":
    main()
