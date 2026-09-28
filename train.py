"""Training script cho SCRFD_MBF trên dataset biển số (YOLO-Pose format, xem dataset/data.yaml).

Ví dụ chạy:
    python train.py --data-root dataset --epochs 50 --batch-size 8 --img-size 640

Chiến lược "elitist" (mặc định bật, tắt bằng --no-elitist):
    Sau mỗi epoch, đánh giá mAP@0.5 + kps_nme thật (models/metrics.py, có NMS +
    IoU-matching, không chỉ dựa vào loss) trên tập valid.

    mAP@0.5 CHỈ đo độ chính xác box (IoU>=0.5 + đúng class) — hoàn toàn không nhìn
    vào keypoint. Nếu chỉ dùng mAP50 để chọn best, 1 epoch có box tốt nhưng nhánh
    keypoint chưa hội tụ (kps_nme cao) vẫn có thể được phong "best" — không phải lỗi
    lý thuyết, đã quan sát thực tế: checkpoint mAP50=0.85 nhưng vẽ ra 4 góc bị lệch
    hẳn so với biển số thật. Vì vậy dùng composite_score để chọn best:

        composite_score = mAP50 - kps_weight * min(kps_nme, 1.0)   (--kps-weight, mặc định 0.5)

    - Nếu composite_score CAO HƠN best hiện tại -> lưu làm best.pt, tiếp tục train
      bình thường từ trọng số hiện tại (đang là best).
    - Nếu KHÔNG cải thiện -> nạp lại trọng số + optimizer state từ best.pt trước khi
      bắt đầu epoch kế tiếp ("lấy best ra train tiếp"), tránh việc 1 epoch tệ (do
      augmentation ngẫu nhiên xấu, LR nhảy...) kéo lùi cả quá trình học trên tập dữ
      liệu nhỏ (~1000 ảnh). LR scheduler vẫn tiến bình thường theo epoch, không bị
      reset khi revert.
    Lưu ý: chiến lược này đánh đổi tốc độ hội tụ lấy sự ổn định — phù hợp dataset nhỏ,
    dễ overfit/nhiễu; với dataset lớn hơn nhiều có thể tắt (--no-elitist) để train
    theo kiểu thông thường (luôn tiếp tục từ epoch vừa train xong).
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.lp_yolo_pose_dataset import LicensePlateYoloPoseDataset, collate_fn
from models.scrfd_mbf import SCRFD_MBF
from models.losses import SCRFDLoss
from models.metrics import evaluate_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Train SCRFD_MBF cho phát hiện biển số")
    p.add_argument("--data-root", type=str, default="dataset")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--fpn-channels", type=int, default=32)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default="runs/scrfd_mbf")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--val-interval", type=int, default=1)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--no-augment", action="store_true", help="Tắt augmentation cho tập train")
    p.add_argument("--no-elitist", action="store_true",
                    help="Tắt chiến lược revert-to-best sau mỗi epoch không cải thiện")
    p.add_argument("--map-iou", type=float, default=0.5, help="Ngưỡng IoU dùng để tính mAP")
    p.add_argument("--nms-iou", type=float, default=0.5, help="Ngưỡng IoU dùng để NMS lúc eval")
    p.add_argument("--kps-weight", type=float, default=0.5,
                    help="Trọng số phạt kps_nme khi chọn best checkpoint: "
                         "composite = mAP50 - kps_weight * min(kps_nme, 1.0)")
    p.add_argument("--lambda-cls", type=float, default=1.0, help="Trọng số cls_loss trong tổng loss")
    p.add_argument("--lambda-bbox", type=float, default=1.0, help="Trọng số bbox_loss trong tổng loss")
    p.add_argument("--lambda-kps", type=float, default=2.0,
                    help="Trọng số kps_loss trong tổng loss — tăng lên nếu keypoint hội tụ "
                         "chậm hơn box (quan sát thực tế: mAP cao nhưng kps_nme vẫn cao)")
    return p.parse_args()


def build_dataloaders(args):
    train_ds = LicensePlateYoloPoseDataset(
        root=args.data_root, split="train", img_size=args.img_size,
        augment=not args.no_augment,
    )
    val_ds = LicensePlateYoloPoseDataset(
        root=args.data_root, split="valid", img_size=args.img_size, augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loss_fn, val_loader, device):
    model.eval()
    totals = {"loss": 0.0, "cls_loss": 0.0, "bbox_loss": 0.0, "kps_loss": 0.0}
    num_batches = 0

    for imgs, targets in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        loss_dict = loss_fn(outputs, targets)
        for k in totals:
            totals[k] += loss_dict[k].item()
        num_batches += 1

    model.train()
    if num_batches == 0:
        return totals
    return {k: v / num_batches for k, v in totals.items()}


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(args)
    print(f"[data] train batches/epoch: {len(train_loader)} | val batches: {len(val_loader)}")

    model = SCRFD_MBF(
        width_mult=args.width_mult, fpn_channels=args.fpn_channels,
        num_classes=args.num_classes,
    ).to(device)

    loss_fn = SCRFDLoss(
        img_size=args.img_size, num_classes=args.num_classes,
        lambda_cls=args.lambda_cls, lambda_bbox=args.lambda_bbox, lambda_kps=args.lambda_kps,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_map = -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_map = ckpt.get("best_map", -1.0)
        print(f"[resume] tiếp tục từ epoch {start_epoch}, best_map hiện tại={best_map:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        running = {"loss": 0.0, "cls_loss": 0.0, "bbox_loss": 0.0, "kps_loss": 0.0}

        for step, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)

            outputs = model(imgs)
            loss_dict = loss_fn(outputs, targets)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
            optimizer.step()

            for k in running:
                running[k] += loss_dict[k].item()

            if (step + 1) % args.log_interval == 0:
                n = args.log_interval
                print(
                    f"[epoch {epoch}] step {step + 1}/{len(train_loader)} "
                    f"loss={running['loss'] / n:.4f} "
                    f"cls={running['cls_loss'] / n:.4f} "
                    f"bbox={running['bbox_loss'] / n:.4f} "
                    f"kps={running['kps_loss'] / n:.4f} "
                    f"num_pos={loss_dict['num_pos'].item():.0f}"
                )
                running = {k: 0.0 for k in running}

        scheduler.step()
        elapsed = time.time() - epoch_start
        print(f"[epoch {epoch}] xong sau {elapsed:.1f}s, lr={scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % args.val_interval == 0:
            val_loss_metrics = evaluate(model, loss_fn, val_loader, device)
            det_metrics = evaluate_metrics(
                model, val_loader, device,
                nms_iou=args.nms_iou, map_iou=args.map_iou, num_classes=args.num_classes,
            )
            # mAP50 chỉ đo box (IoU>=map_iou + đúng class) -> KHÔNG phản ánh độ chính
            # xác keypoint. Dùng composite_score để chọn best, tránh trường hợp 1 epoch
            # có box tốt nhưng keypoint tệ (kps_nme cao) vẫn được phong "best".
            kps_nme = det_metrics["kps_nme"]
            kps_penalty = 1.0 if kps_nme == float("inf") else min(kps_nme, 1.0)
            composite_score = det_metrics["mAP50"] - args.kps_weight * kps_penalty
            det_metrics["composite_score"] = composite_score

            print(
                f"[epoch {epoch}] VAL loss={val_loss_metrics['loss']:.4f} "
                f"| mAP@{args.map_iou:.2f}={det_metrics['mAP50']:.4f} "
                f"kps_nme={kps_nme:.4f} composite={composite_score:.4f}"
            )

            last_ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_map": best_map,
                "val_loss": val_loss_metrics["loss"],
                "det_metrics": det_metrics,
                "args": vars(args),
            }
            torch.save(last_ckpt, output_dir / "last.pt")

            if composite_score > best_map:
                best_map = composite_score
                last_ckpt["best_map"] = best_map
                torch.save(last_ckpt, output_dir / "best.pt")
                print(
                    f"[epoch {epoch}] best model moi, composite={best_map:.4f} "
                    f"(mAP@{args.map_iou:.2f}={det_metrics['mAP50']:.4f}, kps_nme={kps_nme:.4f})"
                )
            elif not args.no_elitist:
                # Epoch này không cải thiện composite_score -> nạp lại trọng số + optimizer
                # từ best.pt trước khi bước sang epoch kế tiếp ("lấy best ra train tiếp").
                best_path = output_dir / "best.pt"
                if best_path.exists():
                    best_ckpt = torch.load(best_path, map_location=device)
                    model.load_state_dict(best_ckpt["model"])
                    optimizer.load_state_dict(best_ckpt["optimizer"])
                    print(
                        f"[epoch {epoch}] khong cai thien (composite={composite_score:.4f} "
                        f"<= best={best_map:.4f}) -> nap lai trong so best.pt cho epoch sau"
                    )

    print("Huấn luyện hoàn tất.")


if __name__ == "__main__":
    main()
