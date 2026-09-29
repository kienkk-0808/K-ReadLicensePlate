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

    - Nếu composite_score CAO HƠN best hiện tại -> lưu làm best.pt, reset bộ đếm
      "số epoch chưa cải thiện" về 0, tiếp tục train bình thường.
    - Nếu KHÔNG cải thiện -> KHÔNG revert ngay. Chỉ sau khi đủ `--patience` (mặc
      định 5) epoch LIÊN TIẾP không cải thiện mới nạp lại trọng số + optimizer từ
      best.pt ("lấy best ra train tiếp"). Lý do đổi từ revert-ngay-lập-tức sang có
      patience: tập valid chỉ ~165 ảnh, mAP có thể dao động do nhiễu (augmentation,
      NMS biên) chứ chưa chắc là plateau thật — revert ngay mỗi epoch không cải
      thiện dễ khiến optimizer bị "đóng băng" quanh 1 điểm hội tụ sớm, không có cơ
      hội đi xuyên qua nhiễu ngắn hạn để tìm điểm tốt hơn (quan sát thực tế: 1 lần
      train 70 epoch với revert-ngay, best rơi ở epoch 38, 31 epoch sau đó không hề
      cải thiện — dấu hiệu bị kẹt tại cực trị cục bộ do revert quá sớm/quá thường xuyên).
    - `composite_ema` (làm mượt theo `--ema-alpha`) được ghi vào metrics.csv chỉ để
      QUAN SÁT xu hướng thật khi vẽ đồ thị — không dùng để quyết định best/revert.
    LR scheduler vẫn tiến bình thường theo epoch, không bị reset khi revert.
    Lưu ý: chiến lược này đánh đổi tốc độ hội tụ lấy sự ổn định — phù hợp dataset nhỏ,
    dễ overfit/nhiễu; với dataset lớn hơn nhiều có thể tắt (--no-elitist) để train
    theo kiểu thông thường (luôn tiếp tục từ epoch vừa train xong).
"""

import argparse
import csv
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
    p.add_argument("--num-classes", type=int, default=1,
                    help="1 = gộp mọi loại biển thành 1 class (mặc định, tối ưu model, "
                         "chỉ cần bbox+kps); đổi lại 2 nếu muốn phân biệt plate-1-line/plate-2-line")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default="runs/scrfd_mbf")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--val-interval", type=int, default=1)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--no-augment", action="store_true", help="Tắt augmentation cho tập train")
    p.add_argument("--no-elitist", action="store_true",
                    help="Tắt chiến lược revert-to-best sau mỗi epoch không cải thiện")
    p.add_argument("--patience", type=int, default=5,
                    help="Số epoch liên tiếp KHÔNG cải thiện composite_score trước khi revert "
                         "về best.pt (thay vì revert ngay lập tức) — chống nhiễu từ tập valid nhỏ")
    p.add_argument("--ema-alpha", type=float, default=0.3,
                    help="Hệ số EMA làm mượt composite_score khi ghi log CSV (chỉ để quan sát xu "
                         "hướng thật, không dùng để quyết định best/revert)")
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

    metrics_csv_path = output_dir / "metrics.csv"
    csv_fieldnames = [
        "epoch", "lr", "train_loss", "train_cls", "train_bbox", "train_kps",
        "val_loss", "mAP50", "kps_nme", "composite_score", "composite_ema",
        "is_best", "reverted", "epochs_since_improve",
    ]
    # Nếu resume, giữ nguyên log cũ (append); nếu train mới, ghi đè + viết header.
    if not args.resume or not metrics_csv_path.exists():
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=csv_fieldnames).writeheader()

    def log_epoch_csv(row: dict):
        with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=csv_fieldnames).writerow(row)

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
    epochs_since_improve = 0
    composite_ema = None

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
        epoch_totals = {"loss": 0.0, "cls_loss": 0.0, "bbox_loss": 0.0, "kps_loss": 0.0}

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
                epoch_totals[k] += loss_dict[k].item()

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
        cur_lr = scheduler.get_last_lr()[0]
        print(f"[epoch {epoch}] xong sau {elapsed:.1f}s, lr={cur_lr:.6f}")

        num_steps = len(train_loader)
        epoch_avg = {k: v / num_steps for k, v in epoch_totals.items()}

        did_eval = (epoch + 1) % args.val_interval == 0
        if not did_eval:
            log_epoch_csv({
                "epoch": epoch, "lr": cur_lr,
                "train_loss": epoch_avg["loss"], "train_cls": epoch_avg["cls_loss"],
                "train_bbox": epoch_avg["bbox_loss"], "train_kps": epoch_avg["kps_loss"],
                "val_loss": "", "mAP50": "", "kps_nme": "", "composite_score": "",
                "composite_ema": "", "is_best": "", "reverted": "", "epochs_since_improve": "",
            })

        if did_eval:
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

            composite_ema = (
                composite_score if composite_ema is None
                else args.ema_alpha * composite_score + (1 - args.ema_alpha) * composite_ema
            )

            print(
                f"[epoch {epoch}] VAL loss={val_loss_metrics['loss']:.4f} "
                f"| mAP@{args.map_iou:.2f}={det_metrics['mAP50']:.4f} "
                f"kps_nme={kps_nme:.4f} composite={composite_score:.4f} "
                f"(ema={composite_ema:.4f})"
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

            is_best = composite_score > best_map
            reverted = False

            if is_best:
                best_map = composite_score
                epochs_since_improve = 0
                last_ckpt["best_map"] = best_map
                torch.save(last_ckpt, output_dir / "best.pt")
                print(
                    f"[epoch {epoch}] best model moi, composite={best_map:.4f} "
                    f"(mAP@{args.map_iou:.2f}={det_metrics['mAP50']:.4f}, kps_nme={kps_nme:.4f})"
                )
            else:
                epochs_since_improve += 1
                print(
                    f"[epoch {epoch}] khong cai thien (composite={composite_score:.4f} "
                    f"<= best={best_map:.4f}) — {epochs_since_improve}/{args.patience} epoch chua cai thien"
                )
                if not args.no_elitist and epochs_since_improve >= args.patience:
                    # Đủ `patience` epoch liên tiếp không cải thiện -> mới thực sự coi là
                    # plateau (không phải nhiễu 1 epoch từ tập valid nhỏ) -> nạp lại trọng
                    # số + optimizer từ best.pt rồi reset bộ đếm ("lấy best ra train tiếp").
                    best_path = output_dir / "best.pt"
                    if best_path.exists():
                        # Lưu lại LR hiện tại của scheduler TRƯỚC khi nạp optimizer từ
                        # best.pt. CosineAnnealingLR của PyTorch tính LR epoch sau bằng
                        # công thức "chainable" (nhân tỉ lệ với LR ĐANG NẰM TRONG optimizer),
                        # không phải hàm độc lập theo epoch — nếu để optimizer.load_state_dict
                        # nạp đè LR cũ (của epoch lúc best.pt được lưu) vào optimizer, lần
                        # scheduler.step() kế tiếp sẽ tính nhân dựa trên LR bị "lùi ngược" đó,
                        # có thể khiến LR tăng ngược hoặc lệch khỏi đường cong giảm dần dự kiến.
                        current_lr = scheduler.get_last_lr()
                        best_ckpt = torch.load(best_path, map_location=device)
                        model.load_state_dict(best_ckpt["model"])
                        optimizer.load_state_dict(best_ckpt["optimizer"])
                        for group, lr in zip(optimizer.param_groups, current_lr):
                            group["lr"] = lr
                        reverted = True
                        epochs_since_improve = 0
                        print(
                            f"[epoch {epoch}] du {args.patience} epoch khong cai thien "
                            f"-> nap lai trong so best.pt cho epoch sau (giu nguyen lr={current_lr[0]:.6f})"
                        )

            log_epoch_csv({
                "epoch": epoch, "lr": cur_lr,
                "train_loss": epoch_avg["loss"], "train_cls": epoch_avg["cls_loss"],
                "train_bbox": epoch_avg["bbox_loss"], "train_kps": epoch_avg["kps_loss"],
                "val_loss": val_loss_metrics["loss"], "mAP50": det_metrics["mAP50"],
                "kps_nme": kps_nme, "composite_score": composite_score,
                "composite_ema": composite_ema, "is_best": int(is_best), "reverted": int(reverted),
                "epochs_since_improve": epochs_since_improve,
            })

    print("Huấn luyện hoàn tất.")
    print(f"[log] toàn bộ đường cong train/val đã lưu tại: {metrics_csv_path}")


if __name__ == "__main__":
    main()
