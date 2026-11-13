import math
import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.transforms import v2
from tqdm import tqdm

from badminton_dataset import BadmintonDetection, collate_fn, CLASSES, DEFAULT_ANN_ROOT


def get_args():
    parser = ArgumentParser(description="Badminton detection (players + shuttle) on BFMD")
    # dataset hyperparameter---------------
    parser.add_argument("--ann_root", "-a", type=str, default=DEFAULT_ANN_ROOT, help="BFMD_data/annotations")
    parser.add_argument("--frames_root", "-f", type=str, default="dataset/bfmd_frames",
                        help="frames extracted with `python badminton_dataset.py extract`")
    parser.add_argument("--video_root", "-v", type=str, default=None, help="fallback: read frames from mp4 files")
    parser.add_argument("--stride", "-s", type=int, default=5, help="use every n-th annotated train frame")
    parser.add_argument("--val_stride", type=int, default=30, help="use every n-th annotated val frame")
    parser.add_argument("--image_size", "-i", type=int, default=720, help="shorter side the model resizes to")
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--num_workers", "-w", type=int, default=4)
    # model hyperparameter---------------------------
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--default_anchors", action="store_true",
                        help="keep torchvision anchors (32-512px); default uses 16-256px for the small shuttle")
    parser.add_argument("--amp", action="store_true", help="mixed-precision training")
    # other hyperparameter--------------------------
    parser.add_argument("--epoch", "-e", type=int, default=10, help="num of epoch")
    parser.add_argument("--trained_model", "-t", type=str, default="trained_models", help="checkpoint dir")
    parser.add_argument("--resume", "-r", action="store_true", help="resume from <trained_model>/last_model.pth")
    return parser.parse_args()


def build_model(args):
    num_classes = len(CLASSES)  # 3 classes + background
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
                                              min_size=args.image_size, max_size=round(args.image_size * 16 / 9))
    if not args.default_anchors:
        # same number of anchors per location as the pretrained RPN, so its head weights still fit
        model.rpn.anchor_generator = AnchorGenerator(((16, 32, 64, 128, 256),) * 3, ((0.5, 1.0, 2.0),) * 3)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def average_precision(preds, gts, iou_thr=0.5):
    """VOC-style AP for one class. preds: list of (scores, boxes) per image, gts: list of boxes per image."""
    n_gt = sum(len(g) for g in gts)
    if n_gt == 0:
        return float("nan")
    records = []  # (score, image_idx, box)
    for img_idx, (scores, boxes) in enumerate(preds):
        records += [(s, img_idx, b) for s, b in zip(scores.tolist(), boxes)]
    records.sort(key=lambda r: -r[0])
    matched = [torch.zeros(len(g), dtype=torch.bool) for g in gts]
    tp = np.zeros(len(records))
    for k, (_, img_idx, box) in enumerate(records):
        g = gts[img_idx]
        if len(g) == 0:
            continue
        ious = box_iou(box[None], g)[0]
        best = int(ious.argmax())
        if ious[best] >= iou_thr and not matched[img_idx][best]:
            matched[img_idx][best] = True
            tp[k] = 1
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(1 - tp)
    recall = np.concatenate([[0], tp_cum / n_gt, [1]])
    precision = np.concatenate([[1], tp_cum / np.maximum(tp_cum + fp_cum, 1e-9), [0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    num_classes = len(CLASSES)
    preds = {c: [] for c in range(1, num_classes)}
    gts = {c: [] for c in range(1, num_classes)}
    for images, targets in tqdm(dataloader, desc="Validation", colour="cyan"):
        outputs = model([image.to(device) for image in images])
        for out, target in zip(outputs, targets):
            out = {k: v.cpu() for k, v in out.items()}
            for c in range(1, num_classes):
                keep = out["labels"] == c
                preds[c].append((out["scores"][keep], out["boxes"][keep]))
                gts[c].append(target["boxes"][target["labels"] == c].as_subclass(torch.Tensor))
    ap = {CLASSES[c]: average_precision(preds[c], gts[c]) for c in range(1, num_classes)}
    return ap, float(np.nanmean(list(ap.values())))


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.trained_model, exist_ok=True)

    # The model resizes + normalizes images itself (GeneralizedRCNNTransform), so the
    # transforms only augment. Avoid random crops: they easily cut out the ~10px shuttle.
    # A horizontal flip keeps top/bottom player labels valid.
    train_transform = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.SanitizeBoundingBoxes(),
    ])
    test_transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
    ])
    common = {"ann_root": args.ann_root, "frames_root": args.frames_root, "video_root": args.video_root}
    train_data = BadmintonDetection(split="train", stride=args.stride, transform=train_transform, **common)
    test_data = BadmintonDetection(split="val", stride=args.val_stride, transform=test_transform, **common)
    print(f"train: {len(train_data)} frames from {len(train_data.matches)} matches | "
          f"val: {len(test_data)} frames from {len(test_data.matches)} matches")

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_model(args).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    total_iters = args.epoch * len(train_dataloader)

    def lr_lambda(it):  # linear warmup, then cosine decay
        if it < args.warmup_iters:
            return 0.001 + (1 - 0.001) * it / args.warmup_iters
        return 0.5 * (1 + math.cos(math.pi * (it - args.warmup_iters) / max(1, total_iters - args.warmup_iters)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    start_epoch, best_map = 0, -1.0
    last_path = os.path.join(args.trained_model, "last_model.pth")
    if args.resume and os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, best_map = checkpoint["epoch"], checkpoint["best_map"]
        print(f"resumed from epoch {start_epoch} (best mAP {best_map:.4f})")

    for epoch in range(start_epoch, args.epoch):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epoch}", colour="green")
        total_train_loss = 0.0
        for i, (images, targets) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            targets = [{"boxes": target["boxes"].to(device), "labels": target["labels"].to(device)}
                       for target in targets]
            # forward
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss_dict = model(images, targets)
                loss_sum = sum(loss for loss in loss_dict.values())
            if not torch.isfinite(loss_sum):
                print(f"non-finite loss at iter {i}, skipping batch: { {k: v.item() for k, v in loss_dict.items()} }")
                optimizer.zero_grad()
                scheduler.step()
                continue
            # backward
            optimizer.zero_grad()
            scaler.scale(loss_sum).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss_sum.item()
            progress_bar.set_postfix(loss=f"{total_train_loss / (i + 1):.4f}", lr=f"{scheduler.get_last_lr()[0]:.5f}")

        ap, val_map = evaluate(model, test_dataloader, device)
        is_best = val_map > best_map
        best_map = max(best_map, val_map)

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_map": val_map,
            "best_map": best_map,
            "classes": CLASSES,
            "args": vars(args),
        }
        torch.save(checkpoint, last_path)
        if is_best:
            torch.save(checkpoint, os.path.join(args.trained_model, "best_model.pth"))
        print(
            f"Epoch {epoch + 1}/{args.epoch} | "
            f"Train Loss: {total_train_loss / len(train_dataloader):.4f} | "
            f"Val mAP@0.5: {val_map:.4f} | "
            + " ".join(f"{k}: {v:.4f}" for k, v in ap.items())
        )


if __name__ == "__main__":
    main()
