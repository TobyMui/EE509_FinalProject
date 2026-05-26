"""Train EfficientNet-B0 as a PM2.5 AQI regressor on PM25Vision.

Validation tracks three metrics: RMSE, MAE, and bucketed category accuracy
(predicted AQI passed through aqi.pm25_to_class and compared against the
ground-truth bucket). The best checkpoint is saved by category accuracy,
since that's what the inference output cares about.

Usage:
    python train.py                            # full run
    python train.py --epochs 2 --subset 500    # quick smoke test
    python train.py --batch-size 96 --bf16     # 5070 Ti recipe
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

from aqi import pm25_to_class
from dataset import build_transforms, make_collate
from model import EfficientNetB0Regressor


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_sq, total_abs, n = 0.0, 0.0, 0
    cat_correct = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        diff = preds - labels
        total_sq += (diff * diff).sum().item()
        total_abs += diff.abs().sum().item()
        n += labels.numel()
        for p, y in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            if pm25_to_class(p) == pm25_to_class(y):
                cat_correct += 1
    rmse = math.sqrt(total_sq / n)
    mae = total_abs / n
    cat_acc = cat_correct / n
    return rmse, mae, cat_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--subset", type=int, default=0,
                        help="If >0, train on only this many samples (smoke test).")
    parser.add_argument("--bf16", action="store_true",
                        help="Mixed precision (bf16). Big speedup on RTX 40/50-series.")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Device: {device}")
    use_amp = args.bf16 and device == "cuda"
    if args.bf16 and not use_amp:
        print("(--bf16 ignored: requires CUDA)")

    print("Loading PM25Vision from HuggingFace (first run downloads ~1 GB)...")
    ds = load_dataset("DeadCardassian/PM25Vision")
    train_ds, val_ds = ds["train"], ds["test"]
    if args.subset > 0:
        train_ds = train_ds.select(range(min(args.subset, len(train_ds))))
        val_ds = val_ds.select(range(min(max(args.subset // 4, 1), len(val_ds))))
    print(f"  train: {len(train_ds)}  val(test split): {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_collate(build_transforms(train=True)),
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(args.batch_size, 64),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(build_transforms(train=False)),
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = EfficientNetB0Regressor(pretrained=True).to(device)

    # Discriminative learning rates: gentle on the pretrained backbone,
    # faster on the freshly-initialised regression head.
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        (head_params if name.startswith("backbone.classifier") else backbone_params).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    history = []
    best_cat_acc = 0.0
    best_path = out / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, batches = 0.0, 0
        for step, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds = model(imgs)
                    loss = criterion(preds, labels)
            else:
                preds = model(imgs)
                loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1
            if step % 20 == 0:
                print(f"  ep{epoch:02d} step {step:4d}/{len(train_loader)}  "
                      f"mse {loss.item():.3f}")

        train_mse = running_loss / max(batches, 1)
        rmse, mae, cat_acc = evaluate(model, val_loader, device)
        scheduler.step()

        flag = ""
        if cat_acc > best_cat_acc:
            best_cat_acc = cat_acc
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_rmse": rmse,
                "val_mae": mae,
                "val_cat_acc": cat_acc,
            }, best_path)
            flag = "  <- best (saved)"

        print(f"epoch {epoch:02d}  train_mse {train_mse:8.3f}  "
              f"val_rmse {rmse:6.2f}  val_mae {mae:6.2f}  "
              f"val_cat_acc {cat_acc:.3f}{flag}")
        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_rmse": rmse,
            "val_mae": mae,
            "val_cat_acc": cat_acc,
        })

    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val category accuracy: {best_cat_acc:.3f}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
