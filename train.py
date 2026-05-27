"""Train a PM2.5 AQI regressor on PM25Vision.

Choose architecture with --model:
    python train.py --model efficientnet     # pretrained EfficientNet-B0
    python train.py --model simplecnn        # baseline CNN

Saves best_<model>.pt by category accuracy.
"""
import argparse, json, math
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

from aqi import CLASSES, pm25_to_class
from dataset import build_transforms, make_collate
from model import build_model


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_sq = total_abs = 0.0
    n = cat_correct = 0
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
    return math.sqrt(total_sq / n), total_abs / n, cat_correct / n


def build_optimizer(model, model_name, backbone_lr, head_lr, weight_decay):
    """Pretrained backbones (EfficientNet B0 / V2-S) get discriminative LRs;
    SimpleCNN trains all params at head_lr since nothing is pretrained."""
    if model_name in ("efficientnet", "efficientnetv2"):
        bb, hd = [], []
        for n, p in model.named_parameters():
            (hd if n.startswith("backbone.classifier") else bb).append(p)
        return torch.optim.AdamW(
            [{"params": bb, "lr": backbone_lr},
             {"params": hd, "lr": head_lr}],
            weight_decay=weight_decay,
        )
    return torch.optim.AdamW(model.parameters(), lr=head_lr, weight_decay=weight_decay)


def print_class_distribution(name, split):
    counts = Counter(pm25_to_class(x) for x in split["pm25"])
    parts = []
    for i in range(len(CLASSES)):
        short = CLASSES[i].split()[0]
        parts.append(f"{short:>8s}={counts.get(i, 0)}")
    print(f"  {name:5s} class counts:  " + "  ".join(parts))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="efficientnet",
                   choices=["efficientnet", "efficientnetv2", "simplecnn"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--backbone-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="checkpoints")
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Device: {device}    Model: {args.model}")
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
    print_class_distribution("train", train_ds)
    print_class_distribution("val", val_ds)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_collate(build_transforms(train=True)),
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(args.batch_size, 64), shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(build_transforms(train=False)),
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = build_model(args.model, pretrained=True).to(device)
    optimizer = build_optimizer(model, args.model, args.backbone_lr,
                                args.head_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    history = []
    best_cat_acc = 0.0
    best_path = out / f"best_{args.model}.pt"

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
                print(f"  ep{epoch:02d} step {step:4d}/{len(train_loader)}  mse {loss.item():.3f}")

        train_mse = running_loss / max(batches, 1)
        rmse, mae, cat_acc = evaluate(model, val_loader, device)
        scheduler.step()

        flag = ""
        if cat_acc > best_cat_acc:
            best_cat_acc = cat_acc
            payload = {
                "model_state": model.state_dict(),
                "model_name": args.model,
                "epoch": epoch,
                "val_rmse": rmse,
                "val_mae": mae,
                "val_cat_acc": cat_acc,
            }
            torch.save(payload, best_path)
            flag = "  <- best (saved)"

        print(f"epoch {epoch:02d}  train_mse {train_mse:8.3f}  "
              f"val_rmse {rmse:6.2f}  val_mae {mae:6.2f}  "
              f"val_cat_acc {cat_acc:.3f}{flag}")
        history.append({
            "epoch": epoch, "train_mse": train_mse,
            "val_rmse": rmse, "val_mae": mae, "val_cat_acc": cat_acc,
        })

    with open(out / f"history_{args.model}.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val category accuracy ({args.model}): {best_cat_acc:.3f}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
