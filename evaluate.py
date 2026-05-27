"""Evaluate one or more trained models on the PM25Vision test set.

Produces a confusion matrix figure (PNG) and prints RMSE / MAE / category
accuracy. If multiple checkpoints are passed, the figure shows them
side-by-side and a comparison table is printed.

Usage:
    python evaluate.py checkpoints/best_efficientnet.pt
    python evaluate.py checkpoints/best_efficientnet.pt checkpoints/best_simplecnn.pt
    python evaluate.py checkpoints/best_*.pt --save-path results/cm.png
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from aqi import CLASSES, pm25_to_class
from dataset import build_transforms, make_collate
from model import build_model


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def run_model(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).cpu().tolist()
        preds_all.extend(preds)
        labels_all.extend(labels.tolist())
    return np.array(preds_all), np.array(labels_all)


def metrics_from_preds(preds, labels):
    pred_cls = np.array([pm25_to_class(p) for p in preds])
    true_cls = np.array([pm25_to_class(y) for y in labels])
    rmse = math.sqrt(((preds - labels) ** 2).mean())
    mae = float(np.abs(preds - labels).mean())
    cat_acc = float((pred_cls == true_cls).mean())
    cm = confusion_matrix(true_cls, pred_cls, labels=list(range(len(CLASSES))))
    return rmse, mae, cat_acc, cm, pred_cls, true_cls


def plot_confusion_matrices(cms, titles, save_path):
    short_names = [c.replace(" for Sensitive Groups", " (Sens)") for c in CLASSES]
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.8))
    if n == 1:
        axes = [axes]
    for ax, cm, title in zip(axes, cms, titles):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm.astype(float) / np.maximum(row_sums, 1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=color, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved confusion matrices to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="One or more checkpoint paths.")
    parser.add_argument("--save-path", default="results/confusion_matrix.png")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--per-class-report", action="store_true",
                        help="Also print sklearn classification_report per model.")
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    print("Loading PM25Vision test split...")
    ds = load_dataset("DeadCardassian/PM25Vision")
    test_ds = ds["test"]
    print(f"  test samples: {len(test_ds)}")

    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(build_transforms(train=False)),
        pin_memory=(device == "cuda"),
    )

    results = []
    cms = []
    titles = []
    for ckpt_path in args.checkpoints:
        print(f"\n--- Evaluating {ckpt_path} ---")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_name = ckpt.get("model_name", "efficientnet")
        model = build_model(model_name, pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)

        preds, labels = run_model(model, loader, device)
        rmse, mae, cat_acc, cm, pred_cls, true_cls = metrics_from_preds(preds, labels)
        print(f"  model:    {model_name}")
        print(f"  RMSE:     {rmse:.2f}")
        print(f"  MAE:      {mae:.2f}")
        print(f"  cat-acc:  {cat_acc:.3f}")

        if args.per_class_report:
            print()
            print(classification_report(
                true_cls, pred_cls,
                labels=list(range(len(CLASSES))),
                target_names=CLASSES,
                zero_division=0,
            ))

        results.append({"path": ckpt_path, "name": model_name,
                        "rmse": rmse, "mae": mae, "cat_acc": cat_acc})
        cms.append(cm)
        titles.append(f"{model_name}\nRMSE={rmse:.1f}  MAE={mae:.1f}  cat-acc={cat_acc:.2f}")

    if len(results) > 1:
        print("\n=== Comparison ===")
        print(f"{'model':16s}  {'RMSE':>8s}  {'MAE':>8s}  {'cat-acc':>8s}")
        for r in results:
            print(f"{r['name']:16s}  {r['rmse']:8.2f}  {r['mae']:8.2f}  {r['cat_acc']:8.3f}")

    plot_confusion_matrices(cms, titles, args.save_path)


if __name__ == "__main__":
    main()
