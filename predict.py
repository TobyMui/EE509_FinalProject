"""Predict PM2.5 AQI for a photo.

Loads the regression checkpoint, runs the model on a JPEG/PNG, and prints
both the raw AQI estimate and the EPA category that AQI falls into.

Usage:
    python predict.py path/to/photo.jpg
    python predict.py path/to/photo.jpg --checkpoint checkpoints/best.pt
"""

import argparse

import torch
from PIL import Image

from aqi import CLASSES, CLASS_RANGES, pm25_to_class
from dataset import build_transforms
from model import EfficientNetB0Regressor


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a photo (jpg or png).")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    device = pick_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = EfficientNetB0Regressor(pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    img = Image.open(args.image).convert("RGB")
    x = build_transforms(train=False)(img).unsqueeze(0).to(device)

    with torch.no_grad():
        aqi = model(x).item()
    aqi_clipped = max(0.0, aqi)
    cls = pm25_to_class(aqi_clipped)

    print(f"\nPhoto:           {args.image}")
    print(f"Predicted AQI:   {aqi_clipped:.1f}" + ("  (raw model output was negative, clipped to 0)" if aqi < 0 else ""))
    print(f"Category:        {CLASSES[cls]}  (range {CLASS_RANGES[cls]})")
    print(f"\nCheckpoint stats: epoch {ckpt.get('epoch', '?')},  "
          f"val cat-acc {ckpt.get('val_cat_acc', 0):.3f},  "
          f"val MAE {ckpt.get('val_mae', 0):.2f}")


if __name__ == "__main__":
    main()
