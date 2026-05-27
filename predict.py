"""Predict PM2.5 AQI for a photo.

Loads a regression checkpoint (either EfficientNet or SimpleCNN), runs the
model on a JPEG/PNG, and prints both the raw AQI estimate and the EPA
category that AQI falls into.

Usage:
    python predict.py path/to/photo.jpg
    python predict.py path/to/photo.jpg --checkpoint checkpoints/best_efficientnet.pt
"""

import argparse
import torch
from PIL import Image

from aqi import CLASSES, CLASS_RANGES, pm25_to_class
from dataset import build_transforms
from model import build_model


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a photo (jpg or png).")
    parser.add_argument("--checkpoint", default="checkpoints/best_efficientnet.pt")
    args = parser.parse_args()

    device = pick_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "efficientnet")

    model = build_model(model_name, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    img = Image.open(args.image).convert("RGB")
    x = build_transforms(train=False)(img).unsqueeze(0).to(device)

    with torch.no_grad():
        aqi = model(x).item()
    aqi_clipped = max(0.0, aqi)
    cls = pm25_to_class(aqi_clipped)

    neg_note = "  (raw output was negative, clipped to 0)" if aqi < 0 else ""
    epoch = ckpt.get("epoch", "?")
    cat_acc = ckpt.get("val_cat_acc", 0)
    mae = ckpt.get("val_mae", 0)

    print()
    print(f"Model:           {model_name}")
    print(f"Photo:           {args.image}")
    print(f"Predicted AQI:   {aqi_clipped:.1f}{neg_note}")
    print(f"Category:        {CLASSES[cls]}  (range {CLASS_RANGES[cls]})")
    print()
    print(f"Checkpoint stats: epoch {epoch},  val cat-acc {cat_acc:.3f},  val MAE {mae:.2f}")


if __name__ == "__main__":
    main()
