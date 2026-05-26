"""Dataset wiring for PM25Vision (regression).

The HuggingFace parquet returns each row's `image` as raw JPEG bytes (the
column is untyped). Labels are floats (AQI values).
"""

from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool):
    if train:
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _open_image(raw) -> Image.Image:
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, (bytes, bytearray)):
        return Image.open(BytesIO(raw)).convert("RGB")
    if isinstance(raw, dict) and raw.get("bytes") is not None:
        return Image.open(BytesIO(raw["bytes"])).convert("RGB")
    raise TypeError(f"Unexpected image type in dataset row: {type(raw)}")


def make_collate(transform):
    """Return a DataLoader collate_fn that yields (images, aqi_floats)."""
    def collate(batch):
        imgs = [transform(_open_image(x["image"])) for x in batch]
        labels = [float(x["pm25"]) for x in batch]
        return torch.stack(imgs), torch.tensor(labels, dtype=torch.float32)
    return collate
