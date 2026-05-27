"""Model architectures for PM2.5 AQI regression.

Three options exposed via build_model(name):
  - "efficientnet":   pretrained EfficientNet-B0    (~5M params)
  - "efficientnetv2": pretrained EfficientNet-V2-S  (~22M params)
  - "simplecnn":      baseline CNN from the dataset card (~25K params)

All three have the same forward signature: input (B, 3, H, W), output (B,) AQI.
"""

import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
)


class EfficientNetB0Regressor(nn.Module):
    """Pretrained EfficientNet-B0 with the 1000-class ImageNet head replaced
    by a single linear unit."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


class EfficientNetV2SRegressor(nn.Module):
    """Pretrained EfficientNet-V2-S with the 1000-class ImageNet head replaced
    by a single linear unit. Note: pretrained at 384x384 — feeding 224x224
    still works (we do this here for compatibility with the other models)
    but accuracy may be slightly lower than at the native resolution."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_v2_s(weights=weights)
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


class SimpleCNN(nn.Module):
    """Baseline CNN from the PM25Vision dataset card. Three conv blocks
    followed by global average pooling and a linear regression head."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(1)


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """Factory. `name` is one of: 'efficientnet', 'efficientnetv2', 'simplecnn'."""
    name = name.lower()
    if name == "efficientnet":
        return EfficientNetB0Regressor(pretrained=pretrained)
    if name == "efficientnetv2":
        return EfficientNetV2SRegressor(pretrained=pretrained)
    if name == "simplecnn":
        return SimpleCNN()
    raise ValueError(
        f"Unknown model: {name!r}. Use 'efficientnet', 'efficientnetv2', or 'simplecnn'."
    )


def is_pretrained_backbone(name: str) -> bool:
    """Whether this architecture has a pretrained backbone that warrants
    discriminative LRs in the optimizer."""
    return name.lower() in ("efficientnet", "efficientnetv2")
