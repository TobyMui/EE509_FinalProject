"""EfficientNet-B0 with a regression head for PM2.5 AQI estimation."""

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0Regressor(nn.Module):
    """Pretrained EfficientNet-B0 with the 1000-class ImageNet head replaced
    by a single linear unit. Forward returns shape (batch,)."""

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
