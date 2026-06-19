import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    """Small CNN that renders grid pixel observations into feature vectors.

    Designed as a drop-in replacement — Phase 2 can swap this
    for MobileViT without changing downstream components.
    """

    def __init__(self, input_channels=3, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, stride=2, padding=2),  # → H/2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # → H/4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # → H/8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        """x: (B, C, H, W) uint8/float32 → (B, feature_dim) float32"""
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        # Normalize input range to [0, 1]
        x = self.cnn(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
