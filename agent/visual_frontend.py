"""Visual frontend — lightweight model (MobileNetV3) for screen understanding.

Processes desktop screenshots into:
1. Feature vectors usable by the world model
2. Enhanced element detection (complements OCR)

Drop-in upgrade path: replace MobileNetV3 with MobileViT when available.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class VisualFrontend(nn.Module):
    """Lightweight CNN (MobileNetV3) for screenshot feature extraction.

    Input:  (B, 3, H, W) uint8 screenshot
    Output: (B, feature_dim) float32 feature vector
    """

    def __init__(self, feature_dim=256, pretrained=True):
        super().__init__()
        self.feature_dim = feature_dim

        # Use MobileNetV3-small as backbone (lightweight, ~2.5M params)
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        # Remove classifier head, keep feature extractor
        self.backbone = backbone.features  # nn.Sequential of conv layers
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(576, feature_dim)  # MobileNetV3-small last channel = 576

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        # Normalize using ImageNet stats (pretrained model expects this)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        features = self.backbone(x)
        features = self.pool(features).flatten(1)
        features = self.fc(features)
        return features

    def get_screen_features(self, screenshot: np.ndarray) -> np.ndarray:
        """Convenience method: numpy screenshot -> numpy feature vector.

        Args:
            screenshot: (H, W, 3) uint8 RGB array
        Returns:
            (feature_dim,) float32 feature vector
        """
        self.eval()
        with torch.no_grad():
            # HWC -> CHW -> BCHW
            img_t = torch.ByteTensor(screenshot).permute(2, 0, 1).unsqueeze(0)
            features = self.forward(img_t)
            return features.squeeze(0).cpu().numpy()


class ElementEncoder(nn.Module):
    """Encodes detected UI elements into a feature vector.

    Takes OCR-detected elements + visual features and produces
    an enhanced WSG encoding.
    """

    def __init__(self, max_entities=50, entity_feat_dim=64):
        super().__init__()
        self.max_entities = max_entities
        self.entity_encoder = nn.Sequential(
            nn.Linear(11, entity_feat_dim),  # 11 = base entity features
            nn.ReLU(),
            nn.Linear(entity_feat_dim, entity_feat_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            entity_feat_dim, num_heads=4, batch_first=True)
        self.to_wsg_features = nn.Linear(
            max_entities * entity_feat_dim, max_entities * 11)

    def forward(self, entity_features, visual_features):
        """Enhance entity features using visual context via cross-attention."""
        # entity_features: (B, N, 11)
        # visual_features: (B, D) -> broadcast
        B, N, _ = entity_features.shape
        encoded = self.entity_encoder(entity_features)  # (B, N, 64)
        # Simple cross-attention (visual -> entity)
        # For now, just pass through the entity encoder
        result = encoded.reshape(B, -1)
        return self.to_wsg_features(result).reshape(B, N, 11)
