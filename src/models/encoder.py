"""
ChestX-MTL Encoder Module
Uses EfficientNet-B4 as backbone with Feature Pyramid Network (FPN)
"""
import torch
import torch.nn as nn
import timm
from typing import List, Dict


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-B4 encoder with FPN for multi-scale feature extraction.
    Outputs features at multiple scales for detection and segmentation heads.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b4",
        pretrained: bool = True,
        dropout: float = 0.3,
        out_channels: int = 256
    ):
        super().__init__()

        # Load EfficientNet backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3, 4],  # P2, P3, P4, P5
        )

        # Get channel dimensions for each feature level
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)
            features = self.backbone(dummy)
            self.in_channels = [f.shape[1] for f in features]

        # Feature Pyramid Network (FPN)
        self.fpn = FeaturePyramidNetwork(
            in_channels=self.in_channels,
            out_channels=out_channels
        )

        # Global pooling for classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.out_channels = out_channels
        self.classifier_dim = self.in_channels[-1]  # Last feature map channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning multi-scale features and global feature.

        Returns:
            dict with keys:
                - 'features': List[Tensor] - FPN features [P2, P3, P4, P5]
                - 'global_feature': Tensor - Global pooled feature for classification
                - 'backbone_features': List[Tensor] - Raw backbone features
        """
        # Extract multi-scale features
        backbone_features = self.backbone(x)

        # Apply FPN
        fpn_features = self.fpn(backbone_features)

        # Global feature for classification
        global_feat = self.global_pool(backbone_features[-1])
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_feat = self.dropout(global_feat)

        return {
            "features": fpn_features,
            "global_feature": global_feat,
            "backbone_features": backbone_features
        }

    def get_classifier_dim(self) -> int:
        return self.classifier_dim


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion.
    """

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Build laterals
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            size = laterals[i].shape[2:]
            laterals[i] = laterals[i] + nn.functional.interpolate(
                laterals[i + 1], size=size, mode="nearest"
            )

        # Apply 3x3 conv
        outputs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        return outputs


if __name__ == "__main__":
    model = EfficientNetEncoder(pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"FPN features: {[f.shape for f in out['features']]}")
    print(f"Global feature: {out['global_feature'].shape}")
