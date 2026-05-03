"""
ChestX-MTL Segmentation Head
UNet++ with attention gates for precise lung/lesion segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class AttentionGate(nn.Module):
    """Attention Gate for focusing on relevant regions."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvBlock(nn.Module):
    """Double convolution block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SegmentationHead(nn.Module):
    """
    UNet++ style segmentation decoder with attention gates.
    """

    def __init__(
        self,
        encoder_channels: List[int] = [32, 56, 160, 448],  # EfficientNet-B4 stages
        decoder_channels: List[int] = [256, 128, 64, 32],
        num_classes: int = 1,
        use_attention: bool = True
    ):
        super().__init__()

        self.use_attention = use_attention
        self.num_classes = num_classes

        # Reverse for decoder path
        encoder_channels = encoder_channels[::-1]

        # Decoder blocks
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attentions = nn.ModuleList() if use_attention else None

        for i in range(len(decoder_channels)):
            in_ch = encoder_channels[i] if i == 0 else decoder_channels[i-1]
            skip_ch = encoder_channels[i+1] if i+1 < len(encoder_channels) else 0
            out_ch = decoder_channels[i]

            self.ups.append(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
            )

            if use_attention and skip_ch > 0:
                self.attentions.append(AttentionGate(out_ch, skip_ch, out_ch // 2))
                decoder_in = out_ch + skip_ch
            else:
                decoder_in = out_ch + skip_ch if skip_ch > 0 else out_ch

            self.decoders.append(ConvBlock(decoder_in, out_ch))

        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1] // 2, 3, padding=1),
            nn.BatchNorm2d(decoder_channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels[-1] // 2, num_classes, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of encoder features [C1, C2, C3, C4] (low to high level)
        """
        # Reverse to start from deepest
        features = features[::-1]

        x = features[0]

        for i in range(len(self.decoders)):
            x = self.ups[i](x)

            if i + 1 < len(features):
                skip = features[i + 1]

                if self.use_attention and self.attentions is not None and i < len(self.attentions):
                    skip = self.attentions[i](x, skip)

                # Resize if needed
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

                x = torch.cat([x, skip], dim=1)

            x = self.decoders[i](x)

        return self.final(x)


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class CombinedSegmentationLoss(nn.Module):
    """Combined Dice + BCE Loss."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(pred, target) + self.bce_weight * self.bce(pred, target)


if __name__ == "__main__":
    head = SegmentationHead(encoder_channels=[32, 56, 160, 448], num_classes=1)
    features = [
        torch.randn(2, 32, 128, 128),
        torch.randn(2, 56, 64, 64),
        torch.randn(2, 160, 32, 32),
        torch.randn(2, 448, 16, 16)
    ]
    out = head(features)
    print(f"Segmentation output: {out.shape}")  # [2, 1, 128, 128]
