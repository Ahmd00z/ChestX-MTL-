"""
ChestX-MTL Unit Tests
"""
import os
import sys
import unittest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.encoder import EfficientNetEncoder
from src.models.classifier import ClassificationHead, FocalLoss
from src.models.detector import DetectionHead
from src.models.segmenter import SegmentationHead, CombinedSegmentationLoss
from src.models.mtl_model import ChestXMTL


class TestEncoder(unittest.TestCase):
    def test_forward(self):
        model = EfficientNetEncoder(pretrained=False)
        x = torch.randn(2, 3, 512, 512)
        out = model(x)
        self.assertIn("features", out)
        self.assertIn("global_feature", out)
        self.assertEqual(len(out["features"]), 4)
        self.assertEqual(out["global_feature"].shape[0], 2)


class TestClassifier(unittest.TestCase):
    def test_forward(self):
        head = ClassificationHead(input_dim=1792, num_classes=14)
        x = torch.randn(2, 1792, 16, 16)
        out = head(x)
        self.assertEqual(out.shape, (2, 14))

    def test_focal_loss(self):
        criterion = FocalLoss()
        pred = torch.randn(2, 14)
        target = torch.randint(0, 2, (2, 14)).float()
        loss = criterion(pred, target)
        self.assertGreater(loss.item(), 0)


class TestDetector(unittest.TestCase):
    def test_forward(self):
        head = DetectionHead(in_channels=256, num_classes=1)
        features = [
            torch.randn(2, 256, 128, 128),
            torch.randn(2, 256, 64, 64),
            torch.randn(2, 256, 32, 32),
            torch.randn(2, 256, 16, 16)
        ]
        cls_out, reg_out = head(features)
        self.assertEqual(len(cls_out), 4)
        self.assertEqual(len(reg_out), 4)


class TestSegmenter(unittest.TestCase):
    def test_forward(self):
        head = SegmentationHead(encoder_channels=[32, 56, 160, 448], num_classes=1)
        features = [
            torch.randn(2, 32, 128, 128),
            torch.randn(2, 56, 64, 64),
            torch.randn(2, 160, 32, 32),
            torch.randn(2, 448, 16, 16)
        ]
        out = head(features)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 1)

    def test_combined_loss(self):
        criterion = CombinedSegmentationLoss()
        pred = torch.randn(2, 1, 128, 128)
        target = torch.randint(0, 2, (2, 1, 128, 128)).float()
        loss = criterion(pred, target)
        self.assertGreater(loss.item(), 0)


class TestMTLModel(unittest.TestCase):
    def test_forward(self):
        model = ChestXMTL()
        x = torch.randn(2, 3, 512, 512)
        out = model(x)
        self.assertIn("cls_logits", out)
        self.assertIn("seg_mask", out)
        self.assertIn("det_cls", out)
        self.assertIn("det_reg", out)
        self.assertEqual(out["cls_logits"].shape, (2, 14))
        self.assertEqual(out["seg_mask"].shape[2:], (512, 512))

    def test_predict(self):
        model = ChestXMTL()
        x = torch.randn(2, 3, 512, 512)
        out = model.predict(x)
        self.assertIn("cls_probs", out)
        self.assertIn("seg_mask", out)

    def test_compute_loss(self):
        model = ChestXMTL()
        x = torch.randn(2, 3, 512, 512)
        predictions = model(x)
        targets = {
            "cls_labels": torch.randint(0, 2, (2, 14)).float(),
            "seg_masks": torch.randint(0, 2, (2, 1, 512, 512)).float()
        }
        losses = model.compute_loss(predictions, targets)
        self.assertIn("combined_loss", losses)
        self.assertGreater(losses["combined_loss"].item(), 0)

    def test_parameter_count(self):
        model = ChestXMTL()
        total = model.get_total_params()
        trainable = model.get_trainable_params()
        self.assertGreater(total, 0)
        self.assertGreaterEqual(trainable, 0)
        self.assertLessEqual(trainable, total)


if __name__ == "__main__":
    unittest.main()
