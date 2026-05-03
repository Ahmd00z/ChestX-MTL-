#!/usr/bin/env python3
"""
Quick Demo: ChestX-MTL Inference
"""
import os
import sys
import torch
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from src.models.mtl_model import ChestXMTL


def create_dummy_image(path="demo_xray.png", size=512):
    """Create a dummy chest X-ray for testing."""
    # Create a grayscale gradient image
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Simulate lung regions
    cv2.ellipse(img, (size//3, size//2), (80, 120), 0, 0, 360, (180, 180, 180), -1)
    cv2.ellipse(img, (2*size//3, size//2), (80, 120), 0, 0, 360, (180, 180, 180), -1)

    # Add some noise
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(path, img)
    print(f"Created demo image: {path}")
    return path


def main():
    print("="*60)
    print("ChestX-MTL Quick Demo")
    print("="*60)

    # Create dummy image
    img_path = create_dummy_image()

    # Initialize model
    print("\nInitializing model...")
    model = ChestXMTL()
    model.eval()

    print(f"Model parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")

    # Load and preprocess image
    print("\nLoading image...")
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))

    # Normalize
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # To tensor
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

    # Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model.predict(tensor)

    # Results
    cls_probs = outputs["cls_probs"][0].numpy()
    seg_mask = outputs["seg_mask"][0, 0].numpy()

    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)

    labels = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Effusion", "Emphysema", "Fibrosis", "Hernia",
        "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
        "Pneumonia", "Pneumothorax"
    ]

    for label, prob in zip(labels, cls_probs):
        bar = "█" * int(prob * 20)
        print(f"{label:20s} | {prob:.3f} | {bar}")

    print("\n" + "="*60)
    print("SEGMENTATION RESULTS")
    print("="*60)
    print(f"Mask shape: {seg_mask.shape}")
    print(f"Mask min: {seg_mask.min():.4f}")
    print(f"Mask max: {seg_mask.max():.4f}")
    print(f"Mask mean: {seg_mask.mean():.4f}")

    print("\n✅ Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()
