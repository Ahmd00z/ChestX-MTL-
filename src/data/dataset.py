"""
ChestX-MTL Dataset Module
Supports NIH, RSNA, and SIIM-ACR datasets with advanced augmentations.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Optional, Tuple, Callable
import pandas as pd
from PIL import Image


class ChestXrayDataset(Dataset):
    """
    Multi-task dataset for chest X-ray analysis.

    Supports:
    - Classification: Multi-label disease labels
    - Detection: Bounding boxes [x, y, w, h]
    - Segmentation: Binary masks
    """

    def __init__(
        self,
        image_dir: str,
        annotations_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        mode: str = "train",
        image_size: int = 512,
        tasks: List[str] = ["classification", "detection", "segmentation"]
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.image_size = image_size
        self.tasks = tasks

        # Load annotations
        if annotations_file and os.path.exists(annotations_file):
            self.annotations = pd.read_csv(annotations_file)
        else:
            self.annotations = self._create_dummy_annotations()

        self.image_files = self.annotations["image_id"].unique().tolist()

        # Disease labels mapping (NIH 14 classes)
        self.disease_labels = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Effusion", "Emphysema", "Fibrosis", "Hernia",
            "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
            "Pneumonia", "Pneumothorax"
        ]

    def _create_dummy_annotations(self) -> pd.DataFrame:
        """Create dummy annotations for testing."""
        images = [f for f in os.listdir(self.image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return pd.DataFrame({"image_id": images})

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_id = self.image_files[idx]

        # Load image
        img_path = os.path.join(self.image_dir, image_id)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotations
        row = self.annotations[self.annotations["image_id"] == image_id].iloc[0]

        sample = {"image": image}

        # Classification labels
        if "classification" in self.tasks:
            if "labels" in row:
                labels = self._parse_labels(row["labels"])
            else:
                labels = np.zeros(len(self.disease_labels), dtype=np.float32)
            sample["cls_labels"] = labels

        # Detection targets
        if "detection" in self.tasks:
            if "bbox" in row:
                bboxes = self._parse_bboxes(row["bbox"])
            else:
                bboxes = np.array([])
            sample["bboxes"] = bboxes

        # Segmentation mask
        if "segmentation" in self.tasks:
            mask_path = os.path.join(self.image_dir, "masks", image_id)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype(np.float32)
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            sample["mask"] = mask

        # Apply transforms
        if self.transform:
            transformed = self.transform(**sample)
            sample = transformed

        # Convert to tensors
        result = {"image_id": image_id}

        if isinstance(sample["image"], np.ndarray):
            result["image"] = torch.from_numpy(sample["image"]).permute(2, 0, 1).float() / 255.0
        else:
            result["image"] = sample["image"]

        if "cls_labels" in sample:
            result["cls_labels"] = torch.from_numpy(sample["cls_labels"]).float()

        if "mask" in sample:
            if isinstance(sample["mask"], np.ndarray):
                result["seg_masks"] = torch.from_numpy(sample["mask"]).unsqueeze(0).float()
            else:
                result["seg_masks"] = sample["mask"].unsqueeze(0)

        if "bboxes" in sample:
            result["det_targets"] = sample["bboxes"]

        return result

    def _parse_labels(self, labels_str: str) -> np.ndarray:
        """Parse disease labels string to binary vector."""
        labels = np.zeros(len(self.disease_labels), dtype=np.float32)
        if pd.isna(labels_str) or labels_str == "No Finding":
            return labels

        diseases = [d.strip() for d in str(labels_str).split("|")]
        for disease in diseases:
            if disease in self.disease_labels:
                labels[self.disease_labels.index(disease)] = 1.0
        return labels

    def _parse_bboxes(self, bbox_str: str) -> np.ndarray:
        """Parse bounding boxes."""
        # Format: "x1,y1,w1,h1;x2,y2,w2,h2"
        bboxes = []
        if pd.isna(bbox_str):
            return np.array(bboxes)

        for box in str(bbox_str).split(";"):
            coords = [float(c) for c in box.split(",")]
            if len(coords) == 4:
                bboxes.append(coords)

        return np.array(bboxes, dtype=np.float32) if bboxes else np.array([])


def get_transforms(
    mode: str = "train",
    image_size: int = 512,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> Callable:
    """Get augmentation transforms."""

    if mode == "train":
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(intensity=(0.1, 0.5), p=1.0),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.RandomGamma(p=1.0),
                A.CLAHE(p=1.0),
            ], p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(p=1.0),
            ], p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))
    else:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    return transform


def get_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 512
) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders."""

    train_dataset = ChestXrayDataset(
        image_dir=os.path.join(data_dir, "train"),
        annotations_file=os.path.join(data_dir, "train.csv"),
        transform=get_transforms("train", image_size),
        mode="train"
    )

    val_dataset = ChestXrayDataset(
        image_dir=os.path.join(data_dir, "val"),
        annotations_file=os.path.join(data_dir, "val.csv"),
        transform=get_transforms("val", image_size),
        mode="val"
    )

    test_dataset = ChestXrayDataset(
        image_dir=os.path.join(data_dir, "test"),
        annotations_file=os.path.join(data_dir, "test.csv"),
        transform=get_transforms("val", image_size),
        mode="test"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }


if __name__ == "__main__":
    # Test with dummy data
    os.makedirs("/tmp/test_images", exist_ok=True)
    dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    cv2.imwrite("/tmp/test_images/test_001.png", dummy_img)

    dataset = ChestXrayDataset(
        image_dir="/tmp/test_images",
        transform=get_transforms("train", 512)
    )
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
