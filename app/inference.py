"""
ChestX-MTL Inference Engine
Lightweight inference wrapper for deployment.
"""
import os
import sys
import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.mtl_model import ChestXMTL
from src.utils.helpers import load_config, get_device


class ChestXInference:
    """
    Production-ready inference engine for ChestX-MTL.

    Usage:
        engine = ChestXInference("checkpoints/best_model.pth")
        results = engine.predict("path/to/xray.jpg")
    """

    DISEASE_LABELS = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Effusion", "Emphysema", "Fibrosis", "Hernia",
        "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
        "Pneumonia", "Pneumothorax"
    ]

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "config/config.yaml",
        device: str = "auto",
        image_size: int = 512
    ):
        self.device = get_device(device)
        self.image_size = image_size
        self.config = load_config(config_path)

        # Build model
        model_config = {
            "encoder": self.config["model"]["encoder"],
            "classifier": self.config["model"]["classifier"],
            "detector": self.config["model"]["detector"],
            "segmenter": self.config["model"]["segmenter"],
            "mtl": self.config["model"]["mtl"],
            "losses": self.config["losses"]
        }
        self.model = ChestXMTL(model_config).to(self.device)

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"Model loaded from {checkpoint_path}")
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")

        # Preprocessing
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def preprocess(self, image_input) -> torch.Tensor:
        """
        Preprocess image input.

        Args:
            image_input: Path string, PIL Image, or numpy array

        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image = np.array(image_input)
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            raise ValueError(f"Unsupported input type: {type(image_input)}")

        # Ensure 3 channels
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Apply transforms
        transformed = self.transform(image=image)
        tensor = transformed["image"].unsqueeze(0)

        return tensor.to(self.device)

    def predict(
        self,
        image_input,
        cls_threshold: float = 0.5,
        seg_threshold: float = 0.5
    ) -> Dict:
        """
        Run inference on a single image.

        Args:
            image_input: Image path, PIL Image, or numpy array
            cls_threshold: Classification probability threshold
            seg_threshold: Segmentation mask threshold

        Returns:
            Dictionary with predictions
        """
        tensor = self.preprocess(image_input)

        with torch.no_grad():
            outputs = self.model.predict(tensor, cls_threshold=cls_threshold)

        # Process classification
        cls_probs = outputs["cls_probs"][0].cpu().numpy()
        cls_preds = outputs["cls_preds"][0].cpu().numpy()

        detected_diseases = [
            {
                "disease": self.DISEASE_LABELS[i],
                "probability": float(cls_probs[i]),
                "detected": bool(cls_preds[i])
            }
            for i in range(len(self.DISEASE_LABELS))
        ]
        detected_diseases.sort(key=lambda x: x["probability"], reverse=True)

        # Process segmentation
        seg_mask = outputs["seg_mask"][0, 0].cpu().numpy()
        seg_mask_binary = (seg_mask > seg_threshold).astype(np.uint8)

        return {
            "classification": {
                "diseases": detected_diseases,
                "top_predictions": [d for d in detected_diseases if d["detected"]]
            },
            "segmentation": {
                "mask": seg_mask,
                "mask_binary": seg_mask_binary,
                "affected_area_ratio": float(seg_mask_binary.sum() / seg_mask_binary.size)
            },
            "metadata": {
                "threshold_cls": cls_threshold,
                "threshold_seg": seg_threshold,
                "image_size": self.image_size
            }
        }

    def predict_batch(
        self,
        image_inputs: List,
        cls_threshold: float = 0.5,
        seg_threshold: float = 0.5
    ) -> List[Dict]:
        """Batch inference."""
        tensors = [self.preprocess(img) for img in image_inputs]
        batch = torch.cat(tensors, dim=0)

        with torch.no_grad():
            outputs = self.model.predict(batch, cls_threshold=cls_threshold)

        results = []
        for i in range(len(image_inputs)):
            cls_probs = outputs["cls_probs"][i].cpu().numpy()
            cls_preds = outputs["cls_preds"][i].cpu().numpy()
            seg_mask = outputs["seg_mask"][i, 0].cpu().numpy()
            seg_mask_binary = (seg_mask > seg_threshold).astype(np.uint8)

            detected = [
                {"disease": self.DISEASE_LABELS[j], "probability": float(cls_probs[j]), "detected": bool(cls_preds[j])}
                for j in range(len(self.DISEASE_LABELS))
            ]
            detected.sort(key=lambda x: x["probability"], reverse=True)

            results.append({
                "classification": {"diseases": detected},
                "segmentation": {
                    "mask": seg_mask,
                    "mask_binary": seg_mask_binary,
                    "affected_area_ratio": float(seg_mask_binary.sum() / seg_mask_binary.size)
                }
            })

        return results

    def visualize(
        self,
        image_input,
        result: Dict,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create visualization of predictions.

        Returns:
            Visualization image as numpy array
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # Load original image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image = np.array(image_input)
        else:
            image = image_input.copy()

        # Resize for display
        image = cv2.resize(image, (self.image_size, self.image_size))

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Original
        axes[0].imshow(image)
        axes[0].set_title("Original X-Ray", fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Classification
        diseases = result["classification"]["diseases"]
        names = [d["disease"] for d in diseases]
        probs = [d["probability"] for d in diseases]
        colors = ['#e74c3c' if d["detected"] else '#3498db' for d in diseases]

        y_pos = np.arange(len(names))
        bars = axes[1].barh(y_pos, probs, color=colors, edgecolor='white', height=0.7)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(names, fontsize=9)
        axes[1].set_xlim(0, 1)
        axes[1].axvline(result["metadata"]["threshold_cls"], color='red', linestyle='--', alpha=0.7)
        axes[1].set_xlabel("Probability", fontsize=12)
        axes[1].set_title("Disease Detection", fontsize=14, fontweight='bold')
        axes[1].invert_yaxis()

        # Add probability labels
        for bar, prob in zip(bars, probs):
            axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{prob:.2f}', va='center', fontsize=8)

        # Segmentation overlay
        seg_mask = result["segmentation"]["mask"]
        seg_mask = cv2.resize(seg_mask, (self.image_size, self.image_size))

        axes[2].imshow(image)
        heatmap = axes[2].imshow(seg_mask, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title("Segmentation Heatmap", fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

        # Convert to numpy
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return vis_image


if __name__ == "__main__":
    print("Inference engine ready!")
