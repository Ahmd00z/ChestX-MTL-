#!/usr/bin/env python3
"""
ChestX-MTL FastAPI Backend
RESTful API for chest X-ray analysis.

Run: uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import sys
import io
import base64
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.inference import ChestXInference


# Initialize FastAPI app
app = FastAPI(
    title="ChestX-MTL API",
    description="Multi-Task Learning API for Chest X-Ray Analysis: Classification, Detection & Segmentation",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/checkpoints/best_model.pth")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config/config.yaml")

print(f"Loading model from {MODEL_PATH}...")
try:
    inference_engine = ChestXInference(
        checkpoint_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        device="auto"
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    inference_engine = None


# Pydantic models
class PredictionResponse(BaseModel):
    success: bool
    timestamp: str
    classification: dict
    segmentation: dict
    metadata: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    device: str


@app.get("/", response_model=HealthResponse)
async def root():
    """API health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=inference_engine is not None,
        version="2.0.0",
        device=str(inference_engine.device) if inference_engine else "N/A"
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return await root()


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    cls_threshold: float = 0.5,
    seg_threshold: float = 0.5
):
    """
    Analyze a chest X-ray image.

    Returns classification probabilities, detected diseases, and segmentation mask.
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file
    allowed_types = {"image/jpeg", "image/png", "image/jpg", "image/dicom"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run inference
        result = inference_engine.predict(
            image,
            cls_threshold=cls_threshold,
            seg_threshold=seg_threshold
        )

        # Generate visualization
        vis_image = inference_engine.visualize(image, result)
        vis_b64 = base64.b64encode(vis_image.tobytes()).decode('utf-8')

        return PredictionResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            classification=result["classification"],
            segmentation={
                "affected_area_ratio": result["segmentation"]["affected_area_ratio"],
                "mask_shape": list(result["segmentation"]["mask"].shape)
            },
            metadata={
                **result["metadata"],
                "filename": file.filename,
                "visualization": f"data:image/png;base64,{vis_b64}"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    cls_threshold: float = 0.5,
    seg_threshold: float = 0.5
):
    """Batch prediction for multiple images."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    images = []
    filenames = []

    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        images.append(image)
        filenames.append(file.filename)

    predictions = inference_engine.predict_batch(
        images,
        cls_threshold=cls_threshold,
        seg_threshold=seg_threshold
    )

    for fname, pred in zip(filenames, predictions):
        results.append({
            "filename": fname,
            "classification": pred["classification"],
            "segmentation": {
                "affected_area_ratio": pred["segmentation"]["affected_area_ratio"]
            }
        })

    return {"success": True, "results": results}


@app.get("/labels")
async def get_labels():
    """Get list of supported disease labels."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "labels": inference_engine.DISEASE_LABELS,
        "count": len(inference_engine.DISEASE_LABELS)
    }


def main():
    import uvicorn
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
