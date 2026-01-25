"""
FastAPI wrapper for Prithvi Soil/Crop Classifier
Provides REST API endpoints for geospatial image analysis
"""
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np

# Add Model directory to path to import inference functions
BACKEND_DIR = Path(__file__).parent.parent
MODEL_DIR = BACKEND_DIR / "Model"
sys.path.insert(0, str(MODEL_DIR))

# Import inference functions
try:
    from infer import (
        load_trained_model,
        analyze_image_grid,
        CLASSES,
        preprocess_image,
        BandSelector,
        load_multispectral_image,
        load_rgb_image
    )
except ImportError as e:
    raise ImportError(
        f"Could not import inference module. Make sure Model/infer.py exists.\n"
        f"Error: {e}\n"
        f"Model directory: {MODEL_DIR}"
    )
import torch

# ==========================================
# 1. FASTAPI APP SETUP
# ==========================================

app = FastAPI(
    title="Soil/Crop Analysis API",
    description="Geospatial image analysis API using Prithvi-100M model",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. MODEL INITIALIZATION (Singleton)
# ==========================================

_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    """Load model singleton"""
    global _model
    if _model is None:
        checkpoint_path = MODEL_DIR / "checkpoints" / "prithvi_soil_best.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        _model = load_trained_model(str(checkpoint_path))
        _model.eval()
    return _model

# ==========================================
# 3. RESPONSE MODELS
# ==========================================

class GridCellPrediction(BaseModel):
    """Prediction for a single grid cell"""
    row: int
    col: int
    class_name: str
    class_index: int
    confidence: float
    health_score: float

class ClassDistribution(BaseModel):
    """Class distribution statistics"""
    class_name: str
    cell_count: int
    percentage: float

class AnalysisResponse(BaseModel):
    """Complete analysis response"""
    overall_prediction: str
    overall_confidence: float
    average_health_score: float
    grid_size: List[int]
    total_cells: int
    grid_predictions: List[GridCellPrediction]
    class_distribution: List[ClassDistribution]
    top_classes: List[Dict[str, Any]]
    image_info: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    classes: List[str]

# ==========================================
# 4. UTILITY FUNCTIONS
# ==========================================

def process_analysis_results(
    predictions: np.ndarray,
    confidences: np.ndarray,
    health_scores: np.ndarray,
    grid_size: tuple,
    overall_pred: int,
    overall_conf: float
) -> Dict[str, Any]:
    """Convert numpy arrays to JSON-serializable format"""
    
    grid_rows, grid_cols = grid_size
    total_cells = grid_rows * grid_cols
    
    # Grid predictions
    grid_predictions = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            grid_predictions.append({
                "row": int(i),
                "col": int(j),
                "class_name": CLASSES[int(predictions[i, j])],
                "class_index": int(predictions[i, j]),
                "confidence": float(confidences[i, j]),
                "health_score": float(health_scores[i, j])
            })
    
    # Class distribution
    unique, counts = np.unique(predictions, return_counts=True)
    class_distribution = []
    for class_idx, count in zip(unique, counts):
        class_distribution.append({
            "class_name": CLASSES[int(class_idx)],
            "cell_count": int(count),
            "percentage": float(100 * count / total_cells)
        })
    
    # Top classes
    top_classes = sorted(
        class_distribution,
        key=lambda x: x["cell_count"],
        reverse=True
    )[:5]
    
    return {
        "overall_prediction": CLASSES[overall_pred],
        "overall_confidence": float(overall_conf),
        "average_health_score": float(health_scores.mean()),
        "grid_size": [int(grid_rows), int(grid_cols)],
        "total_cells": int(total_cells),
        "grid_predictions": grid_predictions,
        "class_distribution": class_distribution,
        "top_classes": top_classes
    }

# ==========================================
# 5. API ENDPOINTS
# ==========================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    try:
        model = get_model()
        model_loaded = model is not None
    except Exception as e:
        model_loaded = False
    
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": str(_device),
        "classes": CLASSES
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model = get_model()
        model_loaded = model is not None
    except Exception as e:
        model_loaded = False
    
    return {
        "status": "ok" if model_loaded else "error",
        "model_loaded": model_loaded,
        "device": str(_device),
        "classes": CLASSES
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(..., description="Image file (TIF multispectral or RGB)"),
    grid_rows: int = Form(32, description="Number of grid rows", ge=32, le=64),
    grid_cols: int = Form(32, description="Number of grid columns", ge=32, le=64),
    is_rgb: Optional[bool] = Form(None, description="Optional: Override auto-detection. If not provided, auto-detects based on file extension")
):
    """
    Analyze geospatial image for crop/soil classification and health
    
    - **file**: Image file (multispectral TIF or RGB image)
    - **grid_rows**: Number of rows in analysis grid (32-64)
    - **grid_cols**: Number of columns in analysis grid (32-64)
    - **is_rgb**: Optional override. If not provided, auto-detects: .tif/.tiff = multispectral, others = RGB
    """
    try:
        # Load model
        model = get_model()
        
        # Save uploaded file temporarily
        file_suffix = Path(file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Auto-detect image type if not explicitly provided
            # RGB formats: .png, .jpg, .jpeg, .bmp, .gif, etc. (anything that's not .tif)
            # Multispectral: .tif, .tiff (13-band Sentinel-2)
            if is_rgb is None:
                # Auto-detect based on file extension
                detected_is_rgb = file_suffix not in ['.tif', '.tiff']
                print(f"[AUTO-DETECT] Image type: {'RGB' if detected_is_rgb else 'Multispectral'} (extension: {file_suffix})")
            else:
                # Use explicit override
                detected_is_rgb = is_rgb
                if detected_is_rgb and file_suffix in ['.tif', '.tiff']:
                    print(f"[WARN] File is .tif but is_rgb=True. Treating as RGB.")
            
            # Analyze image
            predictions, confidences, img_13band, health_scores = analyze_image_grid(
                tmp_path,
                model,
                grid_size=(grid_rows, grid_cols),
                is_rgb=detected_is_rgb
            )
            
            # Get overall prediction
            # Ensure model is in eval mode (critical for correct inference)
            model.eval()
            band_selector = BandSelector()
            overall_tensor = preprocess_image(img_13band, band_selector)
            with torch.no_grad():
                overall_output = model(overall_tensor, return_health=False)
                overall_probs = torch.softmax(overall_output, dim=1)
                overall_pred = torch.argmax(overall_probs).item()
                overall_conf = overall_probs[0][overall_pred].item()
            
            # Process results
            results = process_analysis_results(
                predictions,
                confidences,
                health_scores,
                (grid_rows, grid_cols),
                overall_pred,
                overall_conf
            )
            
            # Add image info
            results["image_info"] = {
                "filename": file.filename,
                "image_shape": list(img_13band.shape),
                "is_rgb": detected_is_rgb,
                "auto_detected": is_rgb is None
            }
            
            return AnalysisResponse(**results)
            
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
            
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Model file not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/batch")
async def analyze_batch(
    files: List[UploadFile] = File(..., description="List of image files"),
    grid_rows: int = Form(32),
    grid_cols: int = Form(32),
    is_rgb: Optional[bool] = Form(None, description="Optional: Override auto-detection for all files")
):
    """
    Analyze multiple images in batch
    
    Returns list of analysis results for each image
    """
    results = []
    model = get_model()
    
    for file in files:
        try:
            # Save uploaded file temporarily
            file_suffix = Path(file.filename).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            try:
                # Auto-detect image type if not explicitly provided
                if is_rgb is None:
                    # Auto-detect based on file extension
                    detected_is_rgb = file_suffix not in ['.tif', '.tiff']
                else:
                    # Use explicit override
                    detected_is_rgb = is_rgb
                
                # Analyze image
                predictions, confidences, img_13band, health_scores = analyze_image_grid(
                    tmp_path,
                    model,
                    grid_size=(grid_rows, grid_cols),
                    is_rgb=detected_is_rgb
                )
                
                # Get overall prediction
                # Ensure model is in eval mode (critical for correct inference)
                model.eval()
                band_selector = BandSelector()
                overall_tensor = preprocess_image(img_13band, band_selector)
                with torch.no_grad():
                    overall_output = model(overall_tensor, return_health=False)
                    overall_probs = torch.softmax(overall_output, dim=1)
                    overall_pred = torch.argmax(overall_probs).item()
                    overall_conf = overall_probs[0][overall_pred].item()
                
                # Process results
                result = process_analysis_results(
                    predictions,
                    confidences,
                    health_scores,
                    (grid_rows, grid_cols),
                    overall_pred,
                    overall_conf
                )
                
                result["image_info"] = {
                    "filename": file.filename,
                    "image_shape": list(img_13band.shape),
                    "is_rgb": detected_is_rgb
                }
                
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "result": result
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
            finally:
                Path(tmp_path).unlink(missing_ok=True)
                
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {"results": results, "total": len(files), "successful": sum(1 for r in results if r.get("success"))}

# ==========================================
# 6. STARTUP EVENT
# ==========================================

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    try:
        get_model()
        print("[OK] Model loaded successfully on startup")
    except Exception as e:
        print(f"[WARN] Could not load model on startup: {e}")
        print("   Model will be loaded on first request")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

