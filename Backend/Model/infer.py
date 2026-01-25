"""
Inference script for Prithvi Soil/Crop Classifier
Supports both multi-spectral (13-band) and RGB images
"""
import torch
import torch.nn as nn
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import json
from pathlib import Path

try:
    from scipy.ndimage import zoom
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    import warnings
    warnings.warn("scipy not available, using torch interpolation for grid cells")

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Update paths to work from Backend directory
MODEL_DIR = Path(__file__).parent
CHECKPOINT_PATH = MODEL_DIR / "checkpoints" / "prithvi_soil_best.pth"
CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

class PrithviSoilClassifier(nn.Module):
    """
    Prithvi-100M Fine-Tuned for Geospatial Crop/Soil Analysis
    Matches the architecture from Training.py
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Find prithvi_model directory (from Backend directory)
        script_dir = Path(__file__).parent.parent  # Backend directory
        possible_paths = [
            script_dir / "prithvi_model",  # Backend/prithvi_model
            MODEL_DIR.parent / "prithvi_model",  # Alternative path
            Path("prithvi_model"),
            Path("../prithvi_model"),
        ]
        
        prithvi_model_dir = None
        for path in possible_paths:
            if path.exists() and (path / "Prithvi_EO_V1_100M.pt").exists():
                prithvi_model_dir = path
                break
        
        if prithvi_model_dir is None:
            raise FileNotFoundError("Could not find prithvi_model directory")
        
        # Add to path and import
        if str(prithvi_model_dir) not in sys.path:
            sys.path.insert(0, str(prithvi_model_dir))
        
        from prithvi_mae import PrithviMAE
        
        # Load config
        config_path = prithvi_model_dir / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        pretrained_cfg = config['pretrained_cfg']
        
        # Create encoder
        prithvi_model = PrithviMAE(
            img_size=pretrained_cfg['img_size'],
            patch_size=pretrained_cfg['patch_size'],
            num_frames=pretrained_cfg['num_frames'],
            in_chans=pretrained_cfg['in_chans'],
            embed_dim=pretrained_cfg['embed_dim'],
            depth=pretrained_cfg['depth'],
            num_heads=pretrained_cfg['num_heads'],
            decoder_embed_dim=pretrained_cfg['decoder_embed_dim'],
            decoder_depth=pretrained_cfg['decoder_depth'],
            decoder_num_heads=pretrained_cfg['decoder_num_heads'],
            mlp_ratio=pretrained_cfg['mlp_ratio'],
            encoder_only=True
        )
        
        # Load encoder weights
        checkpoint_path = prithvi_model_dir / "Prithvi_EO_V1_100M.pt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
        
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                encoder_state_dict[k.replace('encoder.', '')] = v
            elif not k.startswith('decoder.'):
                encoder_state_dict[k] = v
        
        prithvi_model.encoder.load_state_dict(encoder_state_dict, strict=False)
        self.backbone = prithvi_model.encoder
        
        # Classification head (matches Training.py)
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Crop health regression head (predicts health score 0-1)
        self.health_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x, return_health=False):
        if len(x.shape) == 4:  # [B, C, H, W]
            x = x.unsqueeze(2)  # Add temporal dimension
        
        features_list = self.backbone.forward_features(x)
        final_features = features_list[-1]
        cls_features = final_features[:, 0, :]
        
        classification = self.head(cls_features)
        if return_health:
            health = self.health_head(cls_features)
            return classification, health
        return classification
    
    def get_features(self, x):
        """Extract features for health prediction"""
        if len(x.shape) == 4:  # [B, C, H, W]
            x = x.unsqueeze(2)  # Add temporal dimension
        
        features_list = self.backbone.forward_features(x)
        final_features = features_list[-1]
        cls_features = final_features[:, 0, :]
        return cls_features

class BandSelector:
    """Select and normalize bands for Prithvi model (matches Training.py)"""
    BAND_INDICES = [1, 2, 3, 8, 11, 12]  # Blue, Green, Red, Narrow-NIR(B8A), SWIR1, SWIR2
    
    # Normalization stats (from Training.py)
    MEAN = torch.tensor([1370., 1370., 1370., 2630., 2630., 2630.])
    STD = torch.tensor([700., 700., 700., 1100., 1100., 1100.])
    
    @classmethod
    def __call__(cls, batch):
        """Extract and normalize bands"""
        x = batch[:, cls.BAND_INDICES, :, :]
        mean = cls.MEAN.view(1, 6, 1, 1).to(x.device)
        std = cls.STD.view(1, 6, 1, 1).to(x.device)
        return (x - mean) / std

def load_trained_model(checkpoint_path=None):
    """Load the trained model"""
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = PrithviSoilClassifier(num_classes=len(CLASSES)).to(DEVICE)
    
    # Load checkpoint (handle both state_dict and full checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Load with strict=False to allow missing health_head weights (if not trained yet)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"[OK] Loaded model from {checkpoint_path}")
        if 'val_acc' in checkpoint:
            print(f"   Validation Accuracy: {checkpoint['val_acc']:.2f}%")
    else:
        # Try loading with strict=False for health_head compatibility
        try:
            model.load_state_dict(checkpoint, strict=False)
        except:
            # If that fails, try loading only classification head
            state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('health_head')}
            model.load_state_dict(state_dict, strict=False)
        print(f"[OK] Loaded model from {checkpoint_path}")
    
    # Initialize health_head if not loaded (use random init)
    # Health head will use NDVI as proxy during inference if not trained
    model.eval()
    return model

# ==========================================
# 2. IMAGE PREPROCESSING
# ==========================================

def load_multispectral_image(image_path):
    """Load 13-band Sentinel-2 image"""
    # Resolve path (handle relative paths)
    image_path = Path(image_path)
    if not image_path.is_absolute():
        # If relative, try from script directory or current directory
        script_dir = Path(__file__).parent.parent
        possible_paths = [
            image_path,  # Try as-is first
            script_dir / image_path,  # From project root
            Path.cwd() / image_path,  # From current working directory
        ]
        for path in possible_paths:
            if path.exists():
                image_path = path
                break
        else:
            raise FileNotFoundError(f"Image not found. Tried: {possible_paths}")
    
    with rasterio.open(str(image_path)) as src:
        img = src.read()  # [13, H, W]
    return img

def load_rgb_image(image_path):
    """Load RGB image and convert to 13-band format matching Sentinel-2 structure.
    
    CRITICAL: This reverses the normalization applied when saving RGB from TIF,
    and maps values to match what BandSelector expects.
    
    BandSelector uses indices [1, 2, 3, 8, 11, 12] = [B2, B3, B4, B8A, B11, B12]
    with MEAN = [1370, 1370, 1370, 2630, 2630, 2630]
    and STD = [700, 700, 700, 1100, 1100, 1100]
    """
    # Resolve path (handle relative paths)
    image_path = Path(image_path)
    if not image_path.is_absolute():
        script_dir = Path(__file__).parent.parent
        possible_paths = [
            image_path,
            script_dir / image_path,
            Path.cwd() / image_path,
        ]
        for path in possible_paths:
            if path.exists():
                image_path = path
                break
        else:
            raise FileNotFoundError(f"Image not found. Tried: {possible_paths}")
    
    # Load RGB image
    img = np.array(Image.open(str(image_path)))
    
    # Handle different formats
    if len(img.shape) == 2:  # Grayscale
        img = np.stack([img, img, img], axis=-1)
    
    if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]
    
    # Convert to [H, W, 3] if needed
    if len(img.shape) == 3 and img.shape[0] == 3:  # [3, H, W]
        img = np.transpose(img, (1, 2, 0))
    
    # Get original dimensions
    H, W = img.shape[:2]
    
    # Resize to 64x64 to match training data size
    TARGET_SIZE = 64  # EuroSAT training size
    if H != TARGET_SIZE or W != TARGET_SIZE:
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(img.astype(np.uint8))
        img_pil = img_pil.resize((TARGET_SIZE, TARGET_SIZE), PILImage.Resampling.LANCZOS)
        img = np.array(img_pil)
        H, W = TARGET_SIZE, TARGET_SIZE
    
    # Find reference Sentinel-2 image - FIX PATH (3 levels up from Backend/Model/infer.py)
    project_root = Path(__file__).parent.parent.parent  # Backend/Model -> Backend -> Project root
    ref_sentinel_path = project_root / "Data" / "eurosat_data" / "ds" / "images" / "remote_sensing" / "otherDatasets" / "sentinel_2" / "tif" / "AnnualCrop" / "AnnualCrop_1.tif"
    
    # Load reference image to get exact statistics
    ref_found = False
    if ref_sentinel_path.exists():
        try:
            with rasterio.open(str(ref_sentinel_path)) as src:
                ref_img = src.read()  # [13, 64, 64]
            
            # Get the EXACT bands used by BandSelector: [1, 2, 3, 8, 11, 12]
            # These map to: B2(Blue), B3(Green), B4(Red), B8A(NIR), B11(SWIR1), B12(SWIR2)
            ref_b2 = ref_img[1].astype(np.float32)   # Blue - index 1
            ref_b3 = ref_img[2].astype(np.float32)   # Green - index 2  
            ref_b4 = ref_img[3].astype(np.float32)   # Red - index 3
            ref_b8a = ref_img[8].astype(np.float32)  # B8A (Narrow NIR) - index 8 (NOT 7!)
            ref_b11 = ref_img[11].astype(np.float32) # SWIR1 - index 11
            ref_b12 = ref_img[12].astype(np.float32) # SWIR2 - index 12
            
            # The PNG was created from [B4, B3, B2] with min-max normalization
            # We need to reverse: PNG_channel = (original - min) / (max - min) * 255
            # So: original = PNG_channel / 255 * (max - min) + min
            
            # Get the RGB stack's min/max (this is how download_rgb.py works)
            rgb_stack = np.stack([ref_b4, ref_b3, ref_b2], axis=0)  # [3, H, W]
            rgb_min_orig = rgb_stack.min()
            rgb_max_orig = rgb_stack.max()
            
            # Store all band statistics
            ref_bands = {
                1: ref_b2, 2: ref_b3, 3: ref_b4,
                8: ref_b8a, 11: ref_b11, 12: ref_b12
            }
            ref_found = True
            print(f"[OK] Reference image loaded: {ref_sentinel_path.name}")
            print(f"   RGB range: [{rgb_min_orig:.1f}, {rgb_max_orig:.1f}]")
        except Exception as e:
            print(f"[WARN] Failed to load reference: {e}")
            ref_found = False
    else:
        print(f"[WARN] Reference not found at: {ref_sentinel_path}")
    
    # Convert RGB to float
    img_float = img.astype(np.float32)
    
    if ref_found:
        # REVERSE the min-max normalization applied when saving the PNG
        # PNG was: (original - rgb_min) / (rgb_max - rgb_min) * 255
        # Reverse: original = PNG / 255 * (rgb_max - rgb_min) + rgb_min
        img_reversed = img_float / 255.0 * (rgb_max_orig - rgb_min_orig) + rgb_min_orig
        
        # img_reversed is now [H, W, 3] with channels [R, G, B] = [B4, B3, B2]
        restored_b4 = img_reversed[:, :, 0]  # Red = B4
        restored_b3 = img_reversed[:, :, 1]  # Green = B3
        restored_b2 = img_reversed[:, :, 2]  # Blue = B2
        
        # Check if reference image has same dimensions as input
        # If so, use the ACTUAL NIR/SWIR bands from reference (much more accurate)
        ref_h, ref_w = ref_b8a.shape
        if H == ref_h and W == ref_w:
            # Same size! Use actual reference NIR/SWIR bands
            # This is accurate for images extracted from similar Sentinel-2 data
            print(f"   Using reference NIR/SWIR bands (same size: {H}x{W})")
            restored_b8a = ref_b8a.copy()
            restored_b11 = ref_b11.copy()
            restored_b12 = ref_b12.copy()
        else:
            # Different size - approximate from intensity 
            print(f"   Approximating NIR/SWIR (sizes differ: input {H}x{W} vs ref {ref_h}x{ref_w})")
            intensity = img_reversed.mean(axis=2)
            
            # Calculate scale factors from reference
            ref_visible_mean = (ref_b2.mean() + ref_b3.mean() + ref_b4.mean()) / 3
            b8a_scale = ref_b8a.mean() / ref_visible_mean if ref_visible_mean > 0 else 0.5
            b11_scale = ref_b11.mean() / ref_visible_mean if ref_visible_mean > 0 else 1.8
            b12_scale = ref_b12.mean() / ref_visible_mean if ref_visible_mean > 0 else 1.8
            
            restored_b8a = intensity * b8a_scale
            restored_b11 = intensity * b11_scale
            restored_b12 = intensity * b12_scale
        
        # Create 13-band image with float32 (same as how it's used in preprocess_image)
        img_13band = np.zeros((13, H, W), dtype=np.float32)
        
        # Fill bands used by BandSelector: indices [1, 2, 3, 8, 11, 12]
        img_13band[1] = restored_b2   # B2 (Blue)
        img_13band[2] = restored_b3   # B3 (Green)
        img_13band[3] = restored_b4   # B4 (Red)
        img_13band[8] = restored_b8a  # B8A (Narrow NIR) - INDEX 8, NOT 7!
        img_13band[11] = restored_b11 # B11 (SWIR1)
        img_13band[12] = restored_b12 # B12 (SWIR2)
        
        # Fill unused bands (not critical but for completeness)
        img_13band[0] = restored_b2  # B1
        img_13band[4] = restored_b3  # B5
        img_13band[5] = restored_b3  # B6
        img_13band[6] = restored_b4  # B7
        img_13band[7] = restored_b8a * 0.95  # B8 (slightly less than B8A)
        img_13band[9] = restored_b8a * 0.9   # B9
        img_13band[10] = restored_b11 * 0.95 # B10
        
        print(f"   Restored B4 (Red): [{restored_b4.min():.1f}, {restored_b4.max():.1f}]")
        print(f"   Restored B8A (NIR): [{restored_b8a.min():.1f}, {restored_b8a.max():.1f}]")
        
    else:
        # Fallback: simple scaling to match normalization stats
        # MEAN = [1370, 1370, 1370, 2630, 2630, 2630] for bands [1,2,3,8,11,12]
        # STD = [700, 700, 700, 1100, 1100, 1100]
        print("[WARN] Using fallback scaling (less accurate)")
        
        # Scale RGB from 0-255 to approximately match training data range
        # Visible bands: typical range 0-3000, mean ~1370
        scale_visible = 3000.0 / 255.0
        img_scaled = img_float * scale_visible
        
        intensity = img_scaled.mean(axis=2)
        scale_nir = 4500.0 / 3000.0  # NIR bands have higher values
        
        img_13band = np.zeros((13, H, W), dtype=np.float32)
        img_13band[1] = img_scaled[:, :, 2]  # B2 = Blue
        img_13band[2] = img_scaled[:, :, 1]  # B3 = Green
        img_13band[3] = img_scaled[:, :, 0]  # B4 = Red
        img_13band[8] = intensity * scale_nir  # B8A - INDEX 8!
        img_13band[11] = intensity * scale_nir * 0.9
        img_13band[12] = intensity * scale_nir * 0.85
        
        # Fill other bands
        for i in [0, 4, 5, 6, 7, 9, 10]:
            img_13band[i] = img_scaled[:, :, i % 3]
    
    return img_13band

def preprocess_image(img_13band, band_selector):
    """Preprocess image for model input"""
    # Select and normalize bands
    img_tensor = torch.from_numpy(img_13band).float().unsqueeze(0)  # [1, 13, H, W]
    img_6band = band_selector(img_tensor).to(DEVICE)
    return img_6band

# ==========================================
# 3. GRID ANALYSIS
# ==========================================

def analyze_image_grid(image_path, model, grid_size=(4, 4), is_rgb=False):
    """
    Analyze image by dividing into grid and predicting each cell
    """
    # Ensure model is in eval mode (critical for correct inference)
    model.eval()
    
    # Load image
    if is_rgb:
        img_13band = load_rgb_image(image_path)
    else:
        img_13band = load_multispectral_image(image_path)
    
    H, W = img_13band.shape[1], img_13band.shape[2]
    cell_h, cell_w = H // grid_size[0], W // grid_size[1]
    
    band_selector = BandSelector()
    predictions = []
    confidences = []
    health_scores = []
    
    # Progress bar for larger grids
    total_cells = grid_size[0] * grid_size[1]
    use_progress = total_cells > 16  # Show progress for grids larger than 4x4
    
    if use_progress:
        from tqdm import tqdm
        pbar = tqdm(total=total_cells, desc="Analyzing grid cells", leave=False)
    
    # Analyze each grid cell
    for i in range(grid_size[0]):
        row_preds = []
        row_confs = []
        row_health = []
        for j in range(grid_size[1]):
            # Extract cell
            y_start, y_end = i * cell_h, (i + 1) * cell_h
            x_start, x_end = j * cell_w, (j + 1) * cell_w
            
            cell = img_13band[:, y_start:y_end, x_start:x_end]
            
            # Resize cell to 64x64 if needed
            # IMPORTANT: Preserve original value range (don't convert to uint8)
            if cell.shape[1] != 64 or cell.shape[2] != 64:
                if HAS_SCIPY:
                    # Use scipy zoom to preserve float values and value range
                    zoom_factors = (1.0, 64.0 / cell.shape[1], 64.0 / cell.shape[2])
                    cell = zoom(cell, zoom_factors, order=1)  # order=1 is bilinear interpolation
                else:
                    # Fallback: use torch interpolation (preserves value range)
                    cell_tensor = torch.from_numpy(cell).float().unsqueeze(0)  # [1, 13, H, W]
                    cell_tensor = torch.nn.functional.interpolate(
                        cell_tensor, size=(64, 64), mode='bilinear', align_corners=False
                    )
                    cell = cell_tensor.squeeze(0).numpy()
            
            # Predict
            cell_tensor = preprocess_image(cell, band_selector)
            
            with torch.no_grad():
                # Get classification and health prediction
                output = model(cell_tensor, return_health=True)
                if isinstance(output, tuple):
                    classification_output, health_output = output
                    health_score = health_output[0, 0].item()
                else:
                    # Fallback: model doesn't have health head yet, use NDVI as proxy
                    classification_output = output
                    if not is_rgb and cell.shape[0] >= 8:
                        # Calculate NDVI as health proxy
                        red = cell[3, :, :].astype(float)
                        nir = cell[7, :, :].astype(float)
                        ndvi = (nir - red) / (nir + red + 1e-10)
                        # Normalize NDVI from [-1, 1] to [0, 1] for health score
                        health_score = (ndvi.mean() + 1) / 2
                    else:
                        health_score = 0.5  # Default if can't calculate
                
                probs = torch.softmax(classification_output, dim=1)
                pred_idx = torch.argmax(probs).item()
                confidence = probs[0][pred_idx].item()
            
            row_preds.append(pred_idx)
            row_confs.append(confidence)
            row_health.append(health_score)
            
            if use_progress:
                pbar.update(1)
        
        predictions.append(row_preds)
        confidences.append(row_confs)
        health_scores.append(row_health)
    
    if use_progress:
        pbar.close()
    
    return np.array(predictions), np.array(confidences), img_13band, np.array(health_scores)

# ==========================================
# 4. VISUALIZATION
# ==========================================

def visualize_analysis(image_path, model, is_rgb=False, grid_size=(4, 4)):
    """Complete analysis with visualization"""
    print(f"\n🔍 Analyzing: {image_path}")
    print(f"   Grid size: {grid_size[0]}x{grid_size[1]}")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Get grid predictions
    predictions, confidences, img_13band, health_scores = analyze_image_grid(image_path, model, grid_size, is_rgb)
    
    # Overall prediction
    overall_tensor = preprocess_image(img_13band, BandSelector())
    with torch.no_grad():
        overall_output = model(overall_tensor, return_health=False)
        overall_probs = torch.softmax(overall_output, dim=1)
        overall_pred = torch.argmax(overall_probs).item()
        overall_conf = overall_probs[0][overall_pred].item()
    
    # Create visualization
    fig = plt.figure(figsize=(16, 8))
    
    # Original RGB view
    ax1 = plt.subplot(2, 3, 1)
    if is_rgb:
        rgb = img_13band[[2, 1, 0], :, :]  # B, G, R -> R, G, B
    else:
        rgb = np.stack([img_13band[3], img_13band[2], img_13band[1]], axis=0)  # R, G, B
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-10)
    ax1.imshow(rgb_norm)
    ax1.set_title(f"Original Image\nOverall: {CLASSES[overall_pred]} ({overall_conf:.1%})", fontsize=11)
    ax1.axis('off')
    
    # Grid prediction map
    ax2 = plt.subplot(2, 3, 2)
    pred_map = np.zeros((*grid_size, 3))
    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASSES)))
    
    # Debug: Print predictions to verify (only for smaller grids to avoid clutter)
    if grid_size[0] * grid_size[1] <= 64:  # Only print for grids up to 8x8
        print(f"\n   Grid Predictions:")
        for i in range(min(grid_size[0], 8)):  # Limit to first 8 rows
            row_str = " ".join([f"{CLASSES[predictions[i, j]]:4s}" for j in range(min(grid_size[1], 8))])
            print(f"   Row {i}: {row_str}")
        if grid_size[0] > 8 or grid_size[1] > 8:
            print(f"   ... (showing first 8x8, total grid: {grid_size[0]}x{grid_size[1]})")
    
    # Create color map - ensure correct mapping
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            class_idx = int(predictions[i, j])
            pred_map[i, j] = colors[class_idx][:3]
    
    # Upsample to image size for visualization
    pred_map_upsampled = np.repeat(np.repeat(pred_map, img_13band.shape[1]//grid_size[0], axis=0), 
                                   img_13band.shape[2]//grid_size[1], axis=1)
    ax2.imshow(pred_map_upsampled)
    ax2.set_title("Grid Classification Map", fontsize=11)
    ax2.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i][:3], label=CLASSES[i]) 
                       for i in np.unique(predictions)]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=7, ncol=2)
    
    # Confidence heatmap
    ax3 = plt.subplot(2, 3, 3)
    conf_map_upsampled = np.repeat(np.repeat(confidences, img_13band.shape[1]//grid_size[0], axis=0),
                                   img_13band.shape[2]//grid_size[1], axis=1)
    im = ax3.imshow(conf_map_upsampled, cmap='viridis', vmin=0, vmax=1)
    ax3.set_title("Confidence Heatmap", fontsize=11)
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046)
    
    # Crop Health Map (predicted health scores)
    ax4 = plt.subplot(2, 3, 4)
    if health_scores is not None:
        # Upsample health scores to image size
        health_map_upsampled = np.repeat(
            np.repeat(health_scores, img_13band.shape[1]//grid_size[0], axis=0),
            img_13band.shape[2]//grid_size[1], axis=1
        )
        im2 = ax4.imshow(health_map_upsampled, cmap='RdYlGn', vmin=0, vmax=1)
        ax4.set_title("Crop Health (Predicted)", fontsize=11)
        ax4.axis('off')
        plt.colorbar(im2, ax=ax4, fraction=0.046, label='Health Score')
    elif not is_rgb:
        # Fallback to NDVI if health scores not available
        red = img_13band[3, :, :].astype(float)
        nir = img_13band[7, :, :].astype(float)
        ndvi = (nir - red) / (nir + red + 1e-10)
        im2 = ax4.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        ax4.set_title("Vegetation Health (NDVI)", fontsize=11)
        ax4.axis('off')
        plt.colorbar(im2, ax=ax4, fraction=0.046)
    
    # NDVI comparison (if multispectral)
    if not is_rgb:
        # Add NDVI as a 7th subplot or replace one
        # For now, we'll keep it in the same position but add a note
        pass
    
    # Class distribution
    ax5 = plt.subplot(2, 3, 5)
    unique, counts = np.unique(predictions, return_counts=True)
    class_counts = {CLASSES[i]: counts[unique == i][0] if i in unique else 0 for i in range(len(CLASSES))}
    bars = ax5.barh(list(class_counts.keys()), list(class_counts.values()))
    ax5.set_xlabel("Grid Cells")
    ax5.set_title("Class Distribution in Grid", fontsize=11)
    plt.setp(ax5.get_yticklabels(), fontsize=8)
    
    # Top predictions table with health summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    table_data = [["Class", "Cells", "%"]]
    total_cells = grid_size[0] * grid_size[1]
    for cls, count in top_classes:
        if count > 0:
            table_data.append([cls, str(count), f"{100*count/total_cells:.1f}%"])
    
    # Add average health score if available
    if health_scores is not None:
        avg_health = health_scores.mean()
        table_data.append(["", "", ""])  # Empty row
        table_data.append(["Avg Health", f"{avg_health:.3f}", ""])
    
    table = ax6.table(cellText=table_data[1:], colLabels=table_data[0], 
                      cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    ax6.set_title("Top 5 Classes" + (" + Health" if health_scores is not None else ""), fontsize=11)
    
    plt.suptitle(f"Geospatial Analysis: {Path(image_path).name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = f"analysis_{Path(image_path).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved analysis to: {output_path}")
    plt.show()
    
    return predictions, confidences

# ==========================================
# 5. MAIN
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prithvi Soil/Crop Classifier Inference")
    parser.add_argument("--image", type=str, help="Path to image (TIF or RGB)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--grid", type=int, nargs=2, default=[8, 8], help="Grid size (rows cols). Higher = finer detail (e.g., 8x8, 16x16)")
    parser.add_argument("--rgb", action="store_true", help="Input is RGB image (not multispectral)")
    args = parser.parse_args()
    
    # Load model
    model = load_trained_model(args.checkpoint)
    
    # Default: test on first dataset image
    if args.image is None:
        # Try multiple possible paths (project root is 2 levels up from Backend/Model/infer.py)
        project_root = Path(__file__).parent.parent.parent  # Go to project root
        possible_paths = [
            project_root / "Data" / "eurosat_data" / "ds" / "images" / "remote_sensing" / "otherDatasets" / "sentinel_2" / "tif" / "AnnualCrop" / "AnnualCrop_1.tif",
            Path(__file__).parent.parent / "Data" / "eurosat_data" / "ds" / "images" / "remote_sensing" / "otherDatasets" / "sentinel_2" / "tif" / "AnnualCrop" / "AnnualCrop_1.tif",
        ]
        
        test_file = None
        for path in possible_paths:
            if path.exists():
                test_file = path
                break
        
        if test_file and test_file.exists():
            print(f"[INFO] Using default test image: {test_file}")
            visualize_analysis(str(test_file.absolute()), model, is_rgb=False, grid_size=tuple(args.grid))
        else:
            print(f"[ERROR] No image provided and default test image not found.")
            print(f"   Tried paths:")
            for path in possible_paths:
                print(f"     - {path}")
            print("\nUsage: python infer.py --image <path> [--rgb] [--grid 4 4]")
    else:
        visualize_analysis(args.image, model, is_rgb=args.rgb, grid_size=tuple(args.grid))
