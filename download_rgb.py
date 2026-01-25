"""
Simple script to extract RGB from an AnnualCrop image and save it to project root
"""
import rasterio
import numpy as np
from PIL import Image
from pathlib import Path

def download_rgb_from_annual_crop():
    """Extract RGB from an AnnualCrop image and save to project root"""
    # Find dataset path
    project_root = Path(__file__).parent
    dataset_path = project_root / "Data" / "eurosat_data" / "ds" / "images" / "remote_sensing" / "otherDatasets" / "sentinel_2" / "tif" / "AnnualCrop"
    
    # Get first image
    image_files = sorted(list(dataset_path.glob("*.tif")))
    if not image_files:
        print(f"❌ No images found in {dataset_path}")
        return
    
    first_image = image_files[0]
    print(f"📂 Loading: {first_image.name}")
    
    # Load image
    with rasterio.open(first_image) as src:
        img = src.read()  # [13, 64, 64]
        print(f"   Shape: {img.shape} (13 bands, {img.shape[1]}x{img.shape[2]} pixels)")
    
    # Extract RGB bands (B4=Red index 3, B3=Green index 2, B2=Blue index 1)
    rgb = np.stack([img[3], img[2], img[1]], axis=0)  # [3, H, W]
    rgb = np.transpose(rgb, (1, 2, 0))  # [H, W, 3]
    
    # Normalize to 0-255 for saving as image
    rgb_min = rgb.min()
    rgb_max = rgb.max()
    rgb_normalized = ((rgb - rgb_min) / (rgb_max - rgb_min + 1e-10) * 255).astype(np.uint8)
    
    # Save RGB image to project root
    output_path = project_root / "annual_crop_rgb.png"
    rgb_image = Image.fromarray(rgb_normalized)
    rgb_image.save(output_path)
    
    print(f"✅ RGB image saved to: {output_path}")
    print(f"   Image size: {rgb_normalized.shape[1]}x{rgb_normalized.shape[0]} pixels")

if __name__ == "__main__":
    download_rgb_from_annual_crop()

