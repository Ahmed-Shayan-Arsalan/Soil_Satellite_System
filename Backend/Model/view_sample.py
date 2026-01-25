"""
Visualize the first image from the EuroSAT dataset
"""
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_first_image():
    """Load and display the first image from AnnualCrop class"""
    # Find dataset path
    script_dir = Path(__file__).parent.parent
    dataset_path = script_dir / "Data" / "eurosat_data" / "ds" / "images" / "remote_sensing" / "otherDatasets" / "sentinel_2" / "tif" / "AnnualCrop"
    
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
        print(f"   Data type: {img.dtype}")
        print(f"   Value range: [{img.min():.1f}, {img.max():.1f}]")
    
    # Create RGB visualization (B4=Red, B3=Green, B2=Blue)
    rgb = np.stack([img[3], img[2], img[1]], axis=0)  # [3, H, W]
    rgb = np.transpose(rgb, (1, 2, 0))  # [H, W, 3]
    
    # Normalize to 0-1 for display
    rgb_normalized = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-10)
    
    # Calculate NDVI for health visualization
    red = img[3, :, :].astype(float)
    nir = img[7, :, :].astype(float)  # B8 is index 7
    ndvi = (nir - red) / (nir + red + 1e-10)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB View
    axes[0].imshow(rgb_normalized)
    axes[0].set_title("RGB Natural Color View\n(B4=Red, B3=Green, B2=Blue)", fontsize=12)
    axes[0].axis('off')
    
    # NDVI Health Map
    im1 = axes[1].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1].set_title("Vegetation Health (NDVI)\nGreen=Healthy | Red=Stressed", fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # False Color (NIR, Red, Green) - better for vegetation
    false_color = np.stack([img[7], img[3], img[2]], axis=0)  # NIR, Red, Green
    false_color = np.transpose(false_color, (1, 2, 0))
    false_color_norm = (false_color - false_color.min()) / (false_color.max() - false_color.min() + 1e-10)
    axes[2].imshow(false_color_norm)
    axes[2].set_title("False Color (NIR-Red-Green)\nBetter for Vegetation Analysis", fontsize=12)
    axes[2].axis('off')
    
    plt.suptitle(f"Sample Image: {first_image.name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("sample_image_visualization.png", dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: sample_image_visualization.png")
    plt.show()

if __name__ == "__main__":
    visualize_first_image()

