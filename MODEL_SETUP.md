# Model Files Setup

The trained model files are too large for Git (600+ MB). They need to be set up separately.

## Required Model Files

1. **Base Pre-trained Model:**
   - `Backend/prithvi_model/Prithvi_EO_V1_100M.pt` (~400 MB)
   - Download from: [NASA Prithvi Model](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) or your original source

2. **Fine-tuned Model Checkpoints:**
   - `Backend/Model/checkpoints/prithvi_soil_best.pth` (~200 MB)
   - `Backend/Model/checkpoints/prithvi_soil_final.pth` (~200 MB)
   - These are your trained models - store them separately or use Git LFS

## Setup Options

### Option 1: Git LFS (Recommended for version control)
```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "Backend/prithvi_model/Prithvi_EO_V1_100M.pt"
git lfs track "Backend/Model/checkpoints/*.pth"

# Add .gitattributes
git add .gitattributes

# Add and commit model files
git add Backend/prithvi_model/Prithvi_EO_V1_100M.pt
git add Backend/Model/checkpoints/*.pth
git commit -m "Add model files via Git LFS"
git push
```

### Option 2: Store Separately (Recommended for large files)
- Upload model files to cloud storage (Google Drive, Dropbox, AWS S3, etc.)
- Share download links in documentation
- Users download and place files in correct directories

### Option 3: Download Script
Create a setup script that downloads models automatically:
```bash
# Example: setup_models.sh
# Downloads models from cloud storage or HuggingFace
```

## Directory Structure After Setup

```
Backend/
├── prithvi_model/
│   ├── Prithvi_EO_V1_100M.pt  ← Download separately
│   ├── config.json
│   └── prithvi_mae.py
└── Model/
    └── checkpoints/
        ├── prithvi_soil_best.pth  ← Download separately
        └── prithvi_soil_final.pth  ← Download separately
```

## Quick Start

1. Clone the repository
2. Download model files to the locations above
3. Install dependencies: `pip install -r Backend/API/requirements.txt`
4. Run the backend: `cd Backend/API && uvicorn app:app --reload`
