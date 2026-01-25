import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchgeo.datasets import EuroSAT
import os
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. GPU & SYSTEM SETUP
# ==========================================
# GPU Detection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"🚀 CUDA Available! Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # Faster matrix ops on Ampere+
    torch.backends.cudnn.allow_tf32 = True
else:
    DEVICE = torch.device("cpu")
    print("⚠️  CUDA not available, using CPU (training will be slow)")

NUM_WORKERS = 0 if os.name == 'nt' else 4  # Windows compatibility

# ==========================================
# 2. CONFIGURATION
# ==========================================
class Config:
    # Model
    num_classes = 10  # EuroSAT classes
    freeze_backbone = True  # Fine-tuning: freeze Prithvi backbone
    unfreeze_last_n_layers = 2  # Unfreeze last N transformer blocks
    
    # Training
    batch_size = 16
    epochs = 25
    base_lr = 1e-4
    backbone_lr_multiplier = 0.1  # Lower LR for backbone
    weight_decay = 0.01
    warmup_epochs = 3
    grad_clip_norm = 1.0
    grad_accumulation_steps = 2  # Effective batch = batch_size * this
    
    # Early Stopping
    patience = 5
    min_delta = 0.001
    
    # Paths
    save_dir = Path("checkpoints")
    best_model_name = "prithvi_soil_best.pth"
    final_model_name = "prithvi_soil_final.pth"

# ==========================================
# 3. MODEL ARCHITECTURE (Fine-Tuning Ready)
# ==========================================

class PrithviSoilClassifier(nn.Module):
    """
    Prithvi-100M Fine-Tuned for Geospatial Crop/Soil Analysis
    - Frozen backbone with trainable top layers
    - Enhanced classification head
    """
    def __init__(self, num_classes=10, freeze_backbone=True, unfreeze_last_n=2):
        super().__init__()
        
        print("📥 Loading Prithvi-100M backbone from local checkpoint...")
        
        # Find prithvi_model directory (could be in project root or parent)
        script_dir = Path(__file__).parent.parent  # Go from Model/ to project root
        possible_paths = [
            script_dir / "prithvi_model",
            Path("prithvi_model"),
            Path("../prithvi_model"),
        ]
        
        prithvi_model_dir = None
        for path in possible_paths:
            if path.exists() and (path / "Prithvi_EO_V1_100M.pt").exists():
                prithvi_model_dir = path
                print(f"  📂 Found model at: {prithvi_model_dir.absolute()}")
                break
        
        if prithvi_model_dir is None:
            raise FileNotFoundError(
                f"Could not find prithvi_model directory. Tried: {[str(p) for p in possible_paths]}\n"
                "Please ensure prithvi_model/ directory exists with Prithvi_EO_V1_100M.pt"
            )
        
        # Add prithvi_model to Python path to import the model code
        if str(prithvi_model_dir) not in sys.path:
            sys.path.insert(0, str(prithvi_model_dir))
        
        # Import PrithviMAE from local file
        from prithvi_mae import PrithviMAE
        
        # Load config.json
        config_path = prithvi_model_dir / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        pretrained_cfg = config['pretrained_cfg']
        
        # Create PrithviMAE model with encoder_only=True (we only need encoder)
        print("  🔧 Creating Prithvi encoder...")
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
            encoder_only=True  # Only load encoder, not decoder
        )
        
        # Load checkpoint
        checkpoint_path = prithvi_model_dir / "Prithvi_EO_V1_100M.pt"
        print(f"  📦 Loading weights from: {checkpoint_path.name}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract encoder weights (checkpoint might have 'model' key or direct keys)
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Filter to only encoder weights and remove 'encoder.' prefix
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                # Remove 'encoder.' prefix
                new_key = k.replace('encoder.', '')
                encoder_state_dict[new_key] = v
            elif not k.startswith('decoder.'):  # Include non-decoder keys (might be encoder)
                encoder_state_dict[k] = v
        
        # Load encoder weights (strict=False to handle missing keys)
        missing_keys, unexpected_keys = prithvi_model.encoder.load_state_dict(encoder_state_dict, strict=False)
        if missing_keys:
            print(f"  ⚠️  Missing keys (will use random init): {len(missing_keys)}")
        if unexpected_keys:
            print(f"  ⚠️  Unexpected keys (ignored): {len(unexpected_keys)}")
        
        # Use encoder as backbone
        self.backbone = prithvi_model.encoder
        print("  ✅ Prithvi encoder loaded successfully!")
        
        # === FREEZE STRATEGY FOR FINE-TUNING ===
        if freeze_backbone:
            self._freeze_backbone(unfreeze_last_n)
        
        # Enhanced classification head with GELU + LayerNorm
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
        # Uses NDVI as ground truth during training
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
        
        # Initialize head weights properly
        self._init_head()
        self._init_health_head()
        
    def _freeze_backbone(self, unfreeze_last_n):
        """Freeze backbone except last N transformer blocks"""
        # Freeze all backbone parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last N transformer blocks (Prithvi uses 'blocks', not 'layers')
        if hasattr(self.backbone, 'blocks'):
            total_layers = len(self.backbone.blocks)
            for i, block in enumerate(self.backbone.blocks):
                if i >= total_layers - unfreeze_last_n:
                    for param in block.parameters():
                        param.requires_grad = True
                    print(f"  🔓 Unfroze transformer block {i}")
        
        # Also unfreeze final norm layer
        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
                
        frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        trainable = sum(1 for p in self.backbone.parameters() if p.requires_grad)
        print(f"  ❄️  Backbone: {frozen} frozen, {trainable} trainable params")
    
    def _init_head(self):
        """Kaiming initialization for classification head"""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _init_health_head(self):
        """Kaiming initialization for health regression head"""
        for m in self.health_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_health=False):
        # Input x: [B, 6, H, W] -> Prithvi expects [B, C, T, H, W]
        # Prithvi encoder expects [B, C, T, H, W] format
        if len(x.shape) == 4:  # [B, C, H, W]
            x = x.unsqueeze(2)  # Add temporal dimension -> [B, C, T, H, W]
        
        # Use forward_features to get features (returns list of layer outputs)
        features_list = self.backbone.forward_features(x)
        # Get final layer output (after norm)
        final_features = features_list[-1]  # [B, N+1, embed_dim] where N+1 includes CLS token
        
        # Extract CLS token (first token)
        cls_features = final_features[:, 0, :]  # [B, embed_dim]
        
        classification = self.head(cls_features)
        if return_health:
            health = self.health_head(cls_features)
            return classification, health
        return classification
    
    def get_param_groups(self, base_lr, backbone_lr_mult=0.1):
        """Separate param groups with different learning rates"""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        
        return [
            {'params': backbone_params, 'lr': base_lr * backbone_lr_mult, 'name': 'backbone'},
            {'params': head_params, 'lr': base_lr, 'name': 'head'}
        ]

# ==========================================
# 4. DATA LOADING & PREPROCESSING
# ==========================================

class BandSelector:
    """Select and normalize bands for Prithvi model"""
    # Prithvi bands: Blue, Green, Red, Narrow-NIR(B8A), SWIR1, SWIR2
    BAND_INDICES = [1, 2, 3, 8, 11, 12]
    
    # Normalization stats (approximate for Sentinel-2)
    MEAN = torch.tensor([1370., 1370., 1370., 2630., 2630., 2630.])
    STD = torch.tensor([700., 700., 700., 1100., 1100., 1100.])
    
    @classmethod
    def __call__(cls, batch):
        """Extract and normalize bands"""
        x = batch[:, cls.BAND_INDICES, :, :]
        # Normalize
        mean = cls.MEAN.view(1, 6, 1, 1).to(x.device)
        std = cls.STD.view(1, 6, 1, 1).to(x.device)
        return (x - mean) / std

def get_dataloaders(batch_size=32, val_split=0.2, seed=42, root=None):
    """Create train/val dataloaders with proper seeding"""
    # Auto-detect dataset path if not provided
    if root is None:
        # Get script directory (Model/) and go up to project root
        script_dir = Path(__file__).parent.parent  # Go from Model/ to project root
        # Try common locations
        possible_paths = [
            script_dir / "Data" / "eurosat_data",
            script_dir / "Data" / "eurosat_data" / "ds",  # Try ds subdirectory
            Path("Data/eurosat_data"),
            Path("eurosat_data"),
            Path("../Data/eurosat_data"),
        ]
        
        for path in possible_paths:
            if path.exists():
                root = str(path.absolute())
                print(f"📂 Found dataset at: {root}")
                break
        
        if root is None:
            raise FileNotFoundError(
                f"Could not find eurosat_data. Tried: {[str(p) for p in possible_paths]}\n"
                "Please specify the correct path in get_dataloaders(root='...')"
            )
    
    # Try loading dataset, with fallback to ds subdirectory if needed
    try:
        dataset = EuroSAT(root=root)
    except Exception as e:
        # If it fails and we're not already in ds, try ds subdirectory
        root_path = Path(root)
        if root_path.name != "ds" and (root_path / "ds").exists():
            print(f"⚠️  Dataset verification failed at: {root}")
            print(f"   Trying alternative path: {root_path / 'ds'}")
            root = str(root_path / "ds")
            dataset = EuroSAT(root=root)
            print(f"✅ Successfully loaded from: {root}")
        else:
            raise
    
    # Reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    
    print(f"📊 Dataset: {len(train_ds)} train, {len(val_ds)} val samples")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=torch.cuda.is_available(),  # Only pin memory if CUDA available
        drop_last=True,  # Consistent batch sizes
        persistent_workers=NUM_WORKERS > 0
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size * 2,  # Larger batch for validation (no grads)
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=torch.cuda.is_available(),  # Only pin memory if CUDA available
        persistent_workers=NUM_WORKERS > 0
    )
    return train_loader, val_loader

# ==========================================
# 5. TRAINING ENGINE
# ==========================================

class EarlyStopping:
    """Stop training when validation loss stops improving"""
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # New best
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

def get_lr_scheduler(optimizer, epochs, warmup_epochs, steps_per_epoch):
    """Cosine schedule with linear warmup"""
    def lr_lambda(step):
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = epochs * steps_per_epoch
        
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

@torch.no_grad()
def validate(model, val_loader, criterion, band_selector):
    """Validation loop with accuracy metrics"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in val_loader:
        images = band_selector(batch["image"]).to(DEVICE)
        labels = batch["label"].to(DEVICE, dtype=torch.long)
        
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, 100.0 * correct / total

def train_model(model, train_loader, val_loader, config):
    """
    Full training loop with:
    - Mixed precision (AMP)
    - Gradient accumulation
    - Learning rate warmup
    - Early stopping
    - Best model checkpointing
    """
    config.save_dir.mkdir(exist_ok=True)
    band_selector = BandSelector()
    
    # Optimizer with layer-wise learning rates
    param_groups = model.get_param_groups(config.base_lr, config.backbone_lr_multiplier)
    optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay)
    
    # Scheduler with warmup
    scheduler = get_lr_scheduler(
        optimizer, config.epochs, config.warmup_epochs, len(train_loader)
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing helps
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    early_stopping = EarlyStopping(config.patience, config.min_delta)
    
    best_acc = 0
    
    print("\n" + "="*60)
    print("🌱 TRAINING PRITHVI SOIL/CROP CLASSIFIER")
    print("="*60)
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
        
        for step, batch in enumerate(pbar):
            images = band_selector(batch["image"]).to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, dtype=torch.long, non_blocking=True)
            
            # Mixed Precision Forward
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / config.grad_accumulation_steps  # Scale for accumulation
            
            # Backward with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation step
            if (step + 1) % config.grad_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # Step scheduler after optimizer step (correct order)
                scheduler.step()
            
            # Metrics
            total_loss += loss.item() * config.grad_accumulation_steps
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Progress bar update
            pbar.set_postfix({
                'loss': f'{total_loss/(step+1):.4f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, band_selector)
        train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # GPU Memory
        mem = torch.cuda.memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0
        current_lr = scheduler.get_last_lr()[0]
        
        # Logging
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{config.epochs} | "
              f"Train: {train_loss:.4f} ({train_acc:.1f}%) | "
              f"Val: {val_loss:.4f} ({val_acc:.1f}%) | "
              f"LR: {current_lr:.2e} | VRAM: {mem:.1f}GB | {elapsed:.1f}s")
        
        # Checkpointing
        is_best = early_stopping(val_loss)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, config.save_dir / config.best_model_name)
            print(f"  💾 New best model saved! Val Acc: {val_acc:.2f}%")
        
        # Early stopping check
        if early_stopping.should_stop:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
    
    return best_acc

# ==========================================
# 6. EXECUTION
# ==========================================

if __name__ == "__main__":
    config = Config()
    
    # Data
    train_loader, val_loader = get_dataloaders(
        batch_size=config.batch_size
    )
    
    # Model
    print("\n--- Initializing Prithvi Soil/Crop Classifier ---")
    model = PrithviSoilClassifier(
        num_classes=config.num_classes,
        freeze_backbone=config.freeze_backbone,
        unfreeze_last_n=config.unfreeze_last_n_layers
    ).to(DEVICE)
    
    # Verify model is on correct device
    model_device = next(model.parameters()).device
    print(f"📍 Model device: {model_device}")
    if torch.cuda.is_available() and model_device.type != 'cuda':
        print("⚠️  WARNING: Model is not on GPU despite CUDA being available!")
    
    # Compile for PyTorch 2.0+ speedup (if available and not Windows)
    # Note: torch.compile() requires Triton which doesn't work well on Windows
    if hasattr(torch, 'compile') and torch.cuda.is_available() and os.name != 'nt':
        print("⚡ Compiling model with torch.compile()...")
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            print(f"  ⚠️  Compilation failed (continuing without): {e}")
    elif os.name == 'nt':
        print("  ℹ️  Skipping torch.compile() on Windows (Triton not supported)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📐 Parameters: {trainable_params:,} trainable / {total_params:,} total "
          f"({100*trainable_params/total_params:.1f}%)")
    
    # GPU memory info
    if torch.cuda.is_available():
        print(f"💾 GPU Memory - Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB, "
              f"Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Train
    best_accuracy = train_model(model, train_loader, val_loader, config)
    
    # Save final model
    torch.save(model.state_dict(), config.save_dir / config.final_model_name)
    
    print("\n" + "="*60)
    print(f"✅ Training Complete! Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"📁 Models saved to: {config.save_dir}")
    print("="*60)
