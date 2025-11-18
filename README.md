# Real-Time Infrared-Visible Fusion via Distilled Diffusion Models

**Adding Real-Time Infrared-Visible Fusion to Any Visual System via Knowledge Distillation**

This project implements a lightweight, real-time infrared-visible image fusion model by distilling knowledge from high-quality diffusion-based fusion models (Teacher) into an efficient student network.

## 🎯 Project Goals

- **Real-time Performance**: < 30ms inference on edge GPUs (NVIDIA Xavier/Orin)
- **Plug-and-Play**: Easy integration into any RGB-based visual system
- **High Quality**: Preserve Teacher model's fusion quality while achieving 50-100x speedup
- **Robustness**: Handle misalignment between IR and VIS inputs

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Results](#results)
- [Citation](#citation)

---

## 🏗️ Architecture

### Student Model Design

```
Input: IR Image (3×H×W) + VIS Image (3×H×W)
       ↓
  [Dual-Branch Lightweight Encoders]
  ├─ IR Encoder  (MobileNetV3-style)
  └─ VIS Encoder (MobileNetV3-style)
       ↓
  [Multi-Scale Cross-Modal Fusion]
  - Channel Attention
  - Spatial Cross-Attention
  - SE Modules
       ↓
  [U-Net Decoder]
  - Skip Connections
  - Multi-scale Upsampling
       ↓
Output: Fused Image (3×H×W)
```

**Key Features:**
- **Parameters**: ~3-5M (vs. 100-500M for Teacher)
- **FLOPs**: ~15-25G (512×512 input)
- **Inference**: <30ms on Xavier, <10ms on RTX 3060

### Loss Functions

1. **Pixel Loss (L1)**: `λ_pix = 1.0`
2. **Perceptual Loss (VGG)**: `λ_perc = 0.1`
3. **SSIM Loss**: `λ_ssim = 1.0`
4. **Gradient Loss**: `λ_grad = 0.5`
5. **Feature Distillation**: `λ_feat = 0.5` (optional)

---

## 🔧 Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (for GPU training)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/realtime-fusion.git
cd realtime-fusion

# Create virtual environment
conda create -n fusion python=3.8
conda activate fusion

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
torch>=1.10.0
torchvision>=0.11.0
opencv-python>=4.5.0
numpy>=1.21.0
pillow>=8.3.0
scikit-image>=0.18.0
albumentations>=1.1.0
lpips>=0.1.4
tensorboard>=2.7.0
tqdm>=4.62.0
matplotlib>=3.4.0
thop>=0.1.1
onnx>=1.10.0
onnxruntime>=1.9.0
```

---

## 📊 Data Preparation

### Directory Structure

Organize your data as follows:

```
data/
├── train/
│   ├── IR/
│   │   ├── image_001.png
│   │   ├── image_002.png
│   │   └── ...
│   ├── VIS/
│   │   ├── image_001.png
│   │   ├── image_002.png
│   │   └── ...
│   └── TEACHER_FUSED/
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
├── val/
│   ├── IR/
│   ├── VIS/
│   └── TEACHER_FUSED/
└── test/
    ├── IR/
    ├── VIS/
    └── TEACHER_FUSED/
```

### Generate Teacher Fused Images

**Option 1: Using Pre-trained Diffusion Model (DDFM/Mask-DiFuser)**

```python
# Example using your Teacher model
from teacher_model import DDFMFusion

teacher = DDFMFusion.from_pretrained('path/to/checkpoint')
teacher.eval()

for ir_path, vis_path in image_pairs:
    ir = load_image(ir_path)
    vis = load_image(vis_path)
    
    fused = teacher.fuse(ir, vis, num_steps=50)  # Multi-step diffusion
    save_image(fused, f'TEACHER_FUSED/{name}.png')
```

**Option 2: Using Existing Fusion Methods**

If you don't have a diffusion teacher yet, you can start with traditional methods:

```bash
# Generate using DenseFuse/DIDFuse as pseudo-teacher
python scripts/generate_pseudo_teacher.py \
    --method densefuse \
    --ir_dir data/raw/IR \
    --vis_dir data/raw/VIS \
    --output_dir data/train/TEACHER_FUSED
```

### Data Augmentation

The dataset automatically applies:
- ✅ Random crop & resize
- ✅ Horizontal flip
- ✅ Color jittering (VIS only)
- ✅ Gaussian noise (IR sensor simulation)
- ✅ **Misalignment augmentation** (random affine transforms)

---

## 🚀 Training

### Basic Training

```bash
python train.py \
    --train_root data/train \
    --val_root data/val \
    --batch_size 8 \
    --epochs 120 \
    --lr 1e-4 \
    --input_size 512 \
    --use_amp \
    --checkpoint_dir checkpoints \
    --log_dir logs
```

### Key Arguments

```bash
# Data
--train_root          # Path to training data
--val_root            # Path to validation data
--input_size 512      # Input image size (512 or 640)

# Model
--in_channels 3       # Input channels (3 for RGB, 1 for grayscale)
--out_channels 3      # Output channels

# Training
--batch_size 8        # Batch size (adjust based on GPU memory)
--epochs 120          # Number of epochs
--lr 1e-4             # Learning rate
--weight_decay 1e-4   # Weight decay
--use_amp             # Enable mixed precision training

# Loss weights
--lambda_pix 1.0      # Pixel loss weight
--lambda_perc 0.1     # Perceptual loss weight
--lambda_ssim 1.0     # SSIM loss weight
--lambda_grad 0.5     # Gradient loss weight
--lambda_feat 0.5     # Feature distillation weight

# Augmentation
--misalign_prob 0.5   # Probability of misalignment augmentation

# Logging
--log_freq 100        # Log every N iterations
--vis_freq 500        # Visualize every N iterations
--save_freq 10        # Save checkpoint every N epochs
```

### Resume Training

```bash
python train.py \
    --train_root data/train \
    --val_root data/val \
    --resume checkpoints/checkpoint_epoch_50.pth \
    [other arguments...]
```

### Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir logs --port 6006

# Open browser
http://localhost:6006
```

**Training Tips:**

1. **Start with L1 warmup** (10-20 epochs) for stable initialization
2. **Gradually add losses**: L1 → L1+SSIM → L1+SSIM+Perceptual
3. **Use learning rate finder** to find optimal LR:
   ```python
   from utils import LRFinder
   lr_finder = LRFinder(model, optimizer, criterion, device)
   lr_finder.range_test(train_loader)
   lr_finder.plot(save_path='lr_finder.png')
   ```
4. **Monitor validation SSIM/LPIPS** as primary metrics

---

## 📈 Evaluation

### Full Evaluation

```bash
python eval.py \
    --checkpoint checkpoints/best_model.pth \
    --data_root data/test \
    --save_dir eval_results \
    --batch_size 1 \
    --benchmark_speed
```

### Metrics Computed

**Reference-Based (with Teacher GT):**
- SSIM (Structural Similarity)
- PSNR (Peak Signal-to-Noise Ratio)
- LPIPS (Learned Perceptual Image Patch Similarity)

**No-Reference (on Fused Output):**
- EN (Entropy - information content)
- SD (Standard Deviation - contrast)
- SF (Spatial Frequency - sharpness)
- AG (Average Gradient - edge strength)

### Speed Benchmark

```bash
python eval.py \
    --checkpoint checkpoints/best_model.pth \
    --data_root data/test \
    --benchmark_speed \
    --input_size 512
```

Output example:
```
Inference Time Statistics:
  Mean: 18.45 ± 2.31 ms
  Min: 15.20 ms
  Max: 25.60 ms
  Median: 17.80 ms
  FPS: 54.20
```

### Model Analysis

```python
from utils import analyze_model

model = create_student_model()
analyze_model(model, input_size=(1, 3, 512, 512), device='cuda')

# Output:
# Total parameters: 3.45M
# FLOPs: 18.2G
# Inference time: 18.45ms
# Memory: 156.3 MB
```

---

## 🎬 Deployment

### 1. Export to ONNX

```python
from utils import export_to_onnx

model = create_student_model()
load_checkpoint('checkpoints/best_model.pth', model)

export_to_onnx(
    model,
    output_path='model.onnx',
    input_size=(1, 3, 512, 512),
    opset_version=11
)
```

### 2. Optimize with TensorRT

```bash
# Convert ONNX to TensorRT engine
trtexec --onnx=model.onnx \
        --saveEngine=model.trt \
        --fp16 \
        --workspace=4096 \
        --minShapes=ir:1x3x512x512,vis:1x3x512x512 \
        --optShapes=ir:1x3x512x512,vis:1x3x512x512 \
        --maxShapes=ir:1x3x512x512,vis:1x3x512x512
```

### 3. Quantization (INT8)

```python
from torch.quantization import quantize_dynamic

# Post-Training Quantization
quantized_model = quantize_dynamic(
    model, 
    {nn.Conv2d, nn.Linear}, 
    dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'model_int8.pth')
```

### 4. Real-Time Inference Pipeline

```python
import cv2
import numpy as np
import torch

class RealtimeFusion:
    def __init__(self, model_path, device='cuda'):
        self.model = create_student_model().to(device)
        load_checkpoint(model_path, self.model)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def fuse_frame(self, ir_frame, vis_frame):
        """
        Fuse a single frame pair
        
        Args:
            ir_frame: numpy array [H, W, 3]
            vis_frame: numpy array [H, W, 3]
        
        Returns:
            fused_frame: numpy array [H, W, 3]
        """
        # Preprocess
        ir = torch.from_numpy(ir_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        vis = torch.from_numpy(vis_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        ir = ir.to(self.device)
        vis = vis.to(self.device)
        
        # Forward
        fused = self.model(ir, vis)
        
        # Postprocess
        fused = fused[0].cpu().numpy().transpose(1, 2, 0)
        fused = (fused * 255).clip(0, 255).astype(np.uint8)
        
        return fused

# Usage
fusion = RealtimeFusion('checkpoints/best_model.pth')

# Video fusion
ir_cap = cv2.VideoCapture('ir_video.mp4')
vis_cap = cv2.VideoCapture('vis_video.mp4')

while True:
    ret_ir, ir_frame = ir_cap.read()
    ret_vis, vis_frame = vis_cap.read()
    
    if not (ret_ir and ret_vis):
        break
    
    fused = fusion.fuse_frame(ir_frame, vis_frame)
    
    cv2.imshow('Fused', fused)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## 🙏 Acknowledgments

- Teacher models: [DDFM](link), [Mask-DiFuser](link)
- Backbone: MobileNetV3 design principles
- Metrics: LPIPS, pytorch-ssim

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
--batch_size 4

# Or reduce input size
--input_size 384
```

**2. Slow Training**
```bash
# Enable AMP
--use_amp

# Reduce number of workers if CPU bottleneck
--num_workers 2
```

**3. Poor Quality**
```bash
# Check Teacher quality first
# Increase perceptual loss weight
--lambda_perc 0.2

# Add more training data
```

**4. NaN Loss**
```bash
# Reduce learning rate
--lr 5e-5

# Enable gradient clipping (already enabled by default)
--grad_clip 1.0
```

---

## 🚧 Future Work

- [ ] Support for video temporal consistency
- [ ] Multi-scale pyramid fusion
- [ ] Attention-based registration module
- [ ] iOS/Android mobile deployment
- [ ] TensorRT INT8 quantization guide
- [ ] Pre-trained model zoo
