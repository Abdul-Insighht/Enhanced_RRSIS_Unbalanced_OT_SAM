# Enhanced_RRSIS_UOT_SAM

**Enhanced Referring Remote Sensing Image Segmentation with Unbalanced Optimal Transport (UOT)**

This repository provides a state-of-the-art solution for Referring Remote Sensing Image Segmentation (RRSIS). It extends the baseline SAM3 architecture with **7 novel enhancements** specifically designed to tackle remote sensing challenges such as extreme foreground-background imbalance, small object detection, arbitrary orientations, and blurry boundaries.

---

## ✨ Key Features & Enhancements

| Enhancement | Module | Description |
|-------------|--------|-------------|
| 🟢 **Text-Guided Dynamic LoRA** | `lib/dynamic_lora.py` | Vision encoder adapts weights dynamically per-caption using a HyperNetwork. |
| 🟢 **Multi-Scale UOT Alignment** | `lib/multiscale_ot_alignment.py` | Unbalanced Optimal Transport (Sinkhorn) across 3 FPN levels. Suppresses 95%+ background via learnable Alpha/Beta relaxation margins. |
| 🟢 **Mask Refinement Head** | `lib/mask_refinement.py` | Progressive mask refinement using high-res FPN skip connections. |
| 🟢 **Orientation-Aware Conv (RDConv)** | `lib/orientation_conv.py` | 8-branch depthwise convolution to handle arbitrary object rotations in satellite imagery. |
| 🟢 **OHEM + Focal + Boundary Loss** | `lib/ohem_loss.py` | Online Hard Example Mining (top 30%), Focal weighting, and Sobel edge boundary supervision. |
| 🟢 **Lovász Hinge Loss** | `lib/ohem_loss.py` | Direct optimization of the Jaccard index (IoU) surrogate. |
| 🟢 **Contrastive Loss (InfoNCE)** | `lib/contrastive_loss.py` | Auxiliary loss aligning masked visual features tightly with text features. |

---

## 💻 Environment Setup (Local GPU System)

This project requires a dedicated GPU system with NVIDIA CUDA support (Linux or Windows). 

**1. Create a Conda Environment:**
```bash
conda create -n rrsis_uot python=3.10 -y
conda activate rrsis_uot
```

**2. Install PyTorch:**
Install PyTorch compatible with your CUDA version (e.g., CUDA 12.1).
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
pip install opencv-python pillow numpy wandb scipy
```

**4. Download Pre-trained SAM3 Weights:**
Create a directory for weights and place the SAM3 checkpoint there.
```bash
mkdir -p pre-trained-weights
# Ensure your SAM3 weights (e.g., sam3.pt) are inside this folder
```

---

## 📂 Dataset Preparation

The model supports 3 major benchmarks. Structure your `data/` folder as follows:

```
data/
├── rrsis_d/
│   ├── images/              # 800x800 Remote Sensing Images
│   └── annotations/         # REFER API format JSONs
├── rrsis_hr/
│   ├── images/              # 1024x1024 High-Res Images
│   └── annotations/
└── RefSegRS/
    ├── images/              # 512x512 Images
    ├── masks/               # Ground Truth binary masks
    ├── output_phrase_train.txt
    ├── output_phrase_val.txt
    └── output_phrase_test.txt
```

---

## 🚀 Training

### Full Enhanced Model (Recommended)
Train the model with all 7 techniques enabled, including the Unbalanced OT with a 5-epoch warmup.

```bash
python train.py \
    --dataset rrsis_d \
    --data_root ./data \
    --output_dir ./output/rrsis_d_enhanced \
    --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --batch_size 2 \
    --grad_accum_steps 4 \
    --epochs 40 \
    --use_dynamic_lora \
    --use_multiscale_ot \
    --learnable_margins \
    --uot_warmup_epochs 5
```

### Unbalanced OT (UOT) Hyperparameter Control
The repository uses Unbalanced Sinkhorn to ignore irrelevant background and padding.
- `--ot_alpha`: Image margin relaxation (Default: 1.0)
- `--ot_beta`: Text margin relaxation (Default: 1.0)
- `--learnable_margins`: Unfreezes Alpha/Beta after the warmup epochs.

To force strict Background Suppression for tiny objects:
```bash
python train.py --dataset rrsis_d --ot_alpha 0.5 --ot_beta 1.0 --no_learnable_margins
```

---

## 📊 Evaluation

Evaluate the model on the test split. The evaluation script automatically applies Test-Time Augmentation (TTA) using horizontal and vertical flips.

```bash
python test.py \
    --dataset rrsis_d \
    --data_root ./data \
    --split test \
    --resume ./output/rrsis_d_enhanced/best_model.pth \
    --visualize
```

**Output Metrics:**
- **mIoU**: Mean Intersection over Union
- **oIoU**: Overall Intersection over Union
- **P@0.5 - P@0.9**: Precision at IoU thresholds. Our Mask Refinement + Lovász Loss specifically boosts P@0.8 and P@0.9.
- Visualizations will be saved in `output_dir/visualizations/`.

---

## 🧠 Architecture Flow

1. **Input:** Image (504×504) + Text Caption
2. **Backbone:** SAM3 ViT + Text Encoder (Frozen)
3. **Adaptation:** Text-Guided Dynamic LoRA
4. **Alignment:** Multi-Scale Unbalanced OT (across 3 FPN levels)
5. **Fusion:** SAM3 Transformer Encoder
6. **Detection:** SAM3 DETR Decoder
7. **Refinement:** Mask Refinement Head (RDConv + FPN skip connections)
8. **Losses:** EnhancedOHEM (Boundary + Lovász + Focal) + Contrastive Loss

---
*Developed for Advanced Referring Remote Sensing Image Segmentation (2026).*
