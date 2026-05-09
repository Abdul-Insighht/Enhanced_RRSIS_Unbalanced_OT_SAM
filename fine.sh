#!/bin/bash
# ====== Enhanced_RRSIS_UOT Training Script ======
# Resume Fine-Tuning from Best Checkpoint (Epoch 22)
# Optimized for Kaggle T4/P100 (16GB VRAM)
#
# Hyperparameter Changes for Resume:
#   - LR: 5e-5 → 1e-5 (prevents destroying learned weights)
#   - LR Backbone: 1e-5 → 5e-6
#   - LR Decoder: 5e-5 → 2e-5
#   - Grad Accum: 4 → 8 (effective batch 16, smoother gradients)
#   - Warmup: 5 → 2 epochs (already initialized)
#   - Epochs: 40 → 50 (gives 28 more epochs from epoch 22)
#
# Usage:
#   Resume training:  bash fine.sh rrsis_d ./data ./output/rrsis_d_enhanced_uot/best_model.pth
#   Fresh training:   bash fine.sh rrsis_d ./data none
#
# Ablation (disable specific techniques):
#   bash fine.sh rrsis_d ./data ./output/best_model.pth --no_dynamic_lora
#   bash fine.sh rrsis_d ./data ./output/best_model.pth --no_contrastive_loss --no_ohem_loss

DATASET=${1:-rrsis_d}
DATA_ROOT=${2:-./data}
RESUME_CKPT=${3:-none}
OUTPUT_DIR="./output/${DATASET}_enhanced_uot"
EXTRA_ARGS="${@:4}"

echo "============================================="
echo "  Enhanced_RRSIS_UOT Training"
echo "  Dataset: ${DATASET}"
echo "  Data Root: ${DATA_ROOT}"
echo "  Resume: ${RESUME_CKPT}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Extra Args: ${EXTRA_ARGS}"
echo "============================================="

# Build resume argument
RESUME_ARG=""
if [ "${RESUME_CKPT}" != "none" ] && [ -f "${RESUME_CKPT}" ]; then
    RESUME_ARG="--resume ${RESUME_CKPT}"
    echo "  → Resuming from checkpoint: ${RESUME_CKPT}"
else
    echo "  → Training from scratch (no resume)"
fi

python train.py \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --image_size 504 \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --epochs 50 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr 1e-5 \
    --lr_backbone 5e-6 \
    --lr_decoder 2e-5 \
    --weight_decay 0.01 \
    --warmup_epochs 2 \
    --fp16 \
    --gradient_checkpointing \
    --seed 42 \
    --num_workers 4 \
    --contrastive_weight 0.1 \
    --ohem_hard_ratio 0.3 \
    --ot_reg 0.1 \
    --ot_num_iter 10 \
    --num_ot_scales 3 \
    --focal_gamma 2.0 \
    ${RESUME_ARG} \
    ${EXTRA_ARGS}
