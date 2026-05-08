import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Enhanced RRSIS-UOT: Enhanced Referring Remote Sensing Image Segmentation with Unbalanced Optimal Transport')

    # ====== Paths ======
    parser.add_argument('--data_root', type=str, default='./data/',
                        help='Root directory of datasets')
    parser.add_argument('--output_dir', type=str, default='./output/',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--sam3_ckpt', type=str, default='./pre-trained-weights/sam3.pt',
                        help='Path to SAM3 pretrained checkpoint')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to resume training from checkpoint')

    # ====== Dataset ======
    parser.add_argument('--dataset', type=str, default='refcoco',
                        choices=['refsegrs', 'rrsis_d', 'rrsis_hr'],
                        help='Dataset name')
    parser.add_argument('--split', type=str, default='train',
                        help='Data split (train/val/test)')
    parser.add_argument('--splitBy', type=str, default='unc',
                        help='Split method')
    parser.add_argument('--max_tokens', type=int, default=32,
                        help='Maximum text token length for SAM3 tokenizer')

    # ====== Model ======
    parser.add_argument('--image_size', type=int, default=504,
                        help='Input image size (divisible by 14 for SAM3 ViT)')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA rank for vision backbone adaptation')
    parser.add_argument('--lora_alpha', type=float, default=32.0,
                        help='LoRA alpha scaling factor')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='Freeze SAM3 ViT backbone (train only LoRA + decoder)')
    parser.add_argument('--freeze_text_encoder', action='store_true', default=True,
                        help='Freeze SAM3 text encoder')

    # ====== Enhancement Flags (NEW) ======
    parser.add_argument('--use_dynamic_lora', action='store_true', default=True,
                        help='Enable text-guided dynamic LoRA adapters')
    parser.add_argument('--no_dynamic_lora', action='store_true', default=False,
                        help='Disable dynamic LoRA (use static LoRA instead)')
    parser.add_argument('--use_contrastive_loss', action='store_true', default=True,
                        help='Enable contrastive language-image loss (InfoNCE)')
    parser.add_argument('--no_contrastive_loss', action='store_true', default=False,
                        help='Disable contrastive loss')
    parser.add_argument('--use_multiscale_ot', action='store_true', default=True,
                        help='Enable multi-scale OT feature alignment')
    parser.add_argument('--no_multiscale_ot', action='store_true', default=False,
                        help='Disable multi-scale OT (use single-scale)')
    parser.add_argument('--use_ohem_loss', action='store_true', default=True,
                        help='Enable OHEM + Focal + Boundary loss')
    parser.add_argument('--no_ohem_loss', action='store_true', default=False,
                        help='Disable OHEM loss (use standard Dice+BCE)')

    # ====== Enhancement Parameters (NEW) ======
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                        help='Weight for contrastive loss (InfoNCE)')
    parser.add_argument('--ohem_hard_ratio', type=float, default=0.3,
                        help='OHEM: fraction of hardest pixels to keep')
    parser.add_argument('--ot_reg', type=float, default=0.1,
                        help='Sinkhorn OT entropy regularization (gamma — FIXED)')
    parser.add_argument('--ot_alpha', type=float, default=1.0,
                        help='UOT image marginal relaxation (lower = more bg suppression)')
    parser.add_argument('--ot_beta', type=float, default=1.0,
                        help='UOT text marginal relaxation (lower = more text filtering)')
    parser.add_argument('--learnable_margins', action='store_true', default=True,
                        help='Make UOT alpha/beta trainable after warmup')
    parser.add_argument('--no_learnable_margins', action='store_true', default=False,
                        help='Keep UOT alpha/beta fixed throughout training')
    parser.add_argument('--uot_warmup_epochs', type=int, default=5,
                        help='Epochs to keep alpha/beta fixed before unfreezing')
    parser.add_argument('--ot_num_iter', type=int, default=10,
                        help='Number of Sinkhorn iterations')
    parser.add_argument('--num_ot_scales', type=int, default=3,
                        help='Number of FPN scales for multi-scale OT')
    parser.add_argument('--boundary_loss_weight', type=float, default=0.5,
                        help='Weight for boundary-aware loss component')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma (higher = more focus on hard)')

    # ====== Mask Refinement (NEW) ======
    parser.add_argument('--use_mask_refinement', action='store_true', default=True,
                        help='Enable mask refinement head for sharper boundaries')
    parser.add_argument('--no_mask_refinement', action='store_true', default=False,
                        help='Disable mask refinement head')

    # ====== Orientation-Aware Conv / RDConv (NEW) ======
    parser.add_argument('--use_rdconv', action='store_true', default=True,
                        help='Enable rotated depthwise conv in mask refinement')
    parser.add_argument('--no_rdconv', action='store_true', default=False,
                        help='Disable RDConv (use standard conv)')
    parser.add_argument('--num_orientations', type=int, default=8,
                        help='Number of RDConv orientation branches')

    # ====== Lovász Loss (NEW) ======
    parser.add_argument('--use_lovasz_loss', action='store_true', default=True,
                        help='Enable Lovász hinge loss for direct IoU optimization')
    parser.add_argument('--no_lovasz_loss', action='store_true', default=False,
                        help='Disable Lovász loss')
    parser.add_argument('--lovasz_weight', type=float, default=1.0,
                        help='Weight for Lovász loss component')

    # ====== Data Augmentation (NEW) ======
    parser.add_argument('--use_augmentation', action='store_true', default=True,
                        help='Enable strong data augmentation (flip/rotate/scale/color)')
    parser.add_argument('--no_augmentation', action='store_true', default=False,
                        help='Disable data augmentation')

    # ====== Test-Time Augmentation (NEW) ======
    parser.add_argument('--use_tta', action='store_true', default=True,
                        help='Enable test-time augmentation (flip averaging)')
    parser.add_argument('--no_tta', action='store_true', default=False,
                        help='Disable TTA')

    # ====== Training ======
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Training batch size per GPU')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Base learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-5,
                        help='Learning rate for backbone (LoRA params)')
    parser.add_argument('--lr_decoder', type=float, default=5e-5,
                        help='Learning rate for decoder/seg head')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                        help='Gradient accumulation steps (effective batch = batch_size * accum)')

    # ====== Loss ======
    parser.add_argument('--dice_weight', type=float, default=0.5,
                        help='Weight for Dice loss')
    parser.add_argument('--ce_weight', type=float, default=0.5,
                        help='Weight for Cross-Entropy loss')
    parser.add_argument('--boundary_weight', type=float, default=0.2,
                        help='Weight for boundary loss')

    # ====== Optimization ======
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use mixed precision (fp16/bf16)')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                        help='Enable gradient checkpointing for memory savings')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Dataloader workers')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='Print frequency')

    # ====== Distributed ======
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of GPUs')

    # ====== Evaluation ======
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='Run evaluation only')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Save visualization of predictions')

    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()

    # Handle --no_* flags to override defaults
    if args.no_dynamic_lora:
        args.use_dynamic_lora = False
    if args.no_contrastive_loss:
        args.use_contrastive_loss = False
    if args.no_multiscale_ot:
        args.use_multiscale_ot = False
    if args.no_ohem_loss:
        args.use_ohem_loss = False
    if args.no_mask_refinement:
        args.use_mask_refinement = False
    if args.no_rdconv:
        args.use_rdconv = False
    if args.no_lovasz_loss:
        args.use_lovasz_loss = False
    if args.no_augmentation:
        args.use_augmentation = False
    if args.no_tta:
        args.use_tta = False
    if args.no_learnable_margins:
        args.learnable_margins = False

    return args
