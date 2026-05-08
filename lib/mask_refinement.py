"""
Mask Refinement Head for Enhanced_RRSIS_UOT.

Replaces the single bilinear upsample with a progressive refinement
pipeline that uses FPN features as high-resolution guidance.
Integrates OrientationAwareConv for rotation-invariant boundary refinement.

The key insight: SAM3 predicts masks at ~144x144 then upsamples to 504x504
via bilinear interpolation, which blurs boundaries. This module refines
the coarse mask using high-res FPN features before the final upsample,
producing sharper boundaries that push Pr@0.8 and Pr@0.9 higher.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .orientation_conv import OrientationAwareConv


class MaskRefinementHead(nn.Module):
    """
    Progressive mask refinement using FPN skip connections + RDConv.

    Architecture:
        coarse_mask (B,1,H,W) + FPN_feat (B,C,H,W)
            → Concat → RDConv → Conv → refined_mask (B,1,H,W)
            → Bilinear upsample to target size
            → Conv3x3 → Conv1x1 → final_mask

    Args:
        d_model: FPN feature dimension (default 256).
        use_rdconv: Use orientation-aware conv (True) or standard conv.
        num_orientations: Number of RDConv orientation branches.
    """

    def __init__(self, d_model=256, use_rdconv=True, num_orientations=8):
        super().__init__()
        self.d_model = d_model
        self.use_rdconv = use_rdconv

        # Stage 1: Fuse coarse mask with FPN features at FPN resolution
        self.fuse_conv = nn.Conv2d(d_model + 1, d_model, 3, padding=1, bias=False)
        self.fuse_norm = nn.GroupNorm(8, d_model)

        # Stage 2: Orientation-aware refinement
        if use_rdconv:
            self.refine_block = OrientationAwareConv(
                channels=d_model,
                num_orientations=num_orientations,
            )
        else:
            self.refine_block = nn.Sequential(
                nn.Conv2d(d_model, d_model, 3, padding=1, bias=False),
                nn.GroupNorm(8, d_model),
                nn.GELU(),
            )

        # Stage 3: Additional refinement conv
        self.refine_conv2 = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, 3, padding=1, bias=False),
            nn.GroupNorm(4, d_model // 2),
            nn.GELU(),
        )

        # Stage 4: Final mask prediction at high resolution
        self.final_refine = nn.Sequential(
            nn.Conv2d(d_model // 2, d_model // 4, 3, padding=1, bias=False),
            nn.GroupNorm(4, d_model // 4),
            nn.GELU(),
            nn.Conv2d(d_model // 4, 1, 1),
        )

        # Gate: learn how much refinement to blend with original
        self.gate = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights so refinement starts near identity."""
        for m in [self.final_refine[-1]]:
            if hasattr(m, 'weight'):
                nn.init.zeros_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, coarse_mask, fpn_feat, target_size):
        """
        Refine coarse mask using FPN features.

        Args:
            coarse_mask: (B, 1, H_mask, W_mask) coarse mask logits from SAM3.
            fpn_feat: (B, C, H_fpn, W_fpn) highest-resolution FPN features.
            target_size: (H, W) tuple for final output size (e.g., 504x504).

        Returns:
            refined_mask: (B, 1, H_target, W_target) refined mask logits.
        """
        # Upsample coarse mask to match FPN spatial dimensions
        fpn_h, fpn_w = fpn_feat.shape[-2:]
        mask_at_fpn = F.interpolate(
            coarse_mask.float(), size=(fpn_h, fpn_w),
            mode='bilinear', align_corners=False
        )

        # Stage 1: Fuse mask channel with FPN features
        fused = torch.cat([fpn_feat, mask_at_fpn], dim=1)  # (B, C+1, H, W)
        fused = F.gelu(self.fuse_norm(self.fuse_conv(fused)))

        # Stage 2: Orientation-aware refinement
        refined = self.refine_block(fused)

        # Stage 3: Further refine
        refined = self.refine_conv2(refined)

        # Upsample to target size
        refined = F.interpolate(
            refined, size=target_size,
            mode='bilinear', align_corners=False
        )
        mask_at_target = F.interpolate(
            mask_at_fpn, size=target_size,
            mode='bilinear', align_corners=False
        )

        # Stage 4: Predict refined mask at target resolution
        refined_mask = self.final_refine(refined)

        # Gated residual: blend refined with original upsampled mask
        gate_input = torch.cat([mask_at_target, refined_mask], dim=1)
        gate_weight = self.gate(gate_input)
        output = gate_weight * refined_mask + (1 - gate_weight) * mask_at_target

        return output
