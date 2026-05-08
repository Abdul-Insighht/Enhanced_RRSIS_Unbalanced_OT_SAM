"""
Orientation-Aware Convolution (Lightweight RDConv) for Enhanced_RRSIS_UOT.

Provides rotation-aware feature extraction by using multiple depthwise
convolutions specialized for different orientations, combined via
input-dependent attention. Much lighter than full ARC while providing
orientation handling needed for remote sensing objects at arbitrary angles.

Reference:
    Inspired by Adaptive Rotated Convolution (Pu et al., ICCV 2023),
    simplified for efficiency on limited GPU budgets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrientationAwareConv(nn.Module):
    """
    Multi-orientation depthwise convolution block.

    Uses K parallel depthwise convolutions that learn to specialize on
    different orientations, weighted by an input-dependent predictor.

    Args:
        channels: Number of input/output channels.
        num_orientations: Number of orientation branches (default 8 = 45° spacing).
        kernel_size: Convolution kernel size.
    """

    def __init__(self, channels, num_orientations=8, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.num_orientations = num_orientations
        padding = kernel_size // 2

        # K depthwise convolution branches (each specializes on an orientation)
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size, padding=padding,
                      groups=channels, bias=False)
            for _ in range(num_orientations)
        ])

        # Orientation predictor: input features → orientation weights
        self.orient_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, num_orientations),
        )

        # Pointwise conv for channel mixing after orientation selection
        self.pw_conv = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)

        # Initialize depthwise convs with rotated versions of a base kernel
        self._init_rotated_kernels()

    def _init_rotated_kernels(self):
        """Initialize each branch's kernel as a rotated version of a base edge detector."""
        import math
        base_kernel = torch.tensor([
            [-1, -1, 0],
            [-1,  0, 1],
            [ 0,  1, 1],
        ], dtype=torch.float32)

        for i, conv in enumerate(self.dw_convs):
            angle_idx = i  # 0 to K-1
            # Use rot90 for 90° multiples, interpolated init for others
            rotations = angle_idx % 4
            with torch.no_grad():
                rotated = torch.rot90(base_kernel, rotations, [0, 1])
                if angle_idx >= 4:
                    # For 45° offsets, use transposed version
                    rotated = rotated.T
                # Apply as initialization pattern (scaled small)
                for c in range(self.channels):
                    conv.weight.data[c, 0] = rotated * 0.01

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input features.
        Returns:
            (B, C, H, W) orientation-aware features.
        """
        B = x.shape[0]

        # Predict orientation weights
        weights = F.softmax(self.orient_pred(x), dim=-1)  # (B, K)

        # Apply all orientation branches
        outputs = torch.stack([conv(x) for conv in self.dw_convs], dim=1)  # (B, K, C, H, W)

        # Weighted combination
        w = weights.view(B, self.num_orientations, 1, 1, 1)
        out = (outputs * w).sum(dim=1)  # (B, C, H, W)

        # Channel mixing + normalization
        out = self.pw_conv(out)
        out = self.norm(out)
        out = F.gelu(out)

        # Residual connection
        return x + out
