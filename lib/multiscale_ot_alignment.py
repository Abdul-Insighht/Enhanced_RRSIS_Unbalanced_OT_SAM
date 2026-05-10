"""
Multi-Scale Unbalanced OT Feature Alignment for Enhanced_RRSIS_UOT.

Extends the single-scale UOT alignment to operate across multiple FPN
(Feature Pyramid Network) levels with scale-aware projections.

Key idea:
    - Different FPN levels capture objects at different scales
    - Small objects benefit from fine-grained (high-res) alignment
    - Large objects benefit from coarse (low-res) alignment
    - Each scale gets its own UOT alignment with scale-specific projections
    - A learnable scale weighting combines the aligned features

Unbalanced OT Enhancement:
    - Alpha (tau_img): Image marginal relaxation — lets background mass vanish.
    - Beta (tau_txt): Text marginal relaxation — lets irrelevant text vanish.
    - Gamma (reg): Entropic regularization (FIXED for stability).
    - Alpha/Beta start fixed for warmup, then become learnable with softplus.

Reference:
    De Plaen et al., "Unbalanced Optimal Transport: A Unified Framework
    for Object Detection", CVPR 2023.
    Chizat et al., "Scaling Algorithms for Unbalanced Optimal Transport
    Problems", Mathematics of Computation, 2018.
    Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleAwareOTAligner(nn.Module):
    """
    UOT-based feature aligner for a single FPN scale with scale-specific
    projection heads and unbalanced marginal relaxation.

    Args:
        d_model: Hidden dimension.
        reg: Sinkhorn entropy regularization (gamma — FIXED).
        alpha: Image marginal relaxation for UOT.
        beta: Text marginal relaxation for UOT.
        num_iter: Number of Sinkhorn iterations.
        residual_weight: Scale factor for the OT-aligned text residual.
        scale_id: Index of this scale (for logging/identification).
        learnable_margins: If True, alpha/beta become trainable after warmup.
    """

    def __init__(self, d_model, reg=0.1, alpha=1.0, beta=1.0,
                 num_iter=10, residual_weight=0.5, scale_id=0,
                 learnable_margins=True):
        super().__init__()
        self.d_model = d_model
        self.reg = reg              # gamma — FIXED
        self.num_iter = num_iter
        self.residual_weight = residual_weight
        self.scale_id = scale_id
        self.learnable_margins = learnable_margins

        # ====== UOT Marginal Relaxation Parameters ======
        if learnable_margins:
            raw_alpha = self._inverse_softplus(alpha)
            raw_beta = self._inverse_softplus(beta)
            self.raw_alpha = nn.Parameter(torch.tensor(raw_alpha))
            self.raw_beta = nn.Parameter(torch.tensor(raw_beta))
            # Start frozen; training loop will unfreeze after warmup
            self.raw_alpha.requires_grad = False
            self.raw_beta.requires_grad = False
        else:
            self.register_buffer('raw_alpha', torch.tensor(self._inverse_softplus(alpha)))
            self.register_buffer('raw_beta', torch.tensor(self._inverse_softplus(beta)))

        # Scale-specific projection heads
        self.text_proj = nn.Linear(d_model, d_model)
        self.img_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Scale-specific gating: learns how much OT-aligned text to add
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # Initialize output projection near zero
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _inverse_softplus(x, threshold=20.0):
        """Compute inverse of softplus: log(exp(x) - 1)."""
        if x > threshold:
            return x
        return float(torch.log(torch.exp(torch.tensor(x)) - 1.0))

    @property
    def alpha(self):
        """Image marginal relaxation (always positive via softplus)."""
        return F.softplus(self.raw_alpha)

    @property
    def beta(self):
        """Text marginal relaxation (always positive via softplus)."""
        return F.softplus(self.raw_beta)

    @torch.no_grad()
    def unbalanced_sinkhorn(self, cost_matrix):
        """
        Unbalanced Sinkhorn algorithm for UOT.

        The key difference from balanced Sinkhorn is the exponent:
            fi_u = alpha / (alpha + reg)
            fi_v = beta / (beta + reg)
        When alpha, beta → ∞, fi → 1.0 → reduces to balanced OT.
        When alpha, beta are finite, mass can be created/destroyed.

        Args:
            cost_matrix: (B, N_img, N_txt) pairwise cost.

        Returns:
            Transport plan P of shape (B, N_img, N_txt).
        """
        B, N, M = cost_matrix.shape

        # Get current alpha/beta (positive via softplus)
        alpha_val = self.alpha
        beta_val = self.beta

        # UOT exponents
        fi_u = alpha_val / (alpha_val + self.reg)
        fi_v = beta_val / (beta_val + self.reg)

        # Uniform marginals
        mu = torch.full((B, N), 1.0 / N, device=cost_matrix.device,
                        dtype=cost_matrix.dtype)
        nu = torch.full((B, M), 1.0 / M, device=cost_matrix.device,
                        dtype=cost_matrix.dtype)

        # Gibbs kernel
        K = torch.exp(-cost_matrix / self.reg)
        K = K.clamp(min=1e-8)  # Numerical stability

        u = torch.ones_like(mu)
        for _ in range(self.num_iter):
            # UNBALANCED update: raise to power fi_v and fi_u
            v = (nu / (torch.bmm(K.transpose(1, 2),
                                  u.unsqueeze(2)).squeeze(2) + 1e-8)) ** fi_v
            u = (mu / (torch.bmm(K,
                                  v.unsqueeze(2)).squeeze(2) + 1e-8)) ** fi_u

        # Transport plan
        P = u.unsqueeze(2) * K * v.unsqueeze(1)
        return P

    def forward(self, img_feat, text_feat, text_mask=None):
        """
        Scale-specific UOT alignment.

        Args:
            img_feat: (B, C, H, W) image features.
            text_feat: (seq, B, C) or (B, seq, C) text features.
            text_mask: (B, seq) boolean mask, True = padding.

        Returns:
            aligned_img: (B, C, H, W) image features enhanced with
                         UOT-aligned text at this scale.
        """
        B, C, H, W = img_feat.shape

        # Reshape image to (B, H*W, C)
        img_flat = img_feat.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # Text: handle both (seq, B, C) and (B, seq, C)
        if text_feat.dim() == 3 and text_feat.shape[1] == B and text_feat.shape[0] != B:
            text_flat = text_feat.permute(1, 0, 2)  # (B, seq, C)
        elif text_feat.dim() == 3:
            text_flat = text_feat  # Already (B, seq, C)
        else:
            return img_feat  # Can't align without proper text shape

        # Mask out padding text tokens
        if text_mask is not None:
            valid_mask = ~text_mask
            text_flat = text_flat * valid_mask.unsqueeze(-1).float()

        # Project to alignment space
        img_proj = self.img_proj(img_flat)
        txt_proj = self.text_proj(text_flat)

        # Cost matrix via negative cosine similarity
        img_norm = F.normalize(img_proj, dim=-1)
        txt_norm = F.normalize(txt_proj, dim=-1)
        cost = 1.0 - torch.bmm(img_norm, txt_norm.transpose(1, 2))  # (B, HW, seq)

        # Compute UOT plan
        P = self.unbalanced_sinkhorn(cost)  # (B, HW, seq)

        # Transport text to image positions
        aligned_text = torch.bmm(P * P.shape[1], text_flat)  # (B, HW, C)
        aligned_text = self.output_proj(aligned_text)

        # Gated residual: learn how much aligned text to mix in
        gate_input = torch.cat([img_flat, aligned_text], dim=-1)  # (B, HW, 2C)
        gate_weight = self.gate(gate_input)  # (B, HW, C) — per-position gate

        img_enhanced = img_flat + gate_weight * aligned_text
        img_enhanced = self.norm(img_enhanced)

        # Reshape back
        img_enhanced = img_enhanced.permute(0, 2, 1).view(B, C, H, W)
        return img_enhanced


class MultiScaleOTAligner(nn.Module):
    """
    Multi-Scale Unbalanced OT Feature Alignment across all FPN levels.

    Creates a scale-specific UOT aligner for each FPN level and applies
    them in parallel, then combines via text-conditioned scale weighting.

    UOT Parameters:
        - Alpha/Beta: Per-scale learnable marginal relaxation (with warmup).
        - Gamma (reg): Fixed entropic regularization.

    Args:
        d_model: Hidden dimension.
        num_scales: Number of FPN levels to align (default 3).
        reg: Sinkhorn entropy regularization (gamma — FIXED).
        alpha: Initial image marginal relaxation.
        beta: Initial text marginal relaxation.
        num_iter: Number of Sinkhorn iterations.
        residual_weight: Residual weight for each scale.
        text_conditioned: Use text-conditioned scale selection.
        learnable_margins: If True, alpha/beta become trainable after warmup.
        warmup_epochs: Epochs to keep alpha/beta fixed before unfreezing.
    """

    def __init__(self, d_model, num_scales=3, reg=0.1, alpha=1.0, beta=1.0,
                 num_iter=10, residual_weight=0.5, text_conditioned=True,
                 learnable_margins=True, warmup_epochs=5):
        super().__init__()
        self.num_scales = num_scales
        self.text_conditioned = text_conditioned
        self.learnable_margins = learnable_margins
        self.warmup_epochs = warmup_epochs

        # Create a scale-specific UOT aligner for each FPN level
        self.scale_aligners = nn.ModuleList([
            ScaleAwareOTAligner(
                d_model=d_model,
                reg=reg,
                alpha=alpha,
                beta=beta,
                num_iter=num_iter,
                residual_weight=residual_weight,
                scale_id=i,
                learnable_margins=learnable_margins,
            )
            for i in range(num_scales)
        ])

        if text_conditioned:
            # Text-conditioned scale selector
            # Pools text features → predicts per-sample scale weights
            self.scale_selector = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, num_scales),
            )
            print(f"[MultiScaleUOT] Text-conditioned {num_scales}-scale "
                  f"Unbalanced OT alignment (d_model={d_model}, "
                  f"alpha={alpha}, beta={beta}, reg={reg}, "
                  f"learnable={learnable_margins})")
        else:
            # Fallback: learnable but static scale weights
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
            print(f"[MultiScaleUOT] Static {num_scales}-scale "
                  f"Unbalanced OT alignment (d_model={d_model})")

    def set_epoch(self, epoch):
        """
        Called by training loop each epoch.
        Unfreezes alpha/beta in all scale aligners after warmup.
        """
        if self.learnable_margins and epoch >= self.warmup_epochs:
            for aligner in self.scale_aligners:
                if hasattr(aligner, 'raw_alpha') and not aligner.raw_alpha.requires_grad:
                    aligner.raw_alpha.requires_grad = True
                    aligner.raw_beta.requires_grad = True
            if epoch == self.warmup_epochs:
                print(f"[MultiScaleUOT] Epoch {epoch}: Alpha/Beta unfrozen "
                      f"across all {self.num_scales} scales")

    def get_uot_params_info(self):
        """Return current alpha/beta values for logging."""
        info = {}
        for i, aligner in enumerate(self.scale_aligners):
            info[f'scale_{i}_alpha'] = aligner.alpha.item()
            info[f'scale_{i}_beta'] = aligner.beta.item()
        return info

    def forward(self, fpn_features, text_feat, text_mask=None):
        """
        Apply scale-aware UOT alignment to each FPN level.

        Args:
            fpn_features: List of FPN feature maps, each (B, C, H_i, W_i).
            text_feat: (seq, B, C) text features.
            text_mask: (B, seq) boolean mask, True = padding.

        Returns:
            aligned_fpn: List of aligned FPN feature maps.
        """
        aligned_fpn = []

        # Compute scale weights
        if self.text_conditioned and hasattr(self, 'scale_selector'):
            B = fpn_features[0].tensors.shape[0] if hasattr(fpn_features[0], 'tensors') else fpn_features[0].shape[0]
            # Pool text features → (B, C)
            if text_feat.dim() == 3:
                if text_feat.shape[1] == B and text_feat.shape[0] != B:
                    # (seq, B, C) format — pool over sequence dim
                    text_pooled = text_feat.mean(dim=0)   # (B, C)
                elif text_feat.shape[0] == B:
                    # (B, seq, C) format — pool over sequence dim
                    text_pooled = text_feat.mean(dim=1)   # (B, C)
                else:
                    # Ambiguous — assume (seq, B, C) and permute
                    text_pooled = text_feat.permute(1, 0, 2).mean(dim=1)  # (B, C)
            else:
                text_pooled = text_feat
            # Per-sample scale weights
            weights = F.softmax(self.scale_selector(text_pooled), dim=-1)  # (B, num_scales)
        else:
            weights = F.softmax(self.scale_weights, dim=0)  # (num_scales,)

        for i, feat in enumerate(fpn_features):
            # Handle both plain tensors and NestedTensor
            if hasattr(feat, 'tensors'):
                feat_tensor = feat.tensors
            else:
                feat_tensor = feat

            if feat_tensor.dim() != 4:
                aligned_fpn.append(feat)
                continue

            # Use the aligner for this scale (or last one if we have more FPN levels)
            aligner_idx = min(i, len(self.scale_aligners) - 1)
            aligner = self.scale_aligners[aligner_idx]

            # Apply scale-specific UOT alignment
            aligned = aligner(feat_tensor, text_feat, text_mask)

            # Weighted blend: original + weight * (aligned - original)
            if self.text_conditioned and hasattr(self, 'scale_selector'):
                B = feat_tensor.shape[0]
                w = weights[:, aligner_idx].view(B, 1, 1, 1)  # Per-sample weight
            else:
                w = weights[aligner_idx]

            blended = (1 - w) * feat_tensor + w * aligned

            if hasattr(feat, 'tensors'):
                feat.tensors = blended
                aligned_fpn.append(feat)
            else:
                aligned_fpn.append(blended)

        return aligned_fpn
