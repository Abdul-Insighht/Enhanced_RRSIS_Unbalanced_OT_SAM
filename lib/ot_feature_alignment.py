"""
Unbalanced OT-Based Feature Alignment: Uses Unbalanced Sinkhorn algorithm
to compute optimal transport between text tokens and image spatial tokens
for better cross-modal fusion before the encoder.

Instead of SAM3's default approach of mean-pooling text and adding uniformly
to all image features, this module computes a soft UOT plan so each image
spatial position receives a unique weighted combination of text tokens.

Key difference from Balanced OT:
    - Alpha (tau_img): Controls how much image background mass can be discarded.
      Lower alpha = more aggressive background suppression.
    - Beta (tau_txt): Controls how much text padding/irrelevant token mass
      can be discarded. Lower beta = more aggressive text filtering.
    - Gamma (reg): Entropic regularization for Sinkhorn convergence (FIXED).

For Remote Sensing: Large images with tiny objects (2-5% foreground) strongly
benefit from UOT since background image tokens can be "ignored" rather than
being forced to receive text features.

Reference:
    De Plaen et al., "Unbalanced Optimal Transport: A Unified Framework
    for Object Detection", CVPR 2023.
    Chizat et al., "Scaling Algorithms for Unbalanced Optimal Transport
    Problems", Mathematics of Computation, 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OTFeatureAligner(nn.Module):
    """
    Computes Unbalanced Sinkhorn-based optimal transport between text and
    image tokens to produce spatially-varying text features aligned to
    image features.

    Each image spatial position receives a text representation weighted by
    its UOT coupling, rather than a uniform mean-pool. Background image
    positions can receive near-zero text mass (unbalanced property).

    Args:
        d_model: Hidden dimension of both text and image features.
        reg: Sinkhorn entropy regularization gamma (FIXED, lower = sharper).
        alpha: Image marginal relaxation (KL penalty for source mass).
            Controls how much image background can be ignored.
        beta: Text marginal relaxation (KL penalty for target mass).
            Controls how much irrelevant text tokens can be ignored.
        num_iter: Number of Sinkhorn iterations.
        residual_weight: Scale factor for the OT-aligned text residual.
        learnable_margins: If True, alpha and beta become trainable after
            warmup (use with warmup_epochs to prevent early divergence).
        warmup_epochs: Number of epochs to keep alpha/beta fixed before
            making them trainnable. Only used if learnable_margins=True.
    """

    def __init__(self, d_model, reg=0.1, alpha=1.0, beta=1.0,
                 num_iter=10, residual_weight=0.5,
                 learnable_margins=True, warmup_epochs=5):
        super().__init__()
        self.d_model = d_model
        self.reg = reg              # gamma — FIXED, never trainable
        self.num_iter = num_iter
        self.residual_weight = residual_weight
        self.learnable_margins = learnable_margins
        self.warmup_epochs = warmup_epochs
        self._current_epoch = 0     # Tracked externally by training loop

        # ====== UOT Marginal Relaxation Parameters ======
        # Store as raw (pre-softplus) values so softplus guarantees positivity
        if learnable_margins:
            # Inverse-softplus of initial values for correct initialization
            raw_alpha = self._inverse_softplus(alpha)
            raw_beta = self._inverse_softplus(beta)
            self.raw_alpha = nn.Parameter(torch.tensor(raw_alpha))
            self.raw_beta = nn.Parameter(torch.tensor(raw_beta))
            # Start frozen; unfreeeze after warmup
            self.raw_alpha.requires_grad = False
            self.raw_beta.requires_grad = False
        else:
            self.register_buffer('raw_alpha', torch.tensor(self._inverse_softplus(alpha)))
            self.register_buffer('raw_beta', torch.tensor(self._inverse_softplus(beta)))

        # Project text and image into shared alignment space
        self.text_proj = nn.Linear(d_model, d_model)
        self.img_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Initialize output projection near zero so residual starts small
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

    def set_epoch(self, epoch):
        """
        Called by training loop to update current epoch.
        Unfreezes alpha/beta after warmup period.
        """
        self._current_epoch = epoch
        if self.learnable_margins and epoch >= self.warmup_epochs:
            if not self.raw_alpha.requires_grad:
                self.raw_alpha.requires_grad = True
                self.raw_beta.requires_grad = True
                print(f"[UOT] Epoch {epoch}: Alpha/Beta unfrozen (learnable)")

    @torch.no_grad()
    def unbalanced_sinkhorn(self, cost_matrix):
        """
        Unbalanced Sinkhorn algorithm for UOT.

        The key difference from balanced Sinkhorn is the exponent applied
        to the update steps:
            fi_u = alpha / (alpha + reg)
            fi_v = beta / (beta + reg)

        When alpha, beta → ∞, fi → 1.0 and this reduces to balanced OT.
        When alpha, beta are finite, mass can be created/destroyed.

        Args:
            cost_matrix: (B, N_img, N_txt) pairwise cost.

        Returns:
            Transport plan P of shape (B, N_img, N_txt).
        """
        B, N, M = cost_matrix.shape

        # Get current alpha/beta values (positive via softplus)
        alpha_val = self.alpha
        beta_val = self.beta

        # UOT exponents: these control the "unbalancedness"
        # fi → 1.0 means balanced; fi < 1.0 means unbalanced (mass can vanish)
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

        # Transport plan: P = diag(u) @ K @ diag(v)
        P = u.unsqueeze(2) * K * v.unsqueeze(1)
        return P

    def forward(self, img_feat, text_feat, text_mask=None):
        """
        Args:
            img_feat: (B, C, H, W) image features from backbone FPN.
            text_feat: (seq_len, B, C) text features (seq-first format).
            text_mask: (B, seq_len) boolean mask, True = padding token.

        Returns:
            aligned_img: (B, C, H, W) image features enhanced with
                         UOT-aligned text information.
        """
        B, C, H, W = img_feat.shape

        # Reshape image to (B, H*W, C)
        img_flat = img_feat.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # Text: (seq, B, C) → (B, seq, C)
        text_flat = text_feat.permute(1, 0, 2)  # (B, seq, C)

        # Mask out padding text tokens
        if text_mask is not None:
            valid_mask = ~text_mask  # True = valid
            text_flat = text_flat * valid_mask.unsqueeze(-1).float()

        # Project to alignment space
        img_proj = self.img_proj(img_flat)     # (B, HW, C)
        txt_proj = self.text_proj(text_flat)   # (B, seq, C)

        # Compute cost matrix as negative cosine similarity
        img_norm = F.normalize(img_proj, dim=-1)
        txt_norm = F.normalize(txt_proj, dim=-1)
        cost = 1.0 - torch.bmm(img_norm, txt_norm.transpose(1, 2))  # (B, HW, seq)

        # Compute UOT plan (stop gradients through Sinkhorn iterations)
        P = self.unbalanced_sinkhorn(cost)  # (B, HW, seq)

        # Transport text features to image positions
        # Scale by N so each position gets ~1 unit of text mass
        aligned_text = torch.bmm(P * P.shape[1], text_flat)  # (B, HW, C)
        aligned_text = self.output_proj(aligned_text)

        # Residual addition
        img_enhanced = img_flat + self.residual_weight * aligned_text
        img_enhanced = self.norm(img_enhanced)

        # Reshape back to spatial format
        img_enhanced = img_enhanced.permute(0, 2, 1).view(B, C, H, W)
        return img_enhanced
