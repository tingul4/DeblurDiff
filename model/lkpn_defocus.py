"""LKPN with defocus map-constrained disk kernel radius.

The key idea:
  - Defocus map M(x,y) provides a radius prior: r0(x,y) = alpha * M(x,y)
  - LKPN predicts kernel shape weights w(x,y) and a radius residual delta_r(x,y)
  - Final kernel is masked by a soft disk: K = disk_mask(r0 + delta_r) * w
  - Focused regions (M ≈ 0) auto-degrade to delta kernel (identity)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lkpn import IDynamicConv
from model.unet import UNetModel


class DefocusLKPN(nn.Module):
    """Defocus-aware Local Kernel Prediction Network.

    Instead of directly predicting full kernel weights, we predict:
      1. Kernel shape weights w(x,y) of size (4 * K * K) via UNet
      2. Radius residual delta_r(x,y) via a small head on UNet features

    The defocus map M provides the radius prior r0 = alpha * M.
    A soft disk mask zeros out kernel entries beyond radius (r0 + delta_r).

    Args:
        kernel_size: Spatial kernel size (default 5).
        alpha_init: Initial scale factor for defocus map → radius mapping.
        max_radius: Maximum allowed kernel radius (clamp).
    """

    def __init__(
        self,
        kernel_size: int = 5,
        alpha_init: float = 2.0,
        max_radius: float = 3.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.max_radius = max_radius

        # Learnable alpha: defocus map → radius scale
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # UNet backbone: 8ch (c_img + hint) + 1ch (defocus map) = 9ch input
        # Output: 4*K*K kernel weights + 1 radius residual = 4*K*K + 1
        out_ch = 4 * kernel_size * kernel_size + 1
        self.unet = UNetModel(
            use_checkpoint=True,
            image_size=32,
            in_channels=9,   # 4 (c_img) + 4 (hint) + 1 (defocus map)
            out_channels=out_ch,
            model_channels=128,
            attention_resolutions=[4, 2, 1],
            num_res_blocks=2,
            channel_mult=[1, 2, 4, 4],
            num_head_channels=64,
            use_spatial_transformer=True,
            use_linear_in_transformer=True,
            transformer_depth=1,
            context_dim=1024,
            legacy=False,
        )

        self.idy_conv = IDynamicConv()

        # Pre-compute distance grid for disk mask
        half = kernel_size // 2
        yy, xx = torch.meshgrid(
            torch.arange(kernel_size, dtype=torch.float32) - half,
            torch.arange(kernel_size, dtype=torch.float32) - half,
            indexing="ij",
        )
        # (K, K) distance from center
        self.register_buffer("dist_grid", torch.sqrt(xx ** 2 + yy ** 2))

    def _soft_disk_mask(self, radius: torch.Tensor) -> torch.Tensor:
        """Generate soft disk mask from per-pixel radius.

        Args:
            radius: (B, 1, H, W) effective radius at each spatial location.

        Returns:
            mask: (B, K*K, H, W) soft disk mask in [0, 1].
        """
        K = self.kernel_size
        # dist_grid: (K, K) → (1, K*K, 1, 1)
        dist = self.dist_grid.reshape(1, K * K, 1, 1)
        # radius: (B, 1, H, W) → broadcast with dist
        # Soft boundary: sigmoid with sharpness β
        beta = 5.0
        mask = torch.sigmoid(beta * (radius - dist))
        return mask

    def forward(
        self,
        x: torch.Tensor,
        hint: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        defocus_map: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: VAE-encoded clean image (B, 4, H, W).
            hint: Noisy latent (B, 4, H, W).
            timesteps: Diffusion timesteps (B,).
            context: CLIP text embeddings.
            defocus_map: Normalised defocus map (B, 1, H, W) in [0, 1].
                         If None, falls back to standard LKPN (no radius constraint).

        Returns:
            result: Kernel-applied output (B, 4, H, W).
        """
        K = self.kernel_size
        n_kernel_ch = 4 * K * K

        if defocus_map is not None:
            # Resize defocus map to match latent spatial size
            if defocus_map.shape[-2:] != x.shape[-2:]:
                defocus_map = F.interpolate(
                    defocus_map, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            merge = torch.cat([x, hint, defocus_map], dim=1)  # (B, 9, H, W)
        else:
            # Fallback: zero defocus map
            zeros = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3],
                                device=x.device, dtype=x.dtype)
            merge = torch.cat([x, hint, zeros], dim=1)

        # UNet forward
        out = self.unet(x=merge, timesteps=timesteps, context=context)

        # Split: kernel weights + radius residual
        kernel_weights = out[:, :n_kernel_ch]       # (B, 4*K*K, H, W)
        delta_r = out[:, n_kernel_ch:n_kernel_ch+1] # (B, 1, H, W)
        delta_r = torch.tanh(delta_r) * 1.0         # clamp residual to [-1, 1]

        # Compute effective radius
        if defocus_map is not None:
            r0 = self.alpha * defocus_map  # (B, 1, H, W)
        else:
            r0 = torch.zeros_like(delta_r)

        radius = torch.clamp(r0 + delta_r, min=0.0, max=self.max_radius)

        # Generate soft disk mask and apply to kernel weights
        # mask: (B, K*K, H, W), need to expand for 4 output channels
        disk_mask = self._soft_disk_mask(radius)  # (B, K*K, H, W)
        # Softmax kernel weights per group to ensure non-negative values,
        # then apply disk mask and re-normalise.  This prevents the old
        # failure mode where positive/negative raw weights could sum to
        # near-zero, causing division-by-epsilon to explode.
        mk = kernel_weights.view(x.shape[0], 4, K * K, x.shape[2], x.shape[3])
        mk = F.softmax(mk, dim=2)                       # (B,4,K*K,H,W) ≥0, Σ=1
        disk_mask_grouped = disk_mask.unsqueeze(1)       # (B,1,K*K,H,W)
        mk = mk * disk_mask_grouped
        mk_sum = mk.sum(dim=2, keepdim=True).clamp(min=1e-6)
        mk = mk / mk_sum
        masked_kernel = mk.view(x.shape[0], 4 * K * K, x.shape[2], x.shape[3])

        # Apply spatially-varying convolution + residual
        result = self.idy_conv(x, masked_kernel) + x

        return result
