"""EAC Spatial Gating with Defocus Map.

Implements:
    F_out = g(M) * F_EAC + (1 - g(M)) * F_in

where g is a learned monotonic gate conditioned on the defocus map M.
Focused regions (M ≈ 0) pass through unchanged; defocused regions get
full EAC processing.

Supervised loss:
    L_gate = ||g(M) - normalize(M)||_1 + lambda * TV(g(M))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DefocusGate(nn.Module):
    """Spatial gating module conditioned on defocus map.

    Takes a defocus map M (B, 1, H, W) in [0, 1] and produces a gate
    g(M) (B, 1, H, W) in [0, 1] via a small learned network that
    preserves monotonicity.

    The gate is applied as:
        output = g * f_eac + (1 - g) * f_input
    """

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        # Small ConvNet: M → g(M), preserving spatial structure
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, defocus_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            defocus_map: (B, 1, H, W) normalised defocus map [0, 1].

        Returns:
            gate: (B, 1, H, W) spatial gate values [0, 1].
        """
        return self.net(defocus_map)


class EACSpatialGating(nn.Module):
    """Wraps ControlNet output with defocus-map spatial gating.

    In the forward pass of ControlLDM, the ControlNet produces control
    signals that are added to UNet skip connections. This module gates
    each control signal spatially using the defocus map:

        control_gated[i] = g_i(M) * control[i]

    where g_i is the gate at resolution level i.

    For focused regions (g ≈ 0), the control signal is suppressed —
    the UNet operates unguided (no deblurring needed).
    For defocused regions (g ≈ 1), the full control signal is applied.
    """

    def __init__(self, num_levels: int = 13, hidden_dim: int = 16) -> None:
        super().__init__()
        # One lightweight gate per control level
        self.gates = nn.ModuleList([
            DefocusGate(hidden_dim=hidden_dim)
            for _ in range(num_levels)
        ])

    def forward(
        self,
        controls: list,
        defocus_map: torch.Tensor,
    ) -> list:
        """Gate each control signal with defocus-aware spatial mask.

        Args:
            controls: List of 13 control tensors from ControlNet,
                      each (B, C_i, H_i, W_i).
            defocus_map: (B, 1, H, W) defocus map at original latent res.

        Returns:
            gated_controls: List of gated control tensors.
        """
        gated = []
        for ctrl, gate_module in zip(controls, self.gates):
            # Resize defocus map to match this level's resolution
            _, _, h, w = ctrl.shape
            dm = F.interpolate(
                defocus_map, size=(h, w), mode="bilinear", align_corners=False
            )
            g = gate_module(dm)  # (B, 1, h, w)
            gated.append(ctrl * g)
        return gated

    def compute_gate_loss(
        self,
        defocus_map: torch.Tensor,
        control_sizes: list,
        tv_lambda: float = 0.01,
    ) -> torch.Tensor:
        """Supervised loss to align gates with normalised defocus map.

        L_gate = sum_i ||g_i(M_i) - M_i||_1 + tv_lambda * TV(g_i(M_i))

        Args:
            defocus_map: (B, 1, H, W) defocus map at original latent res.
            control_sizes: List of (H_i, W_i) tuples for each level.
            tv_lambda: Weight for total variation regularization.

        Returns:
            loss: Scalar gate supervision loss.
        """
        loss = torch.tensor(0.0, device=defocus_map.device)
        for (h, w), gate_module in zip(control_sizes, self.gates):
            dm = F.interpolate(
                defocus_map, size=(h, w), mode="bilinear", align_corners=False
            )
            g = gate_module(dm)
            # L1 alignment
            loss = loss + F.l1_loss(g, dm)
            # Total variation
            if tv_lambda > 0:
                tv_h = torch.abs(g[:, :, 1:, :] - g[:, :, :-1, :]).mean()
                tv_w = torch.abs(g[:, :, :, 1:] - g[:, :, :, :-1]).mean()
                loss = loss + tv_lambda * (tv_h + tv_w)
        return loss / len(self.gates)
