"""ControlLDM with defocus-aware LKPN and EAC spatial gating.

Changes from original cldm.py:
  1. Uses DefocusLKPN instead of LKPN (defocus map → kernel radius constraint)
  2. Adds EACSpatialGating to gate ControlNet outputs by defocus map
  3. Forward pass accepts defocus_map in cond dict
"""

from typing import Tuple, Set, List, Dict

import torch
from torch import nn
import torch.nn.functional as F

from model import (
    ControlledUnetModel, ControlNet,
    AutoencoderKL, FrozenOpenCLIPEmbedder
)
from utils.common import sliding_windows, count_vram_usage, gaussian_weights

from model.lkpn_defocus import DefocusLKPN
from model.eac_gate import EACSpatialGating


def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class ControlLDMDefocus(nn.Module):

    def __init__(
            self,
            unet_cfg,
            vae_cfg,
            clip_cfg,
            controlnet_cfg,
            latent_scale_factor,
            # Defocus-specific params
            kernel_size: int = 5,
            alpha_init: float = 2.0,
            max_radius: float = 3.0,
            gate_hidden_dim: int = 16,
            gate_tv_lambda: float = 0.01,
    ):
        super().__init__()
        self.unet = ControlledUnetModel(**unet_cfg)
        self.kpn = DefocusLKPN(
            kernel_size=kernel_size,
            alpha_init=alpha_init,
            max_radius=max_radius,
        )
        self.vae = AutoencoderKL(**vae_cfg)
        self.clip = FrozenOpenCLIPEmbedder(**clip_cfg)
        self.controlnet = ControlNet(**controlnet_cfg)
        self.eac_gate = EACSpatialGating(
            num_levels=13,
            hidden_dim=gate_hidden_dim,
        )
        self.scale_factor = latent_scale_factor
        self.control_scales = [1.0] * 13
        self.gate_tv_lambda = gate_tv_lambda

    @torch.no_grad()
    def load_pretrained_sd(self, sd: Dict[str, torch.Tensor]) -> Set[str]:
        module_map = {
            "unet": "model.diffusion_model",
            "vae": "first_stage_model",
            "clip": "cond_stage_model",
        }
        modules = [("unet", self.unet), ("vae", self.vae), ("clip", self.clip)]
        used = set()
        for name, module in modules:
            init_sd = {}
            scratch_sd = module.state_dict()
            for key in scratch_sd:
                target_key = ".".join([module_map[name], key])
                init_sd[key] = sd[target_key].clone()
                used.add(target_key)
            module.load_state_dict(init_sd, strict=True)
        unused = set(sd.keys()) - used
        for module in [self.vae, self.clip, self.unet]:
            module.eval()
            module.train = disabled_train
            for p in module.parameters():
                p.requires_grad = False
        return unused

    @torch.no_grad()
    def load_controlnet_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:
        self.controlnet.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def load_controlnet_from_unet(self) -> Tuple[Set[str]]:
        unet_sd = self.unet.state_dict()
        scratch_sd = self.controlnet.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()
        for key in scratch_sd:
            if key in unet_sd:
                this, target = scratch_sd[key], unet_sd[key]
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        self.controlnet.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch

    def vae_encode(self, image: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            return self.vae.encode(image).sample() * self.scale_factor
        else:
            return self.vae.encode(image).mode() * self.scale_factor

    def vae_encode_tiled(self, image: torch.Tensor, tile_size: int, tile_stride: int,
                         sample: bool = True) -> torch.Tensor:
        bs, _, h, w = image.shape
        z = torch.zeros((bs, 4, h // 8, w // 8), dtype=torch.float32, device=image.device)
        weights = gaussian_weights(tile_size // 8, tile_size // 8)[None, None]
        weights = torch.tensor(weights, dtype=torch.float32, device=image.device)
        tiles = sliding_windows(h // 8, w // 8, tile_size // 8, tile_stride // 8)
        for hi, hi_end, wi, wi_end in tiles:
            tile_image = image[:, :, hi * 8:hi_end * 8, wi * 8:wi_end * 8]
            z[:, :, hi:hi_end, wi:wi_end] = self.vae_encode(tile_image, sample=sample)
        return z

    def vae_decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z / self.scale_factor)

    @count_vram_usage
    def vae_decode_tiled(self, z: torch.Tensor, tile_size: int, tile_stride: int) -> torch.Tensor:
        bs, _, h, w = z.shape
        image = torch.zeros((bs, 3, h * 8, w * 8), dtype=torch.float32, device=z.device)
        count = torch.zeros_like(image, dtype=torch.float32)
        weights = gaussian_weights(tile_size * 8, tile_size * 8)[None, None]
        weights = torch.tensor(weights, dtype=torch.float32, device=z.device)
        tiles = sliding_windows(h, w, tile_size, tile_stride)
        for hi, hi_end, wi, wi_end in tiles:
            tile_z = z[:, :, hi:hi_end, wi:wi_end]
            image[:, :, hi * 8:hi_end * 8, wi * 8:wi_end * 8] += self.vae_decode(tile_z) * weights
            count[:, :, hi * 8:hi_end * 8, wi * 8:wi_end * 8] += weights
        image.div_(count)
        return image

    def prepare_condition(self, clean: torch.Tensor, txt: List[str]) -> Dict[str, torch.Tensor]:
        return dict(
            c_txt=self.clip.encode(txt),
            c_img=self.vae_encode(clean, sample=True)
        )

    @count_vram_usage
    def prepare_condition_tiled(self, clean: torch.Tensor, txt: List[str], tile_size: int, tile_stride: int) -> Dict[
        str, torch.Tensor]:
        return dict(
            c_txt=self.clip.encode(txt),
            c_img=self.vae_encode_tiled(clean, tile_size, tile_stride, sample=False)
        )

    def forward(self, x_noisy, t, cond):
        c_txt = cond["c_txt"]
        c_img = cond["c_img"]
        defocus_map = cond.get("defocus_map", None)  # (B, 1, H_latent, W_latent) or None

        c_img_kpn, x_noisy_kpn = c_img.contiguous(), x_noisy.contiguous()

        # DefocusLKPN: pass defocus map for radius constraint
        lr_kpn = self.kpn(c_img_kpn, x_noisy_kpn, t, c_txt, defocus_map=defocus_map)

        cond_hint = torch.cat((c_img, lr_kpn), dim=1)
        control = self.controlnet(
            x=x_noisy, hint=cond_hint,
            timesteps=t, context=c_txt
        )
        control = [c * scale for c, scale in zip(control, self.control_scales)]

        # EAC Spatial Gating: gate control signals by defocus map
        if defocus_map is not None:
            control = self.eac_gate(control, defocus_map)

        eps = self.unet(
            x=x_noisy, timesteps=t,
            context=c_txt, control=control, only_mid_control=False
        )

        return eps, lr_kpn

    def compute_gate_loss(self, cond) -> torch.Tensor:
        """Compute EAC gate supervision loss for the current forward pass."""
        defocus_map = cond.get("defocus_map", None)
        if defocus_map is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # Get control sizes by doing a dummy check on ControlNet architecture
        # The 13 control levels have known downsampling pattern for SD 2.1:
        # 4 levels × {2 res_blocks + 1 downsample} + 1 mid = 13
        # Resolutions: base, base, base, base/2, base/2, base/2, base/4, ...
        h, w = defocus_map.shape[-2:]
        # Standard SD 2.1 control level sizes
        sizes = []
        for level_mult in [1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 8]:
            sizes.append((h // level_mult, w // level_mult))

        return self.eac_gate.compute_gate_loss(
            defocus_map, sizes, tv_lambda=self.gate_tv_lambda
        )
