"""Inference script for defocus deblurring with depth-based defocus map.

Usage:
    python inference_defocus.py \
        --model ./exp_dpdd_defocus/checkpoints/0200000.pt \
        --input /path/to/blurred/images \
        --depth /path/to/depth_maps \
        --output results/defocus \
        --device cuda
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from accelerate.utils import set_seed
from einops import rearrange

from model.cldm_defocus import ControlLDMDefocus
from model.gaussian_diffusion import Diffusion
from utils.common import instantiate_from_config, count_vram_usage
from utils.pipeline import Pipeline, pad_to_multiples_of, adaptive_instance_normalization
from utils.sampler import SpacedSampler
from utils.cond_fn import MSEGuidance, WeightedMSEGuidance


def check_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    return device


class DefocusPipeline(Pipeline):
    """Extended pipeline that passes defocus map through the model."""

    @count_vram_usage
    def run_diff(
        self,
        clean: torch.Tensor,
        steps: int,
        strength: float,
        tiled: bool,
        tile_size: int,
        tile_stride: int,
        pos_prompt: str,
        neg_prompt: str,
        cfg_scale: float,
        better_start: float,
        defocus_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bs, _, ori_h, ori_w = clean.shape
        pad_clean = pad_to_multiples_of(clean, multiple=64)
        h, w = pad_clean.shape[2:]

        if not tiled:
            cond = self.cldm.prepare_condition(pad_clean, [pos_prompt] * bs)
            uncond = self.cldm.prepare_condition(pad_clean, [neg_prompt] * bs)
        else:
            cond = self.cldm.prepare_condition_tiled(
                pad_clean, [pos_prompt] * bs, tile_size, tile_stride
            )
            uncond = self.cldm.prepare_condition_tiled(
                pad_clean, [neg_prompt] * bs, tile_size, tile_stride
            )

        # Add defocus map to conditions (resize to latent space)
        if defocus_map is not None:
            pad_defocus = pad_to_multiples_of(defocus_map, multiple=64)
            latent_h, latent_w = h // 8, w // 8
            dm_latent = F.interpolate(
                pad_defocus, size=(latent_h, latent_w),
                mode="bilinear", align_corners=False
            )
            cond["defocus_map"] = dm_latent
            uncond["defocus_map"] = dm_latent

        if self.cond_fn:
            self.cond_fn.load_target(pad_clean * 2 - 1)

        old_control_scales = self.cldm.control_scales
        self.cldm.control_scales = [strength] * 13

        if better_start:
            from utils.common import wavelet_decomposition
            _, low_freq = wavelet_decomposition(pad_clean)
            if not tiled:
                x_0 = self.cldm.vae_encode(low_freq)
            else:
                x_0 = self.cldm.vae_encode_tiled(low_freq, tile_size, tile_stride)
            x_T = self.diffusion.q_sample(
                x_0,
                torch.full((bs,), self.diffusion.num_timesteps - 1,
                           dtype=torch.long, device=self.device),
                torch.randn(x_0.shape, dtype=torch.float32, device=self.device)
            )
        else:
            x_T = torch.randn((bs, 4, h // 8, w // 8),
                              dtype=torch.float32, device=self.device)

        sampler = SpacedSampler(self.diffusion.betas)
        z = sampler.sample(
            model=self.cldm, device=self.device, steps=steps,
            batch_size=bs, x_size=(4, h // 8, w // 8),
            cond=cond, uncond=uncond, cfg_scale=cfg_scale,
            x_T=x_T, progress=True, progress_leave=True,
            cond_fn=self.cond_fn, tiled=tiled,
            tile_size=tile_size, tile_stride=tile_stride
        )

        if not tiled:
            x = self.cldm.vae_decode(z)
        else:
            x = self.cldm.vae_decode_tiled(z, tile_size // 8, tile_stride // 8)

        self.cldm.control_scales = old_control_scales
        sample = x[:, :, :ori_h, :ori_w]
        return sample

    @torch.no_grad()
    def run(
        self,
        lq: np.ndarray,
        steps: int,
        strength: float,
        tiled: bool,
        tile_size: int,
        tile_stride: int,
        pos_prompt: str,
        neg_prompt: str,
        cfg_scale: float,
        better_start: bool,
        defocus_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        lq_t = torch.tensor((lq / 255.).clip(0, 1),
                            dtype=torch.float32, device=self.device)
        lq_t = rearrange(lq_t, "n h w c -> n c h w").contiguous()
        self.set_final_size(lq_t)
        clean = lq_t

        # Prepare defocus map tensor
        dm_t = None
        if defocus_map is not None:
            dm_t = torch.tensor(defocus_map, dtype=torch.float32, device=self.device)
            if dm_t.ndim == 2:
                dm_t = dm_t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            elif dm_t.ndim == 3:
                dm_t = dm_t.unsqueeze(1)  # (B, 1, H, W)

        sample = self.run_diff(
            clean, steps, strength, tiled, tile_size, tile_stride,
            pos_prompt, neg_prompt, cfg_scale, better_start,
            defocus_map=dm_t
        )

        sample = (sample + 1) / 2
        sample = adaptive_instance_normalization(sample, clean)
        sample = rearrange(sample * 255., "n c h w -> n h w c")
        sample = sample.contiguous().clamp(0, 255).to(torch.uint8).cpu().numpy()
        return sample


class DefocusInferenceLoop:

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.loop_ctx = {}
        self.pipeline: DefocusPipeline = None
        self.init_model()
        self.init_cond_fn()
        self.init_pipeline()

    @count_vram_usage
    def init_model(self) -> None:
        self.cldm: ControlLDMDefocus = instantiate_from_config(
            OmegaConf.load("configs/inference/cldm_defocus.yaml")
        )
        self.cldm.load_state_dict(torch.load(self.args.model))
        self.cldm.eval().to(self.args.device)
        self.diffusion: Diffusion = instantiate_from_config(
            OmegaConf.load("configs/inference/diffusion.yaml")
        )
        self.diffusion.to(self.args.device)

    def init_cond_fn(self) -> None:
        if not self.args.guidance:
            self.cond_fn = None
            return
        if self.args.g_loss == "mse":
            cond_fn_cls = MSEGuidance
        elif self.args.g_loss == "w_mse":
            cond_fn_cls = WeightedMSEGuidance
        else:
            raise ValueError(self.args.g_loss)
        self.cond_fn = cond_fn_cls(
            scale=self.args.g_scale, t_start=self.args.g_start,
            t_stop=self.args.g_stop, space=self.args.g_space,
            repeat=self.args.g_repeat
        )

    def init_pipeline(self) -> None:
        self.pipeline = DefocusPipeline(
            self.cldm, self.diffusion, self.cond_fn, self.args.device
        )

    def setup(self) -> None:
        self.output_dir = self.args.output
        os.makedirs(self.output_dir, exist_ok=True)

    def load_depth_as_defocus(self, stem: str) -> Optional[np.ndarray]:
        """Load depth map and convert to normalised defocus map [0, 1]."""
        if not self.args.depth:
            return None
        npz_path = os.path.join(self.args.depth, f"{stem}.npz")
        png_path = os.path.join(self.args.depth, f"{stem}.png")
        if os.path.exists(npz_path):
            depth = np.load(npz_path)["depth"].astype(np.float32)
            # Normalise to [0, 1]
            valid = depth[np.isfinite(depth)]
            if valid.size > 0:
                lo, hi = np.percentile(valid, [2, 98])
                if hi > lo:
                    depth = np.clip((depth - lo) / (hi - lo), 0, 1)
                else:
                    depth = np.zeros_like(depth)
            return depth
        elif os.path.exists(png_path):
            return np.array(Image.open(png_path).convert("L")).astype(np.float32) / 255.0
        return None

    @torch.no_grad()
    def run(self) -> None:
        self.setup()
        img_exts = {".png", ".jpg", ".jpeg", ".PNG"}

        if os.path.isdir(self.args.input):
            file_names = sorted(
                f for f in os.listdir(self.args.input)
                if os.path.splitext(f)[-1] in img_exts
            )
            file_paths = [os.path.join(self.args.input, f) for f in file_names]
        else:
            file_paths = [self.args.input]

        for file_path in file_paths:
            lq = np.array(Image.open(file_path).convert("RGB"))
            stem = os.path.splitext(os.path.basename(file_path))[0]
            print(f"Processing: {file_path}")

            defocus = self.load_depth_as_defocus(stem)

            for i in range(self.args.n_samples):
                sample = self.pipeline.run(
                    lq[None], self.args.steps, 1.0, self.args.tiled,
                    self.args.tile_size, self.args.tile_stride,
                    self.args.pos_prompt, self.args.neg_prompt,
                    self.args.cfg_scale, self.args.better_start,
                    defocus_map=defocus
                )[0]

                suffix = f"_{i}" if self.args.n_samples > 1 else ""
                save_path = os.path.join(self.output_dir, f"{stem}{suffix}.png")
                Image.fromarray(sample).save(save_path)
                print(f"  Saved: {save_path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--better_start", action="store_true")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_stride", type=int, default=256)
    parser.add_argument("--pos_prompt", type=str, default="")
    parser.add_argument("--neg_prompt", type=str,
                        default="low quality, blurry, low-resolution, noisy, unsharp, weird textures")
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--depth", type=str, default=None,
                        help="Directory containing depth maps (.npz or .png)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--guidance", action="store_true")
    parser.add_argument("--g_loss", type=str, default="w_mse", choices=["mse", "w_mse"])
    parser.add_argument("--g_scale", type=float, default=0.0)
    parser.add_argument("--g_start", type=int, default=1001)
    parser.add_argument("--g_stop", type=int, default=-1)
    parser.add_argument("--g_space", type=str, default="latent")
    parser.add_argument("--g_repeat", type=int, default=1)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = check_device(args.device)
    set_seed(args.seed)
    DefocusInferenceLoop(args).run()
    print("done!")


if __name__ == "__main__":
    main()
