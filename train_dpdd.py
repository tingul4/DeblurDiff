"""Training script for defocus deblurring on DPDD dataset.

Usage:
    accelerate launch --main_process_port 4562 train_dpdd.py \
        --config configs/train/train_dpdd.yaml
"""

import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from safetensors.torch import load_file as load_safetensors

from model.cldm_defocus import ControlLDMDefocus
from model.gaussian_diffusion import Diffusion
from utils.common import instantiate_from_config
from utils.sampler import SpacedSampler


def load_sd_weights(path: str) -> dict:
    """Load SD pretrained weights from .ckpt or .safetensors."""
    if path.endswith(".safetensors"):
        return load_safetensors(path)
    else:
        return torch.load(path, map_location="cpu")["state_dict"]


def log_txt_as_img(wh, xc):
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.load_default()
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))
        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            pass
        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    return torch.tensor(txts)


@torch.no_grad()
def log_validation_images(
    pure_cldm, sampler, dataset, writer, global_step, device,
    n_vis=4, steps=50,
):
    """Sample a few validation images and log to TensorBoard."""
    pure_cldm.eval()
    indices = list(range(0, min(len(dataset), n_vis * 10), max(1, len(dataset) // n_vis)))[:n_vis]

    all_lq, all_sample = [], []
    for idx in indices:
        batch = dataset[idx]
        lq = torch.tensor(batch["lq"]).permute(2, 0, 1).unsqueeze(0).float().to(device)  # (1,3,H,W)
        gt = torch.tensor(batch["gt"]).permute(2, 0, 1).unsqueeze(0).float().to(device)

        z_0 = pure_cldm.vae_encode(gt)
        cond = pure_cldm.prepare_condition(lq, [batch["prompt"]])

        if "depth" in batch:
            depth = torch.tensor(batch["depth"]).float().to(device)
            if depth.ndim == 2:
                depth = depth.unsqueeze(0).unsqueeze(0)
            elif depth.ndim == 3:
                depth = depth.unsqueeze(1)
            d_flat = depth.view(1, -1)
            d_min = d_flat.min(dim=1, keepdim=True)[0].view(1, 1, 1, 1)
            d_max = d_flat.max(dim=1, keepdim=True)[0].view(1, 1, 1, 1)
            defocus_map = (depth - d_min) / (d_max - d_min + 1e-8)
            defocus_map = F.interpolate(defocus_map, size=z_0.shape[-2:], mode="bilinear", align_corners=False)
            cond["defocus_map"] = defocus_map

        uncond = pure_cldm.prepare_condition(lq, [""])
        if "defocus_map" in cond:
            uncond["defocus_map"] = cond["defocus_map"]

        h, w = z_0.shape[-2:]
        x_T = torch.randn((1, 4, h, w), dtype=torch.float32, device=device)
        z = sampler.sample(
            model=pure_cldm, device=device, steps=steps,
            batch_size=1, x_size=(4, h, w),
            cond=cond, uncond=uncond, cfg_scale=1.0,
            x_T=x_T, progress=False, progress_leave=False,
            cond_fn=None, tiled=False, tile_size=512, tile_stride=256,
        )
        sample = pure_cldm.vae_decode(z)
        sample = (sample + 1) / 2  # [-1,1] → [0,1]
        lq_vis = lq.clamp(0, 1)

        all_lq.append(lq_vis[0])
        all_sample.append(sample[0].clamp(0, 1))

    if all_lq:
        grid_lq = make_grid(all_lq, nrow=n_vis, normalize=False)
        grid_sample = make_grid(all_sample, nrow=n_vis, normalize=False)
        grid = make_grid([grid_lq, grid_sample], nrow=1, normalize=False)
        writer.add_image("val/lq_vs_sample", grid, global_step)

    pure_cldm.train()


def main(args) -> None:
    accelerator = Accelerator(split_batches=True)
    set_seed(231)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Setup experiment folder
    if accelerator.is_local_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    # Create model (ControlLDMDefocus with defocus-aware LKPN + EAC gating)
    cldm: ControlLDMDefocus = instantiate_from_config(cfg.model.cldm)
    sd = load_sd_weights(cfg.train.sd_path)
    unused = cldm.load_pretrained_sd(sd)
    if accelerator.is_local_main_process:
        print(f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
              f"unused weights: {unused}")

    if cfg.train.resume:
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
        if accelerator.is_local_main_process:
            print(f"strictly load controlnet weight from checkpoint: {cfg.train.resume}")
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_local_main_process:
            print(f"strictly load controlnet weight from pretrained SD\n"
                  f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                  f"weights initialized from scratch: {init_with_scratch}")

    if cfg.train.resume_kpn:
        sd_kpn = torch.load(cfg.train.resume_kpn, map_location="cpu")
        cldm.kpn.load_state_dict(sd_kpn, strict=True)
        if accelerator.is_local_main_process:
            print(f"strictly load KPN weight from checkpoint: {cfg.train.resume_kpn}")

    # Create diffusion
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    # Count trainable parameters
    if accelerator.is_local_main_process:
        n_ctrl = sum(p.numel() for p in cldm.controlnet.parameters())
        n_kpn = sum(p.numel() for p in cldm.kpn.parameters())
        n_gate = sum(p.numel() for p in cldm.eac_gate.parameters())
        print(f"Trainable params — ControlNet: {n_ctrl/1e6:.1f}M, "
              f"KPN: {n_kpn/1e6:.1f}M, EAC gate: {n_gate/1e3:.1f}K")

    # Setup optimizer: ControlNet + KPN + EAC gate
    global_lr = cfg.train.learning_rate
    controlnet_params = list(cldm.controlnet.parameters())
    kpn_params = list(cldm.kpn.parameters())
    gate_params = list(cldm.eac_gate.parameters())

    opt = torch.optim.AdamW([
        {'params': controlnet_params, 'lr': global_lr},
        {'params': kpn_params, 'lr': global_lr},
        {'params': gate_params, 'lr': global_lr},
    ])

    # Setup data
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True, drop_last=True
    )
    if accelerator.is_local_main_process:
        print(f"Dataset contains {len(dataset):,} images")

    # Prepare models for training
    cldm.train().to(device)
    diffusion.to(device)
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDMDefocus = accelerator.unwrap_model(cldm)

    # Variables for monitoring
    global_step = 0
    max_steps = cfg.train.train_steps
    step_losses = {"total": [], "simple": [], "kpn": [], "gate": []}
    epoch = 0
    epoch_loss = []
    sampler = SpacedSampler(diffusion.betas)
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

    while global_step < max_steps:
        pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process,
                    unit="batch", total=len(loader))
        for batch in loader:
            # DPDDDataset returns dict with 'gt', 'lq', 'prompt', optionally 'depth'
            gt = batch["gt"]
            lq = batch["lq"]
            prompt = batch["prompt"]

            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(device)
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(device)

            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)
                clean = lq
                cond = pure_cldm.prepare_condition(clean, prompt)

                # Prepare defocus map for latent space
                if "depth" in batch:
                    depth = batch["depth"].float().to(device)  # (B, H, W)
                    if depth.ndim == 3:
                        depth = depth.unsqueeze(1)  # (B, 1, H, W)
                    # Normalise depth to [0, 1] as defocus proxy
                    b = depth.shape[0]
                    depth_flat = depth.view(b, -1)
                    d_min = depth_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
                    d_max = depth_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
                    defocus_map = (depth - d_min) / (d_max - d_min + 1e-8)
                    # Resize to latent space
                    defocus_map = F.interpolate(
                        defocus_map, size=z_0.shape[-2:],
                        mode="bilinear", align_corners=False
                    )
                    cond["defocus_map"] = defocus_map
                elif "defocus" in batch:
                    defocus = batch["defocus"].float().to(device)
                    if defocus.ndim == 3:
                        defocus = defocus.unsqueeze(1)
                    defocus = F.interpolate(
                        defocus, size=z_0.shape[-2:],
                        mode="bilinear", align_corners=False
                    )
                    cond["defocus_map"] = defocus

            t = torch.randint(0, diffusion.num_timesteps, (z_0.shape[0],), device=device)
            loss, loss_dict = diffusion.p_losses(cldm, z_0, t, cond, return_dict=True)

            opt.zero_grad()
            accelerator.backward(loss)
            # Gradient clipping
            accelerator.clip_grad_norm_(
                list(cldm.parameters()), max_norm=1.0
            )
            opt.step()

            accelerator.wait_for_everyone()

            global_step += 1
            step_losses["total"].append(loss_dict["loss_total"])
            step_losses["simple"].append(loss_dict["loss_simple"])
            step_losses["kpn"].append(loss_dict["loss_kpn"])
            step_losses["gate"].append(loss_dict["loss_gate"])
            epoch_loss.append(loss_dict["loss_total"])
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Step: {global_step:07d}, "
                f"L={loss_dict['loss_total']:.4f} "
                f"(eps={loss_dict['loss_simple']:.4f} "
                f"kpn={loss_dict['loss_kpn']:.4f} "
                f"gate={loss_dict['loss_gate']:.4f})"
            )

            # Log losses and metrics
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                if accelerator.is_local_main_process:
                    n = len(step_losses["total"])
                    writer.add_scalar("loss/total", sum(step_losses["total"]) / n, global_step)
                    writer.add_scalar("loss/eps_simple", sum(step_losses["simple"]) / n, global_step)
                    writer.add_scalar("loss/kpn", sum(step_losses["kpn"]) / n, global_step)
                    writer.add_scalar("loss/gate", sum(step_losses["gate"]) / n, global_step)

                    # Log KPN alpha (defocus→radius scale)
                    writer.add_scalar("params/kpn_alpha", pure_cldm.kpn.alpha.item(), global_step)

                    # Log EAC gate statistics (mean gate value per level)
                    if hasattr(pure_cldm, 'eac_gate') and "defocus_map" in cond:
                        with torch.no_grad():
                            dm = cond["defocus_map"][:1]  # use first sample
                            for li, gate in enumerate(pure_cldm.eac_gate.gates):
                                g_val = gate(dm)
                                writer.add_scalar(f"gate/level_{li:02d}_mean", g_val.mean().item(), global_step)
                            # Also log overall gate mean across all levels
                            g_all = torch.stack([gate(dm).mean() for gate in pure_cldm.eac_gate.gates])
                            writer.add_scalar("gate/mean_all_levels", g_all.mean().item(), global_step)

                    # Log gradient norms
                    ctrl_grad = torch.nn.utils.clip_grad_norm_(
                        pure_cldm.controlnet.parameters(), float('inf')
                    )
                    kpn_grad = torch.nn.utils.clip_grad_norm_(
                        pure_cldm.kpn.parameters(), float('inf')
                    )
                    gate_grad = torch.nn.utils.clip_grad_norm_(
                        pure_cldm.eac_gate.parameters(), float('inf')
                    )
                    writer.add_scalar("grad_norm/controlnet", ctrl_grad.item(), global_step)
                    writer.add_scalar("grad_norm/kpn", kpn_grad.item(), global_step)
                    writer.add_scalar("grad_norm/eac_gate", gate_grad.item(), global_step)

                    writer.flush()

                for k in step_losses:
                    step_losses[k].clear()

            # Save checkpoint
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_local_main_process:
                    checkpoint = pure_cldm.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)
                    print(f"Saved checkpoint to {ckpt_path}")

            # Log validation images
            if global_step % cfg.train.image_every == 0 and global_step > 0:
                if accelerator.is_local_main_process:
                    try:
                        log_validation_images(
                            pure_cldm, sampler, dataset,
                            writer, global_step, device, n_vis=4, steps=50,
                        )
                    except Exception as e:
                        print(f"Warning: validation image logging failed: {e}")

            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1
        if accelerator.is_local_main_process:
            avg_epoch_loss = sum(epoch_loss) / max(len(epoch_loss), 1)
            writer.add_scalar("loss/loss_epoch", avg_epoch_loss, global_step)
        epoch_loss.clear()

    if accelerator.is_local_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
