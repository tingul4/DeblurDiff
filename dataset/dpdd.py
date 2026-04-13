"""DPDD (Dual-Pixel Defocus Deblurring) dataset loader.

Loads paired (blurred source, sharp target) images and optionally depth maps
/ defocus maps for defocus deblurring training.

Directory layout expected:
    <split>_c/
        source/          # blurred images
        target/          # sharp ground-truth images
        depth_map_depthpro/   # metric depth (.npz + normalised .png)
"""

from typing import Dict, Union, Optional, Tuple
import os
import random

import numpy as np
from PIL import Image
import torch.utils.data as data


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DPDDDataset(data.Dataset):
    """Dataset for DPDD defocus deblurring.

    Args:
        dataset_root: Root of the DPDD dataset (e.g. .../dd_dp_dataset_png).
        split: One of 'train', 'val', 'test'.
        out_size: Output crop size (square).
        crop_type: 'random' | 'center' | 'none'.
        depth_subdir: Subdirectory name for depth maps (None = don't load).
        defocus_subdir: Subdirectory name for defocus maps (None = don't load).
        prompt_prob: Probability of returning a non-empty text prompt.
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        out_size: int = 512,
        crop_type: str = "random",
        depth_subdir: Optional[str] = "depth_map_depthpro",
        defocus_subdir: Optional[str] = None,
        prompt_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.out_size = out_size
        self.crop_type = crop_type
        assert crop_type in ("random", "center", "none")
        self.depth_subdir = depth_subdir
        self.defocus_subdir = defocus_subdir
        self.prompt_prob = prompt_prob

        split_dir = os.path.join(dataset_root, f"{split}_c")
        self.source_dir = os.path.join(split_dir, "source")
        self.target_dir = os.path.join(split_dir, "target")
        self.depth_dir = (
            os.path.join(split_dir, depth_subdir) if depth_subdir else None
        )
        self.defocus_dir = (
            os.path.join(split_dir, defocus_subdir) if defocus_subdir else None
        )

        # Build paired file list: match by sorted order (source & target same count)
        source_files = sorted(
            f for f in os.listdir(self.source_dir)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        )
        target_files = sorted(
            f for f in os.listdir(self.target_dir)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        )
        assert len(source_files) == len(target_files), (
            f"Mismatch: {len(source_files)} sources vs {len(target_files)} targets"
        )

        self.pairs = []
        for src, tgt in zip(source_files, target_files):
            src_stem = os.path.splitext(src)[0]
            entry = {
                "source": os.path.join(self.source_dir, src),
                "target": os.path.join(self.target_dir, tgt),
                "stem": src_stem,
            }
            if self.depth_dir:
                npz = os.path.join(self.depth_dir, f"{src_stem}.npz")
                if os.path.exists(npz):
                    entry["depth"] = npz
            if self.defocus_dir:
                defocus_path = os.path.join(self.defocus_dir, f"{src_stem}.png")
                if os.path.exists(defocus_path):
                    entry["defocus"] = defocus_path
            self.pairs.append(entry)

    def _load_image(self, path: str) -> np.ndarray:
        """Load image as RGB float32 [0, 1]."""
        img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        return img

    def _load_depth(self, path: str) -> np.ndarray:
        """Load metric depth from npz as float32 (H, W)."""
        return np.load(path)["depth"].astype(np.float32)

    def _load_defocus(self, path: str) -> np.ndarray:
        """Load defocus map as float32 [0, 1] (H, W)."""
        img = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
        return img

    def _crop(
        self, *arrays: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """Apply consistent random/center crop to all arrays."""
        h, w = arrays[0].shape[:2]
        s = self.out_size

        if self.crop_type == "none" or (h <= s and w <= s):
            return arrays

        if self.crop_type == "random":
            top = random.randint(0, h - s)
            left = random.randint(0, w - s)
        else:  # center
            top = (h - s) // 2
            left = (w - s) // 2

        out = []
        for arr in arrays:
            if arr.ndim == 3:
                out.append(arr[top:top + s, left:left + s, :])
            else:
                out.append(arr[top:top + s, left:left + s])
        return tuple(out)

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        entry = self.pairs[index]

        source = self._load_image(entry["source"])  # blurred
        target = self._load_image(entry["target"])  # sharp GT

        # Optional maps
        depth = self._load_depth(entry["depth"]) if "depth" in entry else None
        defocus = self._load_defocus(entry["defocus"]) if "defocus" in entry else None

        # Crop all consistently
        crop_list = [source, target]
        if depth is not None:
            crop_list.append(depth)
        if defocus is not None:
            crop_list.append(defocus)

        cropped = self._crop(*crop_list)
        source, target = cropped[0], cropped[1]
        idx = 2
        if depth is not None:
            depth = cropped[idx]; idx += 1
        if defocus is not None:
            defocus = cropped[idx]; idx += 1

        # Prompt
        prompt = "" if random.random() > self.prompt_prob else "sharp, high quality"

        # gt: [-1, 1], lq: [0, 1]  (matching CodeformerDataset convention)
        gt = (target * 2 - 1).astype(np.float32)
        lq = source.astype(np.float32)

        result = {
            "gt": gt,
            "lq": lq,
            "prompt": prompt,
        }
        if depth is not None:
            result["depth"] = depth
        if defocus is not None:
            result["defocus"] = defocus

        return result

    def __len__(self) -> int:
        return len(self.pairs)
