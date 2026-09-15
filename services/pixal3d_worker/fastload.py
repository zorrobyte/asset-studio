"""Speed adapters for the upstream Pixal3D / o_voxel code paths. Output-neutral: they change how long the stage
takes, not what it produces. Both are installed by generate.py before the upstream modules run; upstream code is
not modified (measured on the mug sample with cProfile, 2026-09-15).

1. skip_random_init: upstream `pixal3d.models.from_pretrained` constructs every model with PyTorch's default
   random init (kaiming/xavier/normal over ~4 B fp32 parameters, single-threaded, ~43 s per job) and then
   overwrites all of it from the safetensors file. We turn the init functions into no-ops while the models are
   constructed. Any parameter the checkpoint does not cover would be left uninitialised, so we check the missing
   keys and fall back to the original loader for that model if there are any.

2. fast_seam_padding: o_voxel.postprocess.to_glb fills the empty ~45 % of the 4096^2 UV atlas with
   cv2.INPAINT_TELEA (4 calls, ~14 s single-threaded; the atlas has thousands of small charts, so the fill front is
   huge). The padding only exists so that bilinear filtering at chart edges does not pull in black; what it
   contains a few texels away from a chart is never sampled. We pad with the nearest valid texel instead (a
   distance transform, ~0.4 s per map) - the standard bake-margin "extend" behaviour. Valid texels are untouched.
   STUDIO_INPAINT_BAND=<px> additionally runs TELEA inside a band of that width around the charts (upstream's
   diffusion look at the seam; ~1 s per map per pixel of band on this atlas).
"""
from __future__ import annotations

import os
import time

_NOOP_INITS = ("uniform_", "normal_", "trunc_normal_", "kaiming_uniform_", "kaiming_normal_",
               "xavier_uniform_", "xavier_normal_", "orthogonal_")


def _drop_page_cache(path: str) -> None:
    """Evict a checkpoint file from the page cache once its tensors are in process memory. The 23 GB of
    checkpoints otherwise stay cached in the 48 GB Docker VM and push the resident weights (and ComfyUI) into
    swap, which made the first job after a session start ~2 minutes slower than the following ones."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        pass


def install_skip_random_init(log=print) -> None:
    import json

    import torch.nn.init as init
    from safetensors.torch import load_file

    import pixal3d.models as models

    original = models.from_pretrained

    def _resolve(path: str):
        if os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors"):
            return f"{path}.json", f"{path}.safetensors"
        from huggingface_hub import hf_hub_download
        parts = path.split("/")
        repo_id = f"{parts[0]}/{parts[1]}"
        name = "/".join(parts[2:])
        return hf_hub_download(repo_id, f"{name}.json"), hf_hub_download(repo_id, f"{name}.safetensors")

    def fast_from_pretrained(path: str, **kwargs):
        # Any failure here must fall back to the upstream loader itself: upstream's pipeline loader retries a
        # failed local load with the Hub name, which cannot work offline and hides the real error.
        t0 = time.time()
        try:
            config_file, model_file = _resolve(path)
            with open(config_file, "r") as f:
                config = json.load(f)
            saved = {n: getattr(init, n) for n in _NOOP_INITS}
            try:
                for n in _NOOP_INITS:
                    setattr(init, n, lambda tensor, *a, **k: tensor)
                model = models.__getattr__(config["name"])(**config["args"], **kwargs)
            finally:
                for n, fn in saved.items():
                    setattr(init, n, fn)
            state = load_file(model_file)
            missing, unexpected = model.load_state_dict(state, strict=False)
            n_tensors = len(state)
            del state
            _drop_page_cache(model_file)
        except Exception as e:  # noqa: BLE001
            log(f"[fastload] {os.path.basename(path)}: fast path failed ({type(e).__name__}: {e}); using upstream loader")
            return original(path, **kwargs)
        if missing:
            log(f"[fastload] {os.path.basename(path)}: checkpoint lacks {len(missing)} keys "
                f"(e.g. {missing[:3]}); reloading with upstream init")
            del model
            return original(path, **kwargs)
        log(f"[fastload] {os.path.basename(path)}: {n_tensors} tensors in {time.time() - t0:.1f}s"
            + (f", {len(unexpected)} unexpected keys ignored" if unexpected else ""))
        return model

    models.from_pretrained = fast_from_pretrained
    # pipelines/base.py imported the module via `from .. import models` and calls models.from_pretrained -> patched.


def install_fast_seam_padding(band_px: int = 0, validate: bool = False, log=print) -> None:
    import cv2
    import numpy as np

    import o_voxel.postprocess as post

    def pad(img, mask_inv, radius, flags):
        t0 = time.time()
        empty = mask_inv.astype(bool)
        if not empty.any():
            return img.copy()
        # nearest valid texel for every empty texel (labels number the zero pixels of mask_inv = the valid texels)
        dist, labels = cv2.distanceTransformWithLabels(mask_inv.astype(np.uint8), cv2.DIST_L2, 3,
                                                       labelType=cv2.DIST_LABEL_PIXEL)
        flat_idx = np.arange(mask_inv.size, dtype=np.int64).reshape(mask_inv.shape)
        lut = np.zeros(int(labels.max()) + 1, dtype=np.int64)
        lut[labels[~empty]] = flat_idx[~empty]
        src = lut[labels[empty]]
        # cv2.inpaint returns (H, W) for a single-channel (H, W, 1) input; upstream relies on that ([..., None])
        img = img[..., 0] if img.ndim == 3 and img.shape[2] == 1 else img
        out = img.copy()
        flat_out = out.reshape(-1, *img.shape[2:]) if img.ndim == 3 else out.reshape(-1)
        flat_out[flat_idx[empty]] = flat_out[src]
        band = None
        if band_px > 0:
            band = (empty & (dist <= band_px)).astype(np.uint8)
            out = cv2.inpaint(out, band, radius, flags)
        if validate:
            ref = cv2.inpaint(img, mask_inv, radius, flags)
            d = np.abs(ref.astype(np.int16) - out.astype(np.int16))
            edge = empty & (dist <= 1)
            log(f"[fastload] padding validate vs TELEA: valid texels max diff {int(d[~empty].max())}, "
                f"1px-ring mean diff {float(d[edge].mean()):.2f} max {int(d[edge].max())}, "
                f"empty {empty.mean():.1%} of atlas")
        log(f"[fastload] seam padding {img.shape[0]}px x{img.shape[2] if img.ndim == 3 else 1}"
            f"{f' +TELEA band {band_px}px' if band is not None else ''}: {time.time() - t0:.1f}s")
        return out

    class _Cv2Proxy:
        """cv2 stand-in for o_voxel.postprocess: everything passes through except inpaint."""

        def __getattr__(self, name):
            return getattr(cv2, name)

        inpaint = staticmethod(pad)

    post.cv2 = _Cv2Proxy()
