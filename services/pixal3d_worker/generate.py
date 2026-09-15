"""Pixal3D stage: reference image -> high-quality textured master GLB.

Runs as a one-shot subprocess (CUDA memory is released on exit). Reuses the upstream single-image path
(`/opt/pixal3d/inference.py`: init_pipeline, preprocess_image, MoGe-2 camera estimation, pipeline.run) and
replaces only the hardcoded export call with a configurable adapter (o_voxel.postprocess.to_glb + PNG textures).

Phases are printed as "[phase] <name>" so the orchestrator can tell whether an OOM hit generation or export.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

PIXAL3D_REPO = "TencentARC/Pixal3D"
PIXAL3D_REVISION = "b0cb2e1b794cab9aa0ac38a95d794a4d9337437f"
LOCAL_PIPELINE_DIR = Path(os.environ.get("STUDIO_PIXAL3D_LOCAL", "/models/pixal3d-local"))


def phase(name: str):
    print(f"[phase] {name}", flush=True)


def _rss_peak_mb() -> int | None:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


def ensure_local_pipeline_dir(rembg_model: str, multiview: bool = False) -> str:
    """Pixal3D's pipeline.json points the matting model at gated briaai/RMBG-2.0. We materialise a local pipeline
    directory (config override + symlinked checkpoints) so the pipeline loads with the configured rembg model.
    This is a documented adapter, not a fork of upstream code."""
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(PIXAL3D_REPO, revision=PIXAL3D_REVISION,
                                  allow_patterns=["pipeline.json", "pipeline_mv.json", "ckpts/*"]))
    LOCAL_PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    ck = LOCAL_PIPELINE_DIR / "ckpts"
    if ck.is_symlink() or ck.exists():
        if ck.is_symlink() and os.readlink(ck) != str(snap / "ckpts"):
            ck.unlink()
            ck.symlink_to(snap / "ckpts")
    else:
        ck.symlink_to(snap / "ckpts")
    for cfg_name in ("pipeline.json", "pipeline_mv.json"):
        src = snap / cfg_name
        if not src.exists():
            continue
        cfg = json.loads(src.read_text(encoding="utf-8"))
        cfg["args"]["rembg_model"]["args"]["model_name"] = rembg_model
        (LOCAL_PIPELINE_DIR / cfg_name).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return str(LOCAL_PIPELINE_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    a = ap.parse_args()
    req = json.loads(Path(a.request).read_text(encoding="utf-8"))
    out_dir = Path(req["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # Attention backend must be chosen before pixal3d modules import (they read ATTN_BACKEND at import time).
    os.environ["ATTN_BACKEND"] = req.get("attn_backend", "flash_attn")
    sys.path.insert(0, "/opt/pixal3d")
    t_start = time.time()
    timings, vram, warnings = {}, {}, []

    import numpy as np
    import torch
    from PIL import Image

    def vram_snapshot(tag):
        vram[tag] = {"max_allocated_mib": int(torch.cuda.max_memory_allocated() // 2**20),
                     "max_reserved_mib": int(torch.cuda.max_memory_reserved() // 2**20)}
        torch.cuda.reset_peak_memory_stats()

    phase("load")
    t0 = time.time()
    import o_voxel  # noqa: E402
    if os.environ.get("STUDIO_FASTLOAD", "1") != "0":
        # output-neutral speedups for the upstream loaders/export (see fastload.py); STUDIO_FASTLOAD=0 disables
        from pixal3d_worker import fastload  # noqa: E402
        fastload.install_skip_random_init()
        fastload.install_fast_seam_padding(band_px=int(os.environ.get("STUDIO_INPAINT_BAND", "0")),
                                           validate=os.environ.get("STUDIO_FASTLOAD_VALIDATE") == "1")
    if req.get("mode") == "multiview":
        from pixal3d_worker.multiview import run_multiview  # noqa: E402
        result = run_multiview(req, out_dir, timings, vram, warnings, vram_snapshot)
        _write_output(out_dir, result, timings, vram, warnings, t_start)
        return
    import inference as up  # upstream single-view inference module (functions; __main__ guarded)

    model_path = ensure_local_pipeline_dir(req.get("rembg_model", "briaai/RMBG-2.0"))
    pipeline = up.init_pipeline(model_path, low_vram=bool(req["low_vram"]))
    timings["load_s"] = round(time.time() - t0, 2)
    vram_snapshot("load")

    phase("preprocess")
    t0 = time.time()
    img = Image.open(req["image_path"])
    image_preprocessed = pipeline.preprocess_image(img)
    pre_path = out_dir / "preprocessed.png"
    image_preprocessed.save(pre_path)
    timings["preprocess_s"] = round(time.time() - t0, 2)

    phase("camera")
    t0 = time.time()
    moge = up.load_moge_model(device="cuda")
    camera_params = up.get_camera_params_wild_moge(str(pre_path), moge, device="cuda", mesh_scale=1.0,
                                                   extend_pixel=0, image_resolution=512)
    moge.cpu()
    del moge
    torch.cuda.empty_cache()
    timings["camera_s"] = round(time.time() - t0, 2)
    print(f"[camera] {camera_params}", flush=True)
    vram_snapshot("preprocess_camera")

    phase("generate")
    t0 = time.time()
    seed = int(req["seed"])
    torch.manual_seed(seed)
    resolution = int(req["resolution"])
    pipeline_type = f"{resolution}_cascade"
    sampler = req.get("sampler") or {}
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        image_preprocessed, camera_params=camera_params, seed=seed,
        sparse_structure_sampler_params=sampler.get("ss", {}),
        shape_slat_sampler_params=sampler.get("shape", {}),
        tex_slat_sampler_params=sampler.get("tex", {}),
        preprocess_image=False, return_latent=True, pipeline_type=pipeline_type,
        max_num_tokens=int(req.get("max_num_tokens", 49152)),
    )
    torch.cuda.synchronize()
    timings["generate_s"] = round(time.time() - t0, 2)
    vram_snapshot("generate")
    mesh = mesh_list[0]
    mesh_stats = {"raw_vertices": int(mesh.vertices.shape[0]), "raw_faces": int(mesh.faces.shape[0]),
                  "grid_resolution": int(res), "requested_resolution": resolution}
    if int(res) != resolution:
        warnings.append(f"Pixal3D reduced the reconstruction resolution from {resolution} to {res} (token limit)")
    print(f"[mesh] {mesh_stats}", flush=True)
    del shape_slat, tex_slat
    torch.cuda.empty_cache()

    phase("export")
    t0 = time.time()
    exp = req["export"]
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout, grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=int(exp["decimation_target"]), texture_size=int(exp["texture_size"]),
        remesh=bool(exp.get("remesh", True)), remesh_band=exp.get("remesh_band", 1), remesh_project=exp.get("remesh_project", 0),
        use_tqdm=False, verbose=True,
    )
    # Same orientation fix as upstream inference.py (Z-up voxel grid -> glTF Y-up, object upright).
    rot = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    glb.apply_transform(rot)
    master_path = out_dir / "master.glb"
    glb.export(str(master_path), extension_webp=False)  # PNG textures: portable, no EXT_texture_webp
    torch.cuda.synchronize()
    timings["export_s"] = round(time.time() - t0, 2)
    vram_snapshot("export")
    mesh_stats.update({"master_vertices": int(len(glb.vertices)), "master_faces": int(len(glb.faces)),
                       "master_bounds": [[float(x) for x in glb.bounds[0]], [float(x) for x in glb.bounds[1]]]})
    result = {"master_glb": "master.glb", "preprocessed_png": "preprocessed.png", "camera_params": camera_params,
              "mesh_stats": mesh_stats,
              "effective": {"resolution": resolution, "grid_resolution": int(res), "low_vram": bool(req["low_vram"]),
                            "max_num_tokens": int(req.get("max_num_tokens", 49152)), "attn_backend": os.environ["ATTN_BACKEND"],
                            "seed": seed, "sampler_overrides": sampler, "export": exp, "rembg_model": req.get("rembg_model"),
                            "pipeline_dir": model_path}}
    _write_output(out_dir, result, timings, vram, warnings, t_start)


def _write_output(out_dir: Path, result: dict, timings: dict, vram: dict, warnings: list, t_start: float):
    timings["total_s"] = round(time.time() - t_start, 2)
    result["warnings"] = warnings
    result["stats"] = {"timings_s": timings, "vram": vram, "peak_rss_mb": _rss_peak_mb()}
    tmp = out_dir / "output.json.tmp"
    tmp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, out_dir / "output.json")
    print(f"[done] {json.dumps(result['stats'])}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
