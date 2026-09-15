"""FastAPI service: job submission, status, artifacts, cancel/retry, queue, library, settings, events; serves the web portal."""
from __future__ import annotations

import asyncio
import io
import json
import os
import secrets
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Literal, Optional

import uvicorn
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .db import JobStore
from .models import AssetFromCandidateRequest, ImageJobRequest, JobCreated, JobPatch, JobRequest, SettingsPatch
from .presets import load_presets, resolve_settings
from .runner_client import Runner

app = FastAPI(title="asset-studio", version=config.VERSION,
              description="Local text-to-3D asset generation service (Qwen-Image-2512 -> Pixal3D -> Blender).")
_store: JobStore | None = None
WEB_DIST = Path(os.environ.get("STUDIO_WEB_DIST", "/app/web/dist"))


def store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore()
    return _store


async def auth(request: Request):
    if not config.API_TOKEN:
        return
    hdr = request.headers.get("authorization", "")
    tok = hdr[7:] if hdr.lower().startswith("bearer ") else ""
    if not tok:
        tok = request.query_params.get("token", "")  # for <img>/<model-viewer> URLs and EventSource
    if not secrets.compare_digest(tok, config.API_TOKEN):
        raise HTTPException(401, "missing or invalid bearer token")


# ----------------------------------------------------------------------------------------------- health / capabilities

@app.get("/health")
def health():
    runners = {k: Runner(k).health() for k in ("image", "pixal3d", "blender")}
    ok = all(r.get("ok") for r in runners.values())
    try:
        store().con.execute("SELECT 1")
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False
    running = store().list_jobs(limit=1, status="running", archived=None)
    queued = store().con.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0]
    held = store().con.execute("SELECT COUNT(*) FROM jobs WHERE status='held'").fetchone()[0]
    return {"ok": ok and db_ok, "version": config.VERSION, "db": db_ok, "runners": runners,
            "queue": {"running": running[0]["id"] if running else None, "queued": queued, "held": held}, "time": time.time()}


@app.get("/capabilities")
def capabilities():
    p = load_presets()
    gpu = Runner("pixal3d").gpu()
    models_root = Path("/models/hf/hub")
    mv_ready = False
    if models_root.exists():
        for snap in (models_root / "models--TencentARC--Pixal3D" / "snapshots").glob("*/ckpts"):
            mv_ready = mv_ready or any(snap.glob("*_mv.safetensors"))
    deps = {}
    dm = config.MANIFESTS_DIR / "dependency-manifest.json"
    if dm.exists():
        deps = json.loads(dm.read_text(encoding="utf-8"))
    return {
        "version": config.VERSION,
        "pipeline": ["prompt->template", "Qwen-Image-2512 reference (diffusers, local)", "Pixal3D preprocessing + MoGe-2 camera",
                     "Pixal3D geometry+PBR (TRELLIS.2 backbone)", "master GLB (PNG textures)",
                     "Blender: meshoptimizer reduction / bake / LODs / collision / previews", "validation + manifest"],
        "styles": {k: {"label": v.get("label"), "description": v.get("description"), "best_for": v.get("best_for"),
                       "style_clause": " ".join((v.get("style_clause") or "").split()), "background": v.get("background"),
                       "negative_extra": v.get("negative_extra"), "default_materials": v.get("default_materials"),
                       "default_palette": v.get("default_palette"),
                       "optimize_defaults": v.get("optimize_defaults")} for k, v in p["styles"].items()},
        "image_models": _image_models_with_availability(p["image_models"]),
        "quality": {k: {"reference": v["reference"], "pixal3d": v["pixal3d"], "master": v["master"], "fallback": v.get("fallback")}
                    for k, v in p["quality"].items()},
        "request_schema": JobRequest.model_json_schema(),
        "image_job_schema": ImageJobRequest.model_json_schema(),
        "asset_from_candidate_schema": AssetFromCandidateRequest.model_json_schema(),
        "limits": {"target_triangles": [200, 2_000_000], "texture_size": [256, 512, 1024, 2048, 4096], "reference_candidates": [1, 8],
                   "lod_fractions_max": 4},
        "features": {"text_to_asset": True, "image_variations": True, "reference_image_input": True, "multiview_input": True,
                     "multiview_checkpoints_cached": mv_ready, "automatic_novel_view_generation": False,
                     "vision_evaluator": bool(os.environ.get("STUDIO_VISION_EVALUATOR_URL")), "web_portal": WEB_DIST.exists()},
        "gpu": gpu, "gpu_uuid": config.GPU_UUID,
        "dependencies": deps.get("summary", deps),
        "auth_required": bool(config.API_TOKEN),
    }


# ----------------------------------------------------------------------------------------------- jobs

@app.post("/v1/jobs", response_model=JobCreated, dependencies=[Depends(auth)])
def create_job(req: JobRequest):
    """One-shot: prompt -> auto-selected reference -> 3D asset (the original agent path)."""
    data = req.model_dump()
    if not data.get("model"):
        data["model"] = store().get_settings().get("default_image_model", "flux2-klein-9b")
    try:
        settings = resolve_settings(data)
    except ValueError as e:
        raise HTTPException(422, str(e))
    job_id = store().create(data, settings, kind="asset", title=req.title)
    return JobCreated(job_id=job_id, status="queued", status_url=f"/v1/jobs/{job_id}", artifacts_url=f"/v1/jobs/{job_id}/artifacts")


@app.post("/v1/image-jobs", response_model=JobCreated, dependencies=[Depends(auth)])
def create_image_job(req: ImageJobRequest):
    """Ideation: generate `variations` reference images of one idea; nothing is turned into 3D until you pick one."""
    data = req.model_dump()
    data["quality"] = "balanced"
    if not data.get("model"):
        data["model"] = store().get_settings().get("default_image_model", "flux2-klein-9b")
    try:
        settings = resolve_settings(data)
    except ValueError as e:
        raise HTTPException(422, str(e))
    job_id = store().create(data, settings, kind="image", title=req.title)
    return JobCreated(job_id=job_id, status="queued", status_url=f"/v1/jobs/{job_id}", artifacts_url=f"/v1/jobs/{job_id}/artifacts")


@app.post("/v1/asset-jobs", response_model=JobCreated, dependencies=[Depends(auth)])
def create_asset_job(req: AssetFromCandidateRequest):
    """Turn one variation of an image job into a 3D asset. `hold=true` parks it in the review queue."""
    parent = _get_job(req.image_job_id)
    if parent["kind"] != "image":
        raise HTTPException(422, "image_job_id is not an image job")
    cand_dir = (config.JOBS_DIR / parent["id"] / "stages" / "reference" / "candidates").resolve()
    cand = (cand_dir / req.candidate).resolve()
    if cand_dir not in cand.parents or not cand.is_file():
        raise HTTPException(404, f"candidate {req.candidate} not found (is the image job complete?)")
    preq = parent["request"]
    data = {
        "prompt": preq["prompt"], "style": preq.get("style", "mobile_factory"), "custom_style": preq.get("custom_style"), "quality": req.quality,
        "seed": parent["settings"].get("seed"), "materials": preq.get("materials"), "palette": preq.get("palette"),
        "negative_extra": preq.get("negative_extra"),
        "height_m": req.height_m if req.height_m is not None else preq.get("height_m"), "width_m": req.width_m, "depth_m": req.depth_m,
        "target_triangles": req.target_triangles, "texture_size": req.texture_size, "generate_lods": req.generate_lods,
        "lod_fractions": req.lod_fractions, "generate_collision": req.generate_collision, "collision_triangles": req.collision_triangles,
        "allow_quality_fallback": req.allow_quality_fallback, "image_job_id": parent["id"], "candidate": req.candidate,
        "render_previews": True, "preview_size": 512,
    }
    try:
        settings = resolve_settings(data)
    except ValueError as e:
        raise HTTPException(422, str(e))
    title = req.title or f"{(parent.get('title') or preq['prompt'])[:60]} · {req.candidate[:-4]}"
    job_id = store().create(data, settings, kind="asset", parent_id=parent["id"], candidate=req.candidate, title=title, held=req.hold)
    return JobCreated(job_id=job_id, status="held" if req.hold else "queued", status_url=f"/v1/jobs/{job_id}",
                      artifacts_url=f"/v1/jobs/{job_id}/artifacts")


@app.get("/v1/jobs", dependencies=[Depends(auth)])
def list_jobs(limit: int = 50, status: str | None = None, kind: str | None = None):
    limit = max(1, min(limit, 500))
    return [_public(j) for j in store().list_jobs(limit=limit, status=status, kind=kind, archived=None)]


_avail_cache: dict = {"t": 0.0, "v": {}}


def _image_models_with_availability(models: dict) -> list[dict]:
    """Registry entries plus a live availability check done inside the image worker (files/caches are mounted there)."""
    now = time.time()
    if now - _avail_cache["t"] > 60:
        try:
            import subprocess  # noqa: F401
            r = Runner("image").http.post("/exec", json={"cmd": ["python", "-m", "image_worker.generate", "--list", "/app/presets/image_models.yaml"]}, timeout=60)
            out = r.json().get("stdout", "[]") if r.status_code == 200 else "[]"
            _avail_cache["v"] = {e["id"]: e for e in json.loads(out.strip().splitlines()[-1])} if out.strip() else {}
        except Exception:  # noqa: BLE001
            _avail_cache["v"] = {}
        _avail_cache["t"] = now
    out = []
    for mid, m in models.items():
        a = _avail_cache["v"].get(mid)
        out.append({"id": mid, "label": m.get("label"), "tier": m.get("tier"), "family": m.get("family"), "est_s": m.get("est_s"),
                    "description": m.get("description"), "params": {k: v for k, v in m.get("params", {}).items() if k in ("steps", "width", "height")},
                    "available": (a["available"] if a else None), "reason": (a.get("reason") if a else "not checked")})
    return out


def _thumb(j: dict) -> str | None:
    art = config.JOBS_DIR / j["id"] / "artifacts" / "thumbs"
    for name in ("card.jpg", "reference.jpg", "cand_00.jpg"):
        if (art / name).exists():
            return f"/v1/jobs/{j['id']}/artifacts/thumbs/{name}"
    if (config.JOBS_DIR / j["id"] / "artifacts" / "reference.png").exists():
        return f"/v1/jobs/{j['id']}/artifacts/reference.png"
    return None


def _public(j: dict, events=None) -> dict:
    out = {k: v for k, v in j.items() if k not in ("request",)}
    req = dict(j["request"])
    req.pop("reference_image_b64", None)
    if req.get("multiview"):
        req["multiview"] = {"num_views": len(req["multiview"].get("frames", [])), "camera_source": req["multiview"].get("camera_source")}
    out["request"] = req
    out["settings"] = {k: v for k, v in (j.get("settings") or {}).items() if not k.startswith("_")}
    out["fallbacks_applied"] = (j.get("settings") or {}).get("_fallbacks_applied", [])
    out["thumb"] = _thumb(j)
    if events is not None:
        out["events"] = events
    return out


def _get_job(job_id: str) -> dict:
    if not _safe_id(job_id):
        raise HTTPException(400, "invalid job id")
    j = store().get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return j


def _safe_id(job_id: str) -> bool:
    return bool(job_id) and len(job_id) < 64 and all(c.isalnum() or c == "-" for c in job_id)


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(auth)])
def get_job(job_id: str, events: int = 20):
    j = _get_job(job_id)
    ev = store().events(job_id, limit=max(0, min(events, 500))) if events else None
    out = _public(j, ev)
    out["artifacts_url"] = f"/v1/jobs/{job_id}/artifacts"
    mp = config.JOBS_DIR / job_id / "manifest.json"
    out["manifest_url"] = f"/v1/jobs/{job_id}/artifacts/manifest.json" if mp.exists() else None
    if j["kind"] == "image":
        out["candidates"] = _candidates(j)
        out["children"] = [_public(c) for c in store().children(job_id)]
    elif j.get("parent_id"):
        pj = store().get(j["parent_id"])
        out["parent"] = _public(pj) if pj else None
    if mp.exists() and j["kind"] == "asset":
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
            out["asset"] = m.get("asset")
            out["summary"] = _asset_summary(m)
        except json.JSONDecodeError:
            pass
    return out


_cand_cache: dict = {}


def _candidates(j: dict) -> list[dict]:
    """Candidates of an image job, including partial results while it is still running. Completed jobs are cached
    by output.json mtime so library listings do not re-read every job's JSON."""
    d = config.JOBS_DIR / j["id"] / "stages" / "reference"
    out = []
    op = d / "output.json"
    if op.exists():
        try:
            mt = op.stat().st_mtime
            hit = _cand_cache.get(j["id"])
            if hit and hit[0] == mt:
                return hit[1]
            o = json.loads(op.read_text(encoding="utf-8"))
            for c in o.get("candidates", []):
                name = Path(c["file"]).name
                out.append({"file": name, "seed": c["seed"], "time_s": c["time_s"], "score": c["metrics"].get("score"),
                            "reasons": c["metrics"].get("reasons") or c["metrics"].get("reject"),
                            "url": f"/v1/jobs/{j['id']}/artifacts/reference_candidates/{name}",
                            "thumb": f"/v1/jobs/{j['id']}/artifacts/thumbs/{Path(name).stem}.jpg"})
            if len(_cand_cache) > 2000:
                _cand_cache.clear()
            _cand_cache[j["id"]] = (mt, out)
            return out
        except json.JSONDecodeError:
            pass
    cd = d / "candidates"
    if cd.exists():  # in progress: images exist before output.json
        for p in sorted(cd.glob("cand_*.png")):
            out.append({"file": p.name, "seed": None, "time_s": None, "score": None, "reasons": None,
                        "url": f"/v1/jobs/{j['id']}/stage-files/reference/candidates/{p.name}", "thumb": None, "partial": True})
    return out


def _asset_summary(m: dict) -> dict:
    st = m.get("stages") or {}
    bl = (st.get("blender") or {})
    a = (bl.get("stats") or {}).get("asset") or {}
    px = st.get("pixal3d") or {}
    v = st.get("validation") or {}
    return {
        "triangles": a.get("triangles"), "master_triangles": (px.get("mesh_stats") or {}).get("master_faces"),
        "method": a.get("method"), "reduction": a.get("reduction"), "maps": a.get("maps"),
        "lods": [{"file": l["file"], "triangles": l["triangles"]} for l in (bl.get("outputs") or {}).get("lods", [])],
        "collision_triangles": ((bl.get("stats") or {}).get("collision") or {}).get("triangles"),
        "resolution": (px.get("effective") or {}).get("resolution"),
        "timings_s": m.get("timings_s"), "validation_ok": all(r.get("ok") for r in v.values()) if v else None,
        "vram_mib": {"reference": ((m.get("resources") or {}).get("reference") or {}).get("peak_gpu_used_mib"),
                     "pixal3d": ((m.get("resources") or {}).get("pixal3d") or {}).get("peak_gpu_used_mib")},
        "warnings": m.get("warnings", []),
    }


@app.patch("/v1/jobs/{job_id}", dependencies=[Depends(auth)])
def patch_job(job_id: str, body: JobPatch):
    _get_job(job_id)
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if "favorite" in fields:
        fields["favorite"] = 1 if fields["favorite"] else 0
    if "archived" in fields:
        fields["archived"] = 1 if fields["archived"] else 0
    if fields:
        store().update(job_id, **fields)
    return _public(store().get(job_id))


@app.delete("/v1/jobs/{job_id}", dependencies=[Depends(auth)])
def delete_job(job_id: str, purge: bool = False):
    j = _get_job(job_id)
    if j["status"] == "running":
        if not (purge and j.get("cancel_requested")):
            raise HTTPException(409, "cancel the job first")
        # cancel is in flight: finish it here so no worker can ever re-claim the row; the running stage stops at its
        # next cancel check (a missing row counts as cancelled) and its process is torn down by the runner
        store().update(job_id, status="cancelled", stage="cancelled", finished_at=time.time())
    if purge:
        d = (config.JOBS_DIR / job_id).resolve()
        if config.JOBS_DIR.resolve() in d.parents and d.exists():
            shutil.rmtree(d, ignore_errors=True)
        store().delete_job(job_id)
        return {"job_id": job_id, "result": "purged"}
    store().update(job_id, archived=1)
    return {"job_id": job_id, "result": "archived"}


@app.get("/v1/jobs/{job_id}/artifacts", dependencies=[Depends(auth)])
def list_artifacts(job_id: str):
    j = _get_job(job_id)
    art = config.JOBS_DIR / job_id / "artifacts"
    items = []
    if art.exists():
        for p in sorted(art.rglob("*")):
            if p.is_file():
                rel = p.relative_to(art).as_posix()
                items.append({"name": rel, "bytes": p.stat().st_size, "url": f"/v1/jobs/{job_id}/artifacts/{rel}"})
    return {"job_id": job_id, "status": j["status"], "artifacts": items}


def _media(target: Path) -> str:
    return {".glb": "model/gltf-binary", ".png": "image/png", ".jpg": "image/jpeg", ".json": "application/json"}.get(
        target.suffix, "application/octet-stream")


@app.get("/v1/jobs/{job_id}/artifacts/{name:path}", dependencies=[Depends(auth)])
def get_artifact(job_id: str, name: str):
    _get_job(job_id)
    art = (config.JOBS_DIR / job_id / "artifacts").resolve()
    target = (art / name).resolve()
    if art not in target.parents or not target.is_file():
        raise HTTPException(404, "artifact not found")
    return FileResponse(str(target), media_type=_media(target), filename=target.name)


@app.get("/v1/jobs/{job_id}/stage-files/{name:path}", dependencies=[Depends(auth)])
def get_stage_file(job_id: str, name: str):
    """Read-only access to stage outputs (used for in-progress candidate previews)."""
    _get_job(job_id)
    base = (config.JOBS_DIR / job_id / "stages").resolve()
    target = (base / name).resolve()
    if base not in target.parents or not target.is_file() or target.suffix not in (".png", ".jpg", ".json"):
        raise HTTPException(404, "not found")
    return FileResponse(str(target), media_type=_media(target))


@app.get("/v1/jobs/{job_id}/download.zip", dependencies=[Depends(auth)])
def download_zip(job_id: str, include_master: bool = False):
    j = _get_job(job_id)
    art = config.JOBS_DIR / job_id / "artifacts"
    if not art.exists():
        raise HTTPException(404, "no artifacts")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(art.rglob("*")):
            if p.is_file() and (include_master or p.name != "master.glb"):
                z.write(p, p.relative_to(art).as_posix())
    buf.seek(0)
    name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (j.get("title") or job_id))[:60]
    return StreamingResponse(buf, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}.zip"'})


@app.get("/v1/jobs/{job_id}/logs/{stage}", dependencies=[Depends(auth)])
def get_log(job_id: str, stage: str, tail: int = 20000):
    _get_job(job_id)
    if stage not in ("reference", "pixal3d", "blender"):
        raise HTTPException(404, "unknown stage")
    p = config.JOBS_DIR / job_id / "stages" / stage / "log.txt"
    if not p.exists():
        return PlainTextResponse("")
    data = p.read_bytes()[-max(0, min(tail, 5_000_000)):]
    return PlainTextResponse(data.decode("utf-8", "replace"))


@app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(auth)])
def cancel_job(job_id: str):
    _get_job(job_id)
    return {"job_id": job_id, "result": store().request_cancel(job_id)}


@app.post("/v1/jobs/{job_id}/release", dependencies=[Depends(auth)])
def release_job(job_id: str):
    """held -> queued (start processing)."""
    _get_job(job_id)
    return {"job_id": job_id, "result": store().set_status_simple(job_id, ("held",), "queued")}


@app.post("/v1/jobs/{job_id}/hold", dependencies=[Depends(auth)])
def hold_job(job_id: str):
    """queued -> held (take it out of the worker's queue without cancelling)."""
    _get_job(job_id)
    return {"job_id": job_id, "result": store().set_status_simple(job_id, ("queued",), "held")}


class RetryRequest(BaseModel):
    from_stage: Optional[Literal["reference", "pixal3d", "blender"]] = Field(
        None, description="Re-run from this stage (its outputs and later ones are discarded); required to reprocess a completed job")
    optimize: Optional[dict] = Field(None, description="Override optimisation settings for the re-run, e.g. {target_triangles, texture_size, lod_fractions, generate_lods, generate_collision, collision_triangles}")


@app.post("/v1/jobs/{job_id}/retry", dependencies=[Depends(auth)])
def retry_job(job_id: str, body: RetryRequest | None = None):
    """Requeue a job. Lifecycle contract (explicit for agent callers):
      * failed / cancelled  -> requeued; stages that completed earlier are reused (resume). Returns 200 {"result": "queued"}.
      * completed           -> only with `from_stage` (and/or `optimize`, which implies from_stage=blender): that stage and the
                               later ones are discarded and re-run. Without it, 409: nothing to retry.
      * running / queued / held -> 409, nothing is changed (cancel first, or wait). A retry never restarts a job in place and
                               never creates a second attempt of a job that is still alive.
    Example: {"from_stage": "blender", "optimize": {"target_triangles": 20000}} re-optimises the same master."""
    j = _get_job(job_id)
    body = body or RetryRequest()
    if j["status"] in ("running", "queued", "held"):
        raise HTTPException(409, f"job is {j['status']}; cancel it first or wait for it to finish (retry only applies to "
                                 "failed, cancelled or completed jobs)")
    if body.optimize and not body.from_stage:
        body.from_stage = "blender"
    if j["status"] == "completed" and not body.from_stage:
        raise HTTPException(409, "job already completed; pass from_stage (reference|pixal3d|blender) to reprocess it")
    settings = None
    if body.optimize:
        allowed = {"target_triangles", "texture_size", "lod_fractions", "generate_lods", "generate_collision", "collision_triangles",
                   "height_m", "width_m", "depth_m", "render_previews", "preview_size"}
        bad = set(body.optimize) - allowed
        if bad:
            raise HTTPException(422, f"unknown optimize keys: {sorted(bad)}")
        settings = dict(j["settings"])
        settings["optimize"] = {**settings["optimize"], **body.optimize}
    if body.from_stage:
        order = ["reference", "pixal3d", "blender"]
        for st in order[order.index(body.from_stage):]:
            rp = config.JOBS_DIR / job_id / "stages" / st / "result.json"
            if rp.exists():
                rp.unlink()
    r = store().retry(job_id, allow_completed=bool(body.from_stage), settings=settings)
    return {"job_id": job_id, "result": r, "from_stage": body.from_stage, "attempt": (j.get("attempt") or 0) + 1}


# ----------------------------------------------------------------------------------------------- queue / library / settings

@app.get("/v1/queue", dependencies=[Depends(auth)])
def get_queue():
    items = [_public(j) for j in store().queue()]
    pos = 0
    for it in items:
        if it["status"] == "queued":
            pos += 1
            it["position"] = pos
    gpu = Runner("pixal3d").gpu()
    return {"items": items, "gpu": gpu, "settings": store().get_settings()}


@app.get("/v1/library", dependencies=[Depends(auth)])
def library(kind: str | None = None, status: str | None = None, q: str | None = None, favorite: bool | None = None,
            archived: bool = False, limit: int = 60, offset: int = 0):
    limit = max(1, min(limit, 500))
    jobs = store().list_jobs(limit=limit, status=status, kind=kind, q=q, favorite=favorite, archived=archived, offset=offset)
    counts = store().children_counts([j["id"] for j in jobs if j["kind"] == "image"])
    cards = []
    for j in jobs:
        c = _public(j)
        if j["kind"] == "image":
            c["children_count"] = counts.get(j["id"], 0)
            c["candidate_count"] = len(_candidates(j)) if j["status"] in ("running", "completed") else 0
        cards.append(c)
    return {"items": cards, "limit": limit, "offset": offset}


@app.get("/v1/settings", dependencies=[Depends(auth)])
def get_settings():
    return store().get_settings()


@app.put("/v1/settings", dependencies=[Depends(auth)])
def put_settings(body: SettingsPatch):
    p = load_presets()
    vals = {k: v for k, v in body.model_dump(exclude_none=True).items()}
    if "style_edits" in vals:  # drop empty entries so a cleared preset really resets
        vals["style_edits"] = {k: {kk: vv for kk, vv in e.items() if vv not in (None, "")} for k, e in vals["style_edits"].items()}
        vals["style_edits"] = {k: e for k, e in vals["style_edits"].items() if e.get("style_clause")}
    if "default_style" in vals and vals["default_style"] not in p["styles"]:
        raise HTTPException(422, "unknown style")
    if "default_image_model" in vals and vals["default_image_model"] not in p["image_models"]:
        raise HTTPException(422, "unknown image model")
    return store().put_settings(vals)


@app.get("/v1/examples", dependencies=[Depends(auth)])
def examples():
    p = config.PRESETS_DIR / "examples.yaml"
    items = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else []
    return {"items": items}


@app.get("/v1/events", dependencies=[Depends(auth)])
async def events(since: int = 0):
    """Server-sent events: every job event row after `since` (id). Clients keep the last id and reconnect."""
    async def gen():
        last = since if since > 0 else store().last_event_id()
        yield f"event: hello\ndata: {json.dumps({'last_id': last})}\n\n"
        idle = 0
        while True:
            rows = store().events_since(last)
            if rows:
                for r in rows:
                    last = r["id"]
                    yield f"id: {r['id']}\nevent: job\ndata: {json.dumps(r)}\n\n"
                idle = 0
            else:
                idle += 1
                if idle % 15 == 0:
                    yield ": keepalive\n\n"
            await asyncio.sleep(1.0)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)


# ----------------------------------------------------------------------------------------------- web portal (SPA)

if (WEB_DIST / "index.html").exists():
    if (WEB_DIST / "static").is_dir():
        app.mount("/static", StaticFiles(directory=str(WEB_DIST / "static")), name="web-static")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith(("v1/", "health", "capabilities", "docs", "openapi.json")):
            raise HTTPException(404)
        target = (WEB_DIST / full_path).resolve() if full_path else None
        if target and WEB_DIST.resolve() in target.parents and target.is_file():
            return FileResponse(str(target))
        return FileResponse(str(WEB_DIST / "index.html"))


def main():
    if config.BIND_HOST not in ("127.0.0.1", "localhost", "::1") and not config.API_TOKEN:
        print("refusing to bind to a non-loopback address without STUDIO_API_TOKEN", file=sys.stderr)
        sys.exit(2)
    # inside compose the container listens on all interfaces; the *published* port is bound to localhost by compose.
    host = "0.0.0.0" if os.environ.get("STUDIO_IN_CONTAINER") else config.BIND_HOST
    uvicorn.run(app, host=host, port=config.PORT, log_level="info")


if __name__ == "__main__":
    main()
