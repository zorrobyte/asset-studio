"""Job pipeline: text -> reference image -> Pixal3D master GLB -> Blender optimisation/previews -> validation -> manifest.

Every stage writes its outputs into /jobs/<id>/stages/<stage>/ and finishes by atomically writing result.json.
A stage whose result.json exists with status "ok" is reused on restart (resume). Stage subprocesses run in their
own worker containers through the stage runners, so CUDA memory is released when each stage process exits.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from . import config
from .presets import load_presets
from .promptbuilder import build_reference_prompt
from .runner_client import Runner, StageCancelled, StageFailed
from .validate import validate_glb

STAGES = ["reference", "pixal3d", "blender", "validate", "package"]


class JobCancelled(Exception):
    pass


class JobFailed(Exception):
    def __init__(self, stage, message, details=None):
        super().__init__(message)
        self.stage = stage
        self.details = details or {}


def atomic_write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Lanes:
    """Which lanes (image / asset) currently have a GPU stage running. Used only to decide what to wait for after an
    out-of-memory error; nothing is gated up front."""

    def __init__(self):
        import threading

        self._lock = threading.Lock()
        self._busy: dict[str, str] = {}

    def enter(self, lane: str, stage: str):
        with self._lock:
            self._busy[lane] = stage

    def leave(self, lane: str):
        with self._lock:
            self._busy.pop(lane, None)

    def other_busy(self, lane: str):
        with self._lock:
            for k, v in self._busy.items():
                if k != lane:
                    return k, v
        return None

    def wait_other_idle(self, lane: str, should_cancel, poll_s: float = 3.0):
        while self.other_busy(lane):
            if should_cancel():
                raise JobCancelled()
            time.sleep(poll_s)


LANES = Lanes()


class JobRun:
    def __init__(self, store, job: dict):
        self.store = store
        self.job = job
        self.id = job["id"]
        self.request = job["request"]
        self.settings = job["settings"]
        # warnings are owned by stages: rebuild from the completed stages' results so reprocessed stages drop stale ones
        self.warnings: list[str] = []
        for st in ("reference", "pixal3d", "blender", "validate"):
            r = self._read_stage_result(job["id"], st)
            if r:
                self.warnings += list(r.get("stage_warnings") or [])
        self._warn_mark = len(self.warnings)
        self.timings: dict = dict(job.get("timings") or {})
        self.dir = config.JOBS_DIR / self.id
        self.stages_dir = self.dir / "stages"
        self.art = self.dir / "artifacts"
        for d in (self.stages_dir, self.art):
            d.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.dir / "request.json", self.request)
        self.runners = {k: Runner(k) for k in ("image", "pixal3d", "blender")}
        self.lane = "image" if job.get("kind") == "image" else "asset"

    # ---- helpers -----------------------------------------------------------
    def log(self, msg: str, level: str = "info"):
        self.store.event(self.id, level, msg)

    def warn(self, msg: str):
        self.warnings.append(msg)
        self.store.update(self.id, warnings=self.warnings)
        self.log(msg, "warn")

    def cancelled(self) -> bool:
        return self.store.is_cancel_requested(self.id)

    def check_cancel(self):
        if self.cancelled():
            raise JobCancelled()

    def set_stage(self, stage: str, progress: float):
        self.store.update(self.id, stage=stage, progress=progress, settings=self.settings, timings=self.timings)

    @staticmethod
    def _read_stage_result(job_id: str, name: str) -> dict | None:
        p = config.JOBS_DIR / job_id / "stages" / name / "result.json"
        try:
            r = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except json.JSONDecodeError:
            return None
        return r if r and r.get("status") == "ok" else None

    def stage_dir(self, name: str) -> Path:
        d = self.stages_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def stage_result(self, name: str) -> dict | None:
        p = self.stages_dir / name / "result.json"
        if p.exists():
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
                if r.get("status") == "ok":
                    return r
            except json.JSONDecodeError:
                return None
        return None

    def finish_stage(self, name: str, result: dict, t0: float):
        result["status"] = "ok"
        result["elapsed_s"] = round(time.time() - t0, 2)
        result["stage_warnings"] = self.warnings[self._warn_mark:]
        self._warn_mark = len(self.warnings)
        (self.stages_dir / name).mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.stages_dir / name / "result.json", result)
        self.timings[name] = result["elapsed_s"]
        self.store.update(self.id, timings=self.timings, settings=self.settings)

    def _fallback(self, stage_key: str, log_path: Path) -> bool:
        """Apply the next OOM fallback for a stage. Returns False if none is permitted."""
        applied = self.settings.setdefault("_fallbacks_applied", [])
        for i, fb in enumerate(self.settings.get("fallback", [])):
            if fb["stage"] != stage_key or i in applied:
                continue
            if fb.get("quality_loss") and not self.settings.get("allow_quality_fallback", True):
                self.warn(f"OOM in {stage_key}: fallback {fb['set']} would reduce quality and allow_quality_fallback=false")
                continue
            target = self.settings[stage_key]
            before = {k: target.get(k) for k in fb["set"]}
            target.setdefault("_requested", {}).update({k: v for k, v in before.items() if k not in target.get("_requested", {})})
            target.update(fb["set"])
            applied.append(i)
            self.warn(f"CUDA out of memory in stage '{stage_key}'; retrying with {fb['set']} (requested {before})"
                      + ("; this reduces quality" if fb.get("quality_loss") else ""))
            return True
        return False

    def _run_gpu_stage(self, runner_name: str, stage: str, cmd: list[str], log_path: Path, fallback_keys: list[str]):
        """Run a GPU stage. No VRAM thresholds: the stage starts right away (other lanes and other apps share the card).
        On CUDA OOM: (1) if the other lane is in a GPU stage, wait for it to finish and retry with the same settings,
        (2) otherwise retry once after a short pause (another app may release memory), (3) then apply the preset's
        fallback ladder (quality-reducing steps only if the job allows them)."""
        r = self.runners[runner_name]
        lane = self.lane
        attempts = 0
        same_settings_retries = 0
        run_file = log_path.with_name("run.json")
        while True:
            attempts += 1
            self.check_cancel()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n===== attempt {attempts} at {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            LANES.enter(lane, stage)
            try:
                return r.run(cmd, str(log_path), should_cancel=self.cancelled, run_file=str(run_file), log=self.log)
            except StageCancelled:
                raise JobCancelled()
            except StageFailed as e:
                if not e.oom:
                    tail = _log_tail(log_path)
                    raise JobFailed(stage, f"{e}", {"log": str(log_path.relative_to(self.dir)), "run": e.run, "log_tail": tail})
                other = LANES.other_busy(lane)
                if other and same_settings_retries < config.MAX_OOM_RETRIES:
                    same_settings_retries += 1
                    self.warn(f"CUDA out of memory in stage '{stage}' while the {other[0]} lane was running its {other[1]} stage; "
                              f"waiting for it to finish, then retrying with the same settings")
                    LANES.wait_other_idle(lane, self.cancelled)
                    continue
                if same_settings_retries == 0:
                    same_settings_retries += 1
                    self.warn(f"CUDA out of memory in stage '{stage}'; retrying once with the same settings in "
                              f"{config.OOM_RETRY_PAUSE_S} s in case another application releases memory")
                    for _ in range(config.OOM_RETRY_PAUSE_S):
                        self.check_cancel()
                        time.sleep(1)
                    continue
                phase = _last_phase(log_path)
                keys = [k for k in fallback_keys if (phase != "export" or k == "master") and (phase == "export" or k != "master")]
                if any(self._fallback(k, log_path) for k in keys or fallback_keys):
                    continue
                raise JobFailed(stage, f"CUDA out of memory in stage '{stage}' and no permitted fallback remains",
                                {"log": str(log_path.relative_to(self.dir)), "run": e.run})
            finally:
                try:
                    run_file.unlink()
                except OSError:
                    pass
                LANES.leave(lane)

    # ---- stages ------------------------------------------------------------
    def run(self):
        self.store.update(self.id, status="running")
        if self.job.get("kind") == "image":
            self.stage_reference(variations_only=True)
            self.stage_package()
            return
        mode = self.settings.get("input_mode", "text")
        if self.job.get("parent_id"):
            self.stage_reference_from_parent()
        elif mode == "text":
            self.stage_reference()
        else:
            self.stage_reference_provided()
        self.stage_pixal3d()
        self.stage_blender()
        self.stage_validate()
        self.stage_package()

    def stage_reference(self, variations_only: bool = False):
        name = "reference"
        if self.stage_result(name):
            self.log("reference stage already complete; reusing")
            return
        t0 = time.time()
        self.set_stage(name, 0.05)
        d = self.stage_dir(name)
        style = self.settings.get("style_def") or load_presets()["styles"][self.settings["style"]]
        built = build_reference_prompt(self.request, style)
        ref = self.settings["reference"]
        req = {
            "prompt": built["prompt"], "negative_prompt": built["negative_prompt"], "template": built["template"],
            "model": "Qwen/Qwen-Image-2512", "model_id": ref.get("model", "qwen-image-2512"), "family": ref.get("family", "qwen"),
            "width": ref["width"], "height": ref["height"], "steps": ref["steps"],
            "true_cfg_scale": ref.get("true_cfg_scale", 1.0), "guidance_scale": ref.get("guidance_scale", 1.0),
            "mode": ref.get("mode", "bf16_split"), "gpu_resident_blocks": ref.get("gpu_resident_blocks", 30),
            "lightning_lora": ref.get("lightning_lora"), "config_repo": ref.get("config_repo"), "config_revision": ref.get("config_revision"),
            "transformer_file": ref.get("transformer_file"), "text_encoder_file": ref.get("text_encoder_file"),
            "seeds": [(self.settings["seed"] + i) % (2**31 - 1) for i in range(ref["candidates"])],
            "out_dir": str(d), "evaluator_url": os.environ.get("STUDIO_VISION_EVALUATOR_URL", ""),
        }
        for k, v in ref.items():  # every registry parameter of the chosen model reaches the worker (distilled, sequential, ...)
            req.setdefault(k, v)
        atomic_write_json(d / "request.json", req)
        log_path = d / "log.txt"
        self.log(f"generating {ref['candidates']} reference candidate(s) with {ref.get('model', 'qwen-image-2512')} ({ref['steps']} steps)")
        # the request may be re-written by fallbacks; the worker re-reads it on each attempt
        run = self._run_gpu_stage("image", name, ["python", "-m", "image_worker.generate", "--request", str(d / "request.json")],
                                  log_path, ["reference"])
        out = json.loads((d / "output.json").read_text(encoding="utf-8"))
        cands = self.art / "reference_candidates"
        cands.mkdir(exist_ok=True)
        thumbs = self.art / "thumbs"
        thumbs.mkdir(exist_ok=True)
        for c in out["candidates"]:
            shutil.copy2(d / c["file"], cands / Path(c["file"]).name)
            make_thumb(d / c["file"], thumbs / (Path(c["file"]).stem + ".jpg"))
        shutil.copy2(d / "selection.json", cands / "selection.json")
        if not variations_only:
            sel = d / out["selected"]
            shutil.copy2(sel, self.art / "reference.png")
            for w in out.get("warnings", []):
                self.warn(w)
        else:
            make_thumb(d / out["selected"], thumbs / "card.jpg")
        res = {"selected": out["selected"], "selection": out["selection"], "stats": out.get("stats", {}), "run": run,
               "prompt": built["prompt"], "negative_prompt": built["negative_prompt"],
               "effective": {k: req[k] for k in ("model_id", "family", "width", "height", "steps", "true_cfg_scale", "guidance_scale", "mode",
                                                  "gpu_resident_blocks", "lightning_lora", "seeds")}}
        self.finish_stage(name, res, t0)

    def stage_reference_from_parent(self):
        name = "reference"
        if self.stage_result(name):
            return
        t0 = time.time()
        d = self.stage_dir(name)
        self.set_stage(name, 0.05)
        parent = self.job["parent_id"]
        cand = self.job.get("candidate") or ""
        src_dir = (config.JOBS_DIR / parent / "stages" / "reference" / "candidates").resolve()
        src = (src_dir / cand).resolve()
        if src_dir not in src.parents or not src.is_file():
            raise JobFailed(name, f"candidate {cand!r} of image job {parent} not found")
        shutil.copy2(src, d / "provided.png")
        shutil.copy2(src, self.art / "reference.png")
        (self.art / "thumbs").mkdir(exist_ok=True)
        make_thumb(src, self.art / "thumbs" / "reference.jpg")
        self.log(f"using variation {cand} of image job {parent} (Qwen stage skipped)")
        self.finish_stage(name, {"selected": "provided.png", "source": "image_job", "parent_id": parent, "candidate": cand}, t0)

    def stage_reference_provided(self):
        name = "reference"
        if self.stage_result(name):
            return
        t0 = time.time()
        d = self.stage_dir(name)
        self.set_stage(name, 0.05)
        if self.request.get("reference_image_b64"):
            raw = base64.b64decode(self.request["reference_image_b64"])
            ext = ".png" if raw[:4] == b"\x89PNG" else ".jpg"
            (d / f"provided{ext}").write_bytes(raw)
            shutil.copy2(d / f"provided{ext}", self.art / f"reference{ext}")
            res = {"selected": f"provided{ext}", "source": "provided_by_caller"}
            self.log("using caller-provided reference image (Qwen stage skipped)")
        else:
            mv = self.request["multiview"]
            vd = d / "views"
            vd.mkdir(exist_ok=True)
            frames = []
            for fr in mv["frames"]:
                raw = base64.b64decode(mv["images_b64"][fr["file_path"]])
                (vd / fr["file_path"]).write_bytes(raw)
                frames.append({k: v for k, v in fr.items() if v is not None})
            transforms = {"camera_angle_x": mv["camera_angle_x"], "mesh_scale": mv["mesh_scale"], "frames": frames}
            atomic_write_json(vd / "transforms.json", transforms)
            if mv.get("camera_source") == "approximate":
                self.warn("multi-view cameras are marked 'approximate': poses are assumed, not measured")
            shutil.copy2(vd / frames[0]["file_path"], self.art / "reference.png")
            res = {"views_dir": str(vd), "source": "multiview", "num_views": len(frames)}
            self.log(f"using caller-provided multi-view input ({len(frames)} views)")
        # strip the large base64 payloads from the durable request record
        self.finish_stage(name, res, t0)

    def stage_pixal3d(self):
        name = "pixal3d"
        if self.stage_result(name):
            self.log("pixal3d stage already complete; reusing master")
            return
        t0 = time.time()
        self.set_stage(name, 0.35)
        d = self.stage_dir(name)
        ref = self.stage_result("reference")
        px = self.settings["pixal3d"]
        master = self.settings["master"]
        req = {
            "mode": "multiview" if ref.get("source") == "multiview" else "single",
            "image_path": None if ref.get("source") == "multiview" else str(self.stages_dir / "reference" / ref["selected"]),
            "views_dir": ref.get("views_dir"),
            "out_dir": str(d), "seed": self.settings["seed"],
            "resolution": px["resolution"], "low_vram": px["low_vram"], "max_num_tokens": px["max_num_tokens"],
            "attn_backend": px.get("attn_backend", "flash_attn"),
            "export": {"decimation_target": master["decimation_target"], "texture_size": master["texture_size"],
                       "remesh": master["remesh"], "remesh_band": master["remesh_band"], "remesh_project": master["remesh_project"]},
            "rembg_model": os.environ.get("STUDIO_REMBG_MODEL", "briaai/RMBG-2.0"),
        }
        atomic_write_json(d / "request.json", req)
        log_path = d / "log.txt"
        self.log(f"Pixal3D {req['mode']} generation at {px['resolution']} (low_vram={px['low_vram']}), master export "
                 f"{master['decimation_target']} tris / {master['texture_size']}px")
        # submit hands the job to the resident session in the pixal3d-worker container (or runs it in-process)
        run = self._run_gpu_stage("pixal3d", name, ["python", "-m", "pixal3d_worker.submit", "--request", str(d / "request.json")],
                                  log_path, ["pixal3d", "master"])
        out = json.loads((d / "output.json").read_text(encoding="utf-8"))
        shutil.copy2(d / out["master_glb"], self.art / "master.glb")
        if out.get("preprocessed_png"):
            shutil.copy2(d / out["preprocessed_png"], self.art / "reference_preprocessed.png")
        for w in out.get("warnings", []):
            self.warn(w)
        res = {"master_glb": out["master_glb"], "camera_params": out.get("camera_params"), "mesh_stats": out.get("mesh_stats"),
               "stats": out.get("stats", {}), "run": run, "effective": out.get("effective", {})}
        self.finish_stage(name, res, t0)

    def stage_blender(self):
        name = "blender"
        if self.stage_result(name):
            self.log("blender stage already complete; reusing")
            return
        t0 = time.time()
        self.set_stage(name, 0.75)
        d = self.stage_dir(name)
        opt = self.settings["optimize"]
        req = {"master_glb": str(self.art / "master.glb"), "out_dir": str(d), "optimize": opt,
               "reference_png": str(self.art / "reference.png") if (self.art / "reference.png").exists() else None,
               "name": _asset_name(self.request["prompt"])}
        atomic_write_json(d / "request.json", req)
        log_path = d / "log.txt"
        self.log(f"Blender optimisation: {opt['target_triangles']} tris, {opt['texture_size']}px, lods={opt['generate_lods']}, "
                 f"collision={opt['generate_collision']}")
        try:
            run = self.runners["blender"].run(["python", "-m", "blender.process_asset", "--request", str(d / "request.json")],
                                              str(log_path), should_cancel=self.cancelled, run_file=str(log_path.with_name("run.json")), log=self.log)
        except StageCancelled:
            raise JobCancelled()
        except StageFailed as e:
            raise JobFailed(name, str(e), {"log": str(log_path.relative_to(self.dir)), "log_tail": _log_tail(log_path)})
        out = json.loads((d / "output.json").read_text(encoding="utf-8"))
        for rel in out["files"]:
            src = d / rel
            dst = self.art / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        for w in out.get("warnings", []):
            self.warn(w)
        thumbs = self.art / "thumbs"
        thumbs.mkdir(exist_ok=True)
        previews = [f for f in out["files"] if f.startswith("previews/") and "asset_front_left" in f]
        if previews:
            make_thumb(self.art / previews[0], thumbs / "card.jpg")
        res = {"files": out["files"], "stats": out.get("stats", {}), "run": run, "outputs": out.get("outputs", {})}
        self.finish_stage(name, res, t0)

    def stage_validate(self):
        name = "validate"
        t0 = time.time()
        self.set_stage(name, 0.92)
        d = self.stage_dir(name)
        opt = self.settings["optimize"]
        bl = self.stage_result("blender")
        outputs = bl["outputs"]
        reports = {}
        reports["master"] = validate_glb(self.art / "master.glb", {"require_uv": True, "require_material": True, "require_basecolor": True})
        reports["asset"] = validate_glb(self.art / outputs["asset_glb"], {
            "max_triangles": int(opt["target_triangles"] * 1.05), "require_uv": True, "require_material": True,
            "require_basecolor": True, "height_m": opt.get("height_m"), "max_texture": opt["texture_size"],
            "require_bottom_origin": True})
        for i, lod in enumerate(outputs.get("lods", [])):
            reports[f"lod{i+1}"] = validate_glb(self.art / lod["file"], {"soft_max_triangles": int(lod["target_triangles"] * 1.1),
                                                                        "require_uv": True, "require_material": True})
        if outputs.get("collision_glb"):
            reports["collision"] = validate_glb(self.art / outputs["collision_glb"], {"max_triangles": int(opt["collision_triangles"] * 1.2)})
        atomic_write_json(d / "validation.json", reports)
        errors = {k: v["errors"] for k, v in reports.items() if v["errors"]}
        for k, v in reports.items():
            for w in v["warnings"]:
                self.warn(f"validate[{k}]: {w}")
        if errors:
            # the Blender outputs are what failed: drop that stage's result so a retry re-runs it (master/reference are kept)
            rp = self.stages_dir / "blender" / "result.json"
            if rp.exists():
                rp.unlink()
            raise JobFailed(name, "validation failed: " + json.dumps(errors), {"validation": reports})
        self.finish_stage(name, {"reports": reports}, t0)

    def _package_image_job(self, name: str, t0: float):
        ref = self.stage_result("reference") or {}
        artifacts = []
        for p in sorted(self.art.rglob("*")):
            if p.is_file():
                rel = p.relative_to(self.art).as_posix()
                artifacts.append({"name": rel, "bytes": p.stat().st_size, "url": f"/v1/jobs/{self.id}/artifacts/{rel}"})
        manifest = {
            "job_id": self.id, "kind": "image", "version": config.VERSION, "created_at": self.job["created_at"], "finished_at": time.time(),
            "request": self.request, "settings_effective": {k: v for k, v in self.settings.items() if not k.startswith("_")},
            "prompt": ref.get("prompt"), "negative_prompt": ref.get("negative_prompt"),
            "candidates": [{"file": Path(c["file"]).name, "seed": c["seed"], "time_s": c["time_s"],
                            "score": c["metrics"].get("score"), "reasons": c["metrics"].get("reasons") or c["metrics"].get("reject"),
                            "url": f"/v1/jobs/{self.id}/artifacts/reference_candidates/{Path(c['file']).name}",
                            "thumb": f"/v1/jobs/{self.id}/artifacts/thumbs/{Path(c['file']).stem}.jpg"}
                           for c in (json.loads((self.stages_dir / "reference" / "output.json").read_text(encoding="utf-8")).get("candidates", []))],
            "stats": ref.get("stats"), "resources": {"reference": ref.get("run", {})},
            "warnings": self.warnings, "timings_s": {**self.timings}, "artifacts": artifacts,
        }
        atomic_write_json(self.art / "manifest.json", manifest)
        atomic_write_json(self.dir / "manifest.json", manifest)
        self.finish_stage(name, {"artifacts": len(artifacts)}, t0)
        self.store.update(self.id, status="completed", stage="done", progress=1.0, finished_at=time.time(),
                          timings=self.timings, warnings=self.warnings, settings=self.settings)
        self.log("variations ready")

    def stage_package(self):
        name = "package"
        t0 = time.time()
        self.set_stage(name, 0.97)
        if self.job.get("kind") == "image":
            return self._package_image_job(name, t0)
        artifacts = []
        for p in sorted(self.art.rglob("*")):
            if p.is_file():
                rel = p.relative_to(self.art).as_posix()
                artifacts.append({"name": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p),
                                  "url": f"/v1/jobs/{self.id}/artifacts/{rel}"})
        px = self.stage_result("pixal3d")
        ref = self.stage_result("reference")
        bl = self.stage_result("blender")
        val = self.stage_result("validate")
        deps = {}
        dm = config.MANIFESTS_DIR / "dependency-manifest.json"
        if dm.exists():
            deps = json.loads(dm.read_text(encoding="utf-8"))
        settings_public = {k: v for k, v in self.settings.items() if not k.startswith("_")}
        manifest = {
            "job_id": self.id, "version": config.VERSION, "created_at": self.job["created_at"], "finished_at": time.time(),
            "request": {k: v for k, v in self.request.items() if k not in ("reference_image_b64", "multiview")},
            "input_mode": self.settings.get("input_mode"),
            "kind": "asset", "parent_id": self.job.get("parent_id"), "candidate": self.job.get("candidate"),
            "settings_effective": settings_public,
            "settings_requested_overrides": {k: v.get("_requested") for k, v in self.settings.items() if isinstance(v, dict) and v.get("_requested")},
            "fallbacks_applied": self.settings.get("_fallbacks_applied", []),
            "warnings": self.warnings,
            "timings_s": {**self.timings, "total": round(time.time() - self.job["started_at"], 2) if self.job.get("started_at") else None},
            "stages": {
                "reference": {k: ref.get(k) for k in ("selected", "selection", "stats", "prompt", "negative_prompt", "effective", "source", "num_views")} if ref else None,
                "pixal3d": {k: px.get(k) for k in ("camera_params", "mesh_stats", "stats", "effective")} if px else None,
                "blender": {k: bl.get(k) for k in ("stats", "outputs")} if bl else None,
                "validation": val["reports"] if val else None,
            },
            "resources": {
                "reference": (ref or {}).get("run", {}),
                "pixal3d": (px or {}).get("run", {}),
                "blender": (bl or {}).get("run", {}),
            },
            "asset": {
                "kind": "optimized static asset (decimated); not animation-ready topology, no rig",
                "master": "master.glb (untouched Pixal3D export, PNG textures)",
                "optimized": bl["outputs"].get("asset_glb") if bl else None,
                "lods": bl["outputs"].get("lods") if bl else None,
                "collision": bl["outputs"].get("collision_glb") if bl else None,
                "previews": bl["outputs"].get("previews") if bl else None,
                "units": "metres, glTF Y-up, origin at bottom centre",
            },
            "dependencies": deps,
            "artifacts": artifacts,
        }
        atomic_write_json(self.art / "manifest.json", manifest)
        atomic_write_json(self.dir / "manifest.json", manifest)
        self.finish_stage(name, {"artifacts": len(artifacts)}, t0)
        self.store.update(self.id, status="completed", stage="done", progress=1.0, finished_at=time.time(),
                          timings=self.timings, warnings=self.warnings, settings=self.settings)
        self.log("job completed")


def make_thumb(src: Path, dst: Path, size: int = 512):
    """JPEG thumbnail for library cards (Pillow is available in the studio image)."""
    try:
        from PIL import Image

        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=86)
    except Exception as e:  # noqa: BLE001
        print(f"[thumb] {src} -> {dst} failed: {e}")


def _asset_name(prompt: str) -> str:
    words = [w for w in "".join(ch if ch.isalnum() else " " for ch in prompt.lower()).split() if w]
    return "_".join(words[:5]) or "asset"


def _last_phase(log_path: Path) -> str | None:
    tail = _log_tail(log_path, 200_000)
    phase = None
    for line in tail.splitlines():
        if line.startswith("[phase] "):
            phase = line.split(" ", 1)[1].strip()
    return phase


def _log_tail(log_path: Path, n: int = 4000) -> str:
    try:
        data = log_path.read_bytes()
        return data[-n:].decode("utf-8", "replace")
    except OSError:
        return ""
