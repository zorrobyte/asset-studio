"""Resident Pixal3D session: loads the models once and runs jobs handed over by `pixal3d_worker.submit`.

Started next to the stage runner by entrypoint.sh (restart loop) when STUDIO_PIXAL3D_RESIDENT != 0. Listens on
127.0.0.1 only; the orchestrator never talks to it directly - it still runs a per-job stage process through the
runner (submit.py), which keeps the runner's log/returncode/cancel/OOM contract unchanged.

  GET  /health            -> {"state": "loading"|"ready"|"busy", "signature": {...}, "load_s": float, "jobs_done": int}
  POST /jobs              -> {"request_path": str, "log_path": str} => {"job_id": str} (409 while busy)
  GET  /jobs/<id>         -> {"state": "running"|"done"|"failed", "error": str|None}
  POST /jobs/<id>/cancel  -> the session cannot interrupt the upstream pipeline mid-stage: the process exits and the
                             entrypoint loop restarts it (the next job pays a cold load, like every job used to)

While a job runs, fds 1/2 are pointed at the job's log file so the stage log looks exactly like the one-shot path
(including C++-level prints from cumesh/xatlas). After a failed job the process exits as well, so a job never
starts on a session left in an unknown state (models half-moved to the GPU after an OOM, etc.).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"state": "loading", "session": None, "job": None, "started": time.time()}
LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
QUEUE: "queue.Queue[tuple[str, dict, str | None]]" = None  # set in main(); jobs run on the main thread


def _release_memory():
    """Return freed heap pages to the OS. Low-VRAM mode re-creates the CPU copy of every model after each stage
    (`model.cpu()`), which leaves glibc arenas fragmented; without this the process RSS ratchets up job after job."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _log(msg: str):
    print(f"[serve] {msg}", flush=True)


def _exit_later(code: int, delay: float = 3.0):
    def _go():
        time.sleep(delay)  # let the client read the final job state first
        _log(f"exiting with code {code} (entrypoint restarts the session)")
        os._exit(code)
    threading.Thread(target=_go, daemon=True).start()


def _run_job(job_id: str, req: dict, log_path: str | None):
    from pixal3d_worker.generate import phase

    rec = JOBS[job_id]
    session = STATE["session"]
    saved = None
    if log_path:
        fd = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        sys.stdout.flush()
        sys.stderr.flush()
        saved = (os.dup(1), os.dup(2))
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
    ok = False
    try:
        phase("load")
        print(f"[resident] session loaded {time.time() - STATE['started']:.0f}s ago in {session.load_s}s, "
              f"{session.jobs_done} jobs done; models resident", flush=True)
        session.run(req)
        ok = True
    except BaseException:  # noqa: BLE001 - reported through the log + job state, then the process restarts
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if saved:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
        if ok:
            _release_memory()
        with LOCK:
            rec["state"] = "done" if ok else "failed"
            rec["ended"] = time.time()
            STATE["job"] = None
            STATE["state"] = "ready" if ok else "restarting"
    rss = _rss_mb()
    _log(f"job {job_id} {rec['state']} ({rec['ended'] - rec['started']:.1f}s); rss {rss} MB")
    if not ok:
        rec["error"] = "job failed; see the stage log"
        _exit_later(3)


def _rss_mb() -> int | None:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:  # the client gave up on a poll while a C++ step held the GIL; it retries
            pass

    def do_GET(self):
        if self.path == "/health":
            s = STATE["session"]
            return self._json(200, {"state": STATE["state"], "signature": s.signature if s else None,
                                    "load_s": s.load_s if s else None, "jobs_done": s.jobs_done if s else 0,
                                    "job": STATE["job"], "uptime_s": round(time.time() - STATE["started"], 1)})
        if self.path.startswith("/jobs/"):
            rec = JOBS.get(self.path.split("/")[2])
            return self._json(200, rec) if rec else self._json(404, {"error": "unknown job"})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/jobs":
            with LOCK:
                if STATE["state"] != "ready":
                    return self._json(409, {"error": STATE["state"], "job": STATE["job"]})
                try:
                    req = json.loads(open(body["request_path"], encoding="utf-8").read())
                except Exception as e:  # noqa: BLE001
                    return self._json(400, {"error": f"bad request_path: {e}"})
                if not STATE["session"].matches(req):
                    return self._json(409, {"error": "signature mismatch", "signature": STATE["session"].signature})
                job_id = uuid.uuid4().hex[:12]
                JOBS[job_id] = {"job_id": job_id, "state": "running", "started": time.time(), "ended": None, "error": None,
                                "request_path": body["request_path"]}
                STATE["job"] = job_id
                STATE["state"] = "busy"
            QUEUE.put((job_id, req, body.get("log_path")))
            return self._json(200, {"job_id": job_id})
        if self.path.startswith("/jobs/") and self.path.endswith("/cancel"):
            job_id = self.path.split("/")[2]
            rec = JOBS.get(job_id)
            if not rec:
                return self._json(404, {"error": "unknown job"})
            if rec["state"] == "running":
                _log(f"job {job_id} cancelled: restarting the session")
                rec["state"] = "failed"
                rec["error"] = "cancelled"
                STATE["state"] = "restarting"
                _exit_later(75, delay=0.5)
            return self._json(200, {"ok": True})
        return self._json(404, {"error": "not found"})


def main():
    global QUEUE
    import queue

    QUEUE = queue.Queue()
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("STUDIO_PIXAL3D_RESIDENT_PORT", "8712")))
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _log(f"listening on 127.0.0.1:{a.port}; loading the single-view pipeline")
    from pixal3d_worker.generate import DEFAULT_REMBG, Session
    try:
        session = Session(rembg_model=os.environ.get("STUDIO_REMBG_MODEL", DEFAULT_REMBG),
                          low_vram=os.environ.get("STUDIO_PIXAL3D_LOW_VRAM", "1") != "0",
                          attn_backend=os.environ.get("STUDIO_PIXAL3D_ATTN_BACKEND", "flash_attn"))
    except Exception:
        traceback.print_exc()
        _log("session load failed")
        sys.exit(2)
    with LOCK:
        STATE["session"] = session
        STATE["state"] = "ready"
    _release_memory()
    _log(f"ready in {session.load_s}s: {session.signature}; rss {_rss_mb()} MB")
    while True:  # jobs run here on the main thread (one at a time; the HTTP threads only queue them)
        job_id, req, log_path = QUEUE.get()
        _run_job(job_id, req, log_path)


if __name__ == "__main__":
    main()
