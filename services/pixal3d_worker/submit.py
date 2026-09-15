"""Stage entry point the orchestrator runs for the Pixal3D stage (through the stage runner, one process per job).

Hands the job to the resident session (serve.py) when one is ready and matches the request; otherwise runs the
job in this process exactly as before (`generate.run_one_shot`). Either way this process's stdout is the stage
log, its exit code is the stage result and SIGTERM (runner cancel) cancels the job, so the orchestrator, OOM
fallback and re-attach logic do not know the difference.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request

PORT = int(os.environ.get("STUDIO_PIXAL3D_RESIDENT_PORT", "8712"))
BASE = f"http://127.0.0.1:{PORT}"
LOAD_WAIT_S = int(os.environ.get("STUDIO_PIXAL3D_RESIDENT_WAIT_S", "300"))


def _api(path: str, body: dict | None = None, timeout: float = 5.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read() or b"{}")


def _health():
    try:
        return _api("/health")[1]
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _own_log_path() -> str | None:
    """The runner redirects our stdout to the stage log; the session appends to that same file."""
    try:
        p = os.readlink("/proc/self/fd/1")
        return p if p.startswith("/") and os.path.isfile(p) else None
    except OSError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    a = ap.parse_args()
    req = json.loads(open(a.request, encoding="utf-8").read())
    log_path = _own_log_path()

    def one_shot(reason: str) -> int:
        print(f"[resident] {reason}; running the stage in this process", flush=True)
        from pixal3d_worker.generate import run_one_shot
        run_one_shot(req)
        return 0

    if os.environ.get("STUDIO_PIXAL3D_RESIDENT", "1") == "0" or req.get("mode", "single") != "single" or not log_path:
        return one_shot("resident session disabled or not applicable")

    t0 = time.time()
    h = _health()
    while h and h["state"] in ("loading", "busy", "restarting") and time.time() - t0 < LOAD_WAIT_S:
        time.sleep(2)
        h = _health()
    if not h:
        return one_shot("no resident session")
    if h["state"] != "ready":
        return one_shot(f"resident session is {h['state']} after {LOAD_WAIT_S}s")

    try:
        status, r = _api("/jobs", {"request_path": a.request, "log_path": log_path})
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        return one_shot(f"resident session refused the job ({e.code}: {detail})")
    except (urllib.error.URLError, OSError) as e:
        return one_shot(f"resident session unreachable ({e})")
    job_id = r["job_id"]

    def on_term(signum, frame):
        try:
            _api(f"/jobs/{job_id}/cancel", {})
        except Exception:  # noqa: BLE001
            pass
        sys.exit(143)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    last_ok = time.time()
    while True:
        time.sleep(1.0)
        try:
            # the session answers late while a long C++ step (xatlas, remeshing) holds the GIL: tolerate that
            _, s = _api(f"/jobs/{job_id}", timeout=20)
            last_ok = time.time()
        except (urllib.error.URLError, OSError, ValueError):
            if time.time() - last_ok > 900:
                print("[resident] lost the resident session mid-job", flush=True)
                return 1
            continue
        if s["state"] == "done":
            return 0
        if s["state"] == "failed":
            print(f"[resident] job failed: {s.get('error')}", flush=True)
            return 1


if __name__ == "__main__":
    sys.exit(main())
