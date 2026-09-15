"""Durable job state in SQLite (WAL). Shared by the API and the worker container through the /data volume."""
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import config

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  stage TEXT NOT NULL DEFAULT 'queued',
  progress REAL NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  started_at REAL,
  finished_at REAL,
  request_json TEXT NOT NULL,
  settings_json TEXT NOT NULL DEFAULT '{}',
  warnings_json TEXT NOT NULL DEFAULT '[]',
  timings_json TEXT NOT NULL DEFAULT '{}',
  error_json TEXT,
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  attempt INTEGER NOT NULL DEFAULT 0,
  worker_id TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created_at);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  ts REAL NOT NULL,
  level TEXT NOT NULL,
  message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_job ON events(job_id, id);
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""

# columns added after the first release; applied with ALTER TABLE when missing (keeps existing job rows)
MIGRATIONS = [
    ("kind", "TEXT NOT NULL DEFAULT 'asset'"),      # 'image' (variations only) | 'asset' (full pipeline)
    ("parent_id", "TEXT"),                          # image job an asset job was created from
    ("candidate", "TEXT"),                          # candidate file (relative to the parent's candidates dir)
    ("title", "TEXT"),
    ("favorite", "INTEGER NOT NULL DEFAULT 0"),
    ("archived", "INTEGER NOT NULL DEFAULT 0"),
    ("notes", "TEXT"),
    ("priority", "INTEGER NOT NULL DEFAULT 0"),     # higher runs first among queued jobs
]

DEFAULT_SETTINGS = {
    "auto_process": False,       # selected images start 3D generation immediately (else they are held for review)
    "default_variations": 4,
    "default_image_model": "flux2-klein-9b",
    "default_quality": "balanced",
    "default_style": "mobile_factory",
    "default_target_triangles": 20000,
    "default_texture_size": 2048,
    "style_edits": {},           # per style id: {label, style_clause, background, negative_extra, target_triangles, texture_size};
                                 # key "custom" is the user's own style. Sent as custom_style with jobs by the portal.
}


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=30, isolation_level=None, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(jobs)").fetchall()}
    for col, decl in MIGRATIONS:
        if col not in have:
            con.execute(f"ALTER TABLE jobs ADD COLUMN {col} {decl}")
    con.execute("CREATE INDEX IF NOT EXISTS jobs_parent ON jobs(parent_id)")
    return con


class JobStore:
    def __init__(self, path: Path | None = None):
        self.con = connect(path)

    @contextmanager
    def tx(self):
        with _LOCK:
            self.con.execute("BEGIN IMMEDIATE")
            try:
                yield self.con
                self.con.execute("COMMIT")
            except Exception:
                self.con.execute("ROLLBACK")
                raise

    # ---- creation / lookup -------------------------------------------------
    def create(self, request: dict, settings: dict, kind: str = "asset", parent_id: str | None = None,
               candidate: str | None = None, title: str | None = None, held: bool = False) -> str:
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        now = time.time()
        status = "held" if held else "queued"
        with self.tx() as c:
            c.execute(
                "INSERT INTO jobs(id,status,stage,created_at,updated_at,request_json,settings_json,kind,parent_id,candidate,title) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, status, status, now, now, json.dumps(request), json.dumps(settings), kind, parent_id, candidate,
                 title or (request.get("prompt") or "")[:80]))
        self.event(job_id, "info", "job held for review" if held else "job queued")
        return job_id

    def get(self, job_id: str) -> dict | None:
        row = self.con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row(row) if row else None

    def list_jobs(self, limit: int = 50, status: str | None = None, kind: str | None = None, parent_id: str | None = None,
                  q: str | None = None, favorite: bool | None = None, archived: bool | None = False, offset: int = 0) -> "list[dict]":
        where, args = [], []
        if status:
            where.append("status=?"); args.append(status)
        if kind:
            where.append("kind=?"); args.append(kind)
        if parent_id:
            where.append("parent_id=?"); args.append(parent_id)
        if q:
            where.append("(title LIKE ? OR request_json LIKE ?)"); args += [f"%{q}%", f"%{q}%"]
        if favorite is not None:
            where.append("favorite=?"); args.append(1 if favorite else 0)
        if archived is not None:
            where.append("archived=?"); args.append(1 if archived else 0)
        sql = "SELECT * FROM jobs" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        rows = self.con.execute(sql, (*args, limit, offset))
        return [self._row(r) for r in rows]

    def children_counts(self, parent_ids: list) -> dict:
        if not parent_ids:
            return {}
        q = ",".join("?" * len(parent_ids))
        rows = self.con.execute(f"SELECT parent_id, COUNT(*) AS n FROM jobs WHERE parent_id IN ({q}) GROUP BY parent_id", parent_ids)
        return {r["parent_id"]: r["n"] for r in rows}

    def children(self, parent_id: str) -> "list[dict]":
        rows = self.con.execute("SELECT * FROM jobs WHERE parent_id=? ORDER BY created_at", (parent_id,))
        return [self._row(r) for r in rows]

    def queue(self) -> "list[dict]":
        rows = self.con.execute("SELECT * FROM jobs WHERE status IN ('held','queued','running') ORDER BY "
                                "CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END, priority DESC, created_at")
        return [self._row(r) for r in rows]

    def set_status_simple(self, job_id: str, from_status: tuple, to_status: str) -> str | None:
        with self.tx() as c:
            row = c.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            if row["status"] not in from_status:
                return row["status"]
            c.execute("UPDATE jobs SET status=?, stage=?, updated_at=? WHERE id=?", (to_status, to_status, time.time(), job_id))
        self.event(job_id, "info", f"{row['status']} -> {to_status}")
        return to_status

    # ---- settings -----------------------------------------------------------
    def get_settings(self) -> dict:
        out = dict(DEFAULT_SETTINGS)
        for r in self.con.execute("SELECT key, value FROM settings"):
            try:
                out[r["key"]] = json.loads(r["value"])
            except json.JSONDecodeError:
                pass
        return out

    def put_settings(self, values: dict) -> dict:
        with self.tx() as c:
            for k, v in values.items():
                if k in DEFAULT_SETTINGS:
                    c.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                              (k, json.dumps(v)))
        return self.get_settings()

    def events_since(self, since_id: int, limit: int = 200) -> "list[dict]":
        rows = self.con.execute("SELECT id, job_id, ts, level, message FROM events WHERE id>? ORDER BY id LIMIT ?", (since_id, limit))
        return [dict(r) for r in rows]

    def last_event_id(self) -> int:
        r = self.con.execute("SELECT COALESCE(MAX(id),0) FROM events").fetchone()
        return int(r[0])

    def delete_job(self, job_id: str):
        with self.tx() as c:
            c.execute("DELETE FROM events WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    @staticmethod
    def _row(r) -> dict:
        d = dict(r)
        for k in ("request_json", "settings_json", "warnings_json", "timings_json", "error_json"):
            v = d.pop(k)
            d[k[:-5]] = json.loads(v) if v else ({} if k != "warnings_json" else [])
        d["cancel_requested"] = bool(d["cancel_requested"])
        d["favorite"] = bool(d.get("favorite"))
        d["archived"] = bool(d.get("archived"))
        return d

    # ---- worker side -------------------------------------------------------
    def claim_next(self, worker_id: str, kind: str | None = None) -> dict | None:
        """Claim the next queued job; `kind` restricts the lane ('image' or 'asset')."""
        with self.tx() as c:
            if kind:
                row = c.execute("SELECT id FROM jobs WHERE status='queued' AND kind=? ORDER BY priority DESC, created_at LIMIT 1",
                                (kind,)).fetchone()
            else:
                row = c.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY priority DESC, created_at LIMIT 1").fetchone()
            if not row:
                return None
            now = time.time()
            c.execute("UPDATE jobs SET status='running', started_at=COALESCE(started_at,?), updated_at=?, worker_id=?, "
                      "attempt=attempt+1 WHERE id=?", (now, now, worker_id, row["id"]))
        return self.get(row["id"])

    def requeue_orphans(self, worker_id: str) -> list[str]:
        """On worker start: jobs left 'running' by a crashed/restarted worker go back to the queue."""
        with self.tx() as c:
            rows = c.execute("SELECT id, cancel_requested FROM jobs WHERE status='running'").fetchall()
            ids = [r["id"] for r in rows if not r["cancel_requested"]]
            dropped = [r["id"] for r in rows if r["cancel_requested"]]
            for i in ids:
                c.execute("UPDATE jobs SET status='queued', updated_at=? WHERE id=?", (time.time(), i))
            for i in dropped:  # a cancel was in flight when the worker died: finish the cancel, never re-run it
                c.execute("UPDATE jobs SET status='cancelled', stage='cancelled', finished_at=?, updated_at=? WHERE id=?",
                          (time.time(), time.time(), i))
        for i in ids:
            self.event(i, "warn", f"requeued after worker restart ({worker_id}); completed stages will be reused")
        for i in dropped:
            self.event(i, "warn", f"cancelled (cancel was pending when the worker restarted: {worker_id})")
        return ids

    def update(self, job_id: str, **fields):
        cols, vals = [], []
        for k, v in fields.items():
            if k in ("settings", "warnings", "timings", "error"):
                k = k + "_json"
                v = json.dumps(v)
            cols.append(f"{k}=?")
            vals.append(v)
        cols.append("updated_at=?")
        vals.append(time.time())
        vals.append(job_id)
        with self.tx() as c:
            c.execute(f"UPDATE jobs SET {', '.join(cols)} WHERE id=?", vals)

    def request_cancel(self, job_id: str) -> str | None:
        with self.tx() as c:
            row = c.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            if row["status"] in ("queued", "held"):
                c.execute("UPDATE jobs SET status='cancelled', stage='cancelled', finished_at=?, updated_at=?, "
                          "cancel_requested=1 WHERE id=?", (time.time(), time.time(), job_id))
                return "cancelled"
            if row["status"] == "running":
                c.execute("UPDATE jobs SET cancel_requested=1, updated_at=? WHERE id=?", (time.time(), job_id))
                return "cancelling"
            return row["status"]

    def retry(self, job_id: str, allow_completed: bool = False, settings: dict | None = None) -> str | None:
        """Requeue a failed/cancelled job (or a completed one when reprocessing from a stage). Completed stages
        (stages/<name>/result.json) are reused by the pipeline unless the caller removed them."""
        with self.tx() as c:
            row = c.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            allowed = ("failed", "cancelled") + (("completed",) if allow_completed else ())
            if row["status"] not in allowed:
                return row["status"]
            if settings is not None:
                c.execute("UPDATE jobs SET settings_json=? WHERE id=?", (json.dumps(settings), job_id))
            c.execute("UPDATE jobs SET status='queued', stage='queued', cancel_requested=0, error_json=NULL, finished_at=NULL, "
                      "updated_at=? WHERE id=?", (time.time(), job_id))
        self.event(job_id, "info", "retry requested; completed stages will be reused")
        return "queued"

    def is_cancel_requested(self, job_id: str) -> bool:
        row = self.con.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
        return row is None or bool(row["cancel_requested"])  # a purged job must stop too

    # ---- events ------------------------------------------------------------
    def event(self, job_id: str, level: str, message: str):
        with self.tx() as c:
            c.execute("INSERT INTO events(job_id,ts,level,message) VALUES(?,?,?,?)", (job_id, time.time(), level, message))

    def events(self, job_id: str, limit: int = 200) -> list[dict]:
        rows = self.con.execute("SELECT ts,level,message FROM events WHERE job_id=? ORDER BY id DESC LIMIT ?",
                                (job_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]
