"""
Standalone pipeline runner — called by the dashboard worker as a subprocess.
Usage: python -m src.run_job <job_id>

Runs in its own process so YOLO/GPU memory is fully isolated from Streamlit.
Writes progress to DB every 3 seconds so the dashboard can poll it.
"""
from __future__ import annotations

import sys
import time
import traceback

from src import database as db
from src.pipeline import run_pipeline


def main(job_id: str) -> None:
    db.init_db()
    job = db.load_job(job_id)
    if not job:
        print(f"[run_job] job {job_id} not found", file=sys.stderr)
        sys.exit(1)

    job["status"]     = "processing"
    job["progress"]   = 0.0
    job["status_msg"] = "Starting…"
    db.save_job(job)

    _last_write = 0.0

    def _cb(frac: float, msg: str) -> None:
        nonlocal _last_write
        now = time.time()
        job["progress"]   = frac
        job["status_msg"] = msg
        if now - _last_write >= 1.0:
            try:
                db.save_job(job)
            except Exception:
                pass
            _last_write = now

    try:
        job["result"]     = run_pipeline(job["path"], progress_cb=_cb)
        job["status"]     = "done"
        job["status_msg"] = "Complete ✓"
        job["progress"]   = 1.0
    except Exception as exc:
        job["status"]     = "error"
        job["error"]      = traceback.format_exc()
        job["status_msg"] = f"Error: {exc}"
        job["progress"]   = 0.0

    try:
        db.save_job(job)
    except Exception as save_exc:
        job["result"]     = None
        job["status"]     = "error"
        job["status_msg"] = "Save failed"
        job["error"]      = f"Result serialization error: {save_exc}"
        try:
            db.save_job(job)
        except Exception:
            pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.run_job <job_id>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
