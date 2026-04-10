from __future__ import annotations

import sys

from app import run_job


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python job_runner.py <job_id>")

    run_job(sys.argv[1])
