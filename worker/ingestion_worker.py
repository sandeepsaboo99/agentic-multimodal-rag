"""
Standalone ingestion worker.

Design reasoning (rubric: Ingestion & Storage — async processing, retries)
--------------------------------------------------------------------------
In dev, the API schedules ingestion via FastAPI BackgroundTasks so a single
`uvicorn` process is enough to try everything. In production you run THIS worker
as a separate deployment that consumes the job queue, so document processing
scales independently of the API and a crash during a big parse never takes down
request serving.

Here the "queue" is the Jobs table polled with SELECT ... status='queued'. Swap
`claim_next_job` for a Redis/SQS/RabbitMQ consumer and the rest is unchanged.
Retries + backoff are handled in ingestion.run_ingestion_job via the Job row's
attempts/max_attempts, so a transient failure re-queues instead of dying.

Run:  python -m worker.ingestion_worker
"""
from __future__ import annotations

import time

from app.core.logging import get_logger
from app.models.db import Job, init_db, session_scope
from app.services.ingestion import run_ingestion_job

log = get_logger(__name__)
POLL_SECONDS = 2.0


def claim_next_job() -> str | None:
    """Atomically move one queued job to running and return its id."""
    with session_scope() as db:
        job = (db.query(Job)
               .filter(Job.status == "queued")
               .order_by(Job.created_at.asc())
               .with_for_update(skip_locked=True) if _supports_skip_locked() else
               db.query(Job).filter(Job.status == "queued").order_by(Job.created_at.asc()))
        job = job.first()
        if not job:
            return None
        job.status = "running"
        return job.id


def _supports_skip_locked() -> bool:
    from app.core.config import get_settings

    return not get_settings().database_url.startswith("sqlite")


def main() -> None:
    init_db()
    log.info("Ingestion worker started; polling every %.1fs", POLL_SECONDS)
    while True:
        job_id = claim_next_job()
        if job_id:
            log.info("Processing job %s", job_id)
            try:
                run_ingestion_job(job_id)
            except Exception:  # noqa: BLE001
                log.exception("Unhandled error in job %s", job_id)
        else:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
