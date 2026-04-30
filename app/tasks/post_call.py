"""
Post-call pipeline (plan §1.3):
  1. Persist transcript + audio to S3, write Call row
  2. Intent re-extraction (non-streaming)
  3. If draft flag: enqueue doc_generation
  4. If handoff flag: enqueue lawyer_match
  5. Emit analytics event to ClickHouse
"""

import logging

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="post_call.run", bind=True, max_retries=3, default_retry_delay=30)
def post_call_run(self, call_uuid: str) -> dict:
    logger.info("post_call start call_uuid=%s", call_uuid)

    # TODO:
    # 1. Load session from Redis (call:{call_uuid}).
    # 2. Upload transcript + audio to S3 under calls/{yyyy-mm-dd}/{uuid}/.
    # 3. Upsert users row + insert calls row with outcome.
    # 4. If session.state in {'drafting','confirm'} and intent has template →
    #    celery_app.send_task('doc_generation.run', args=[call_uuid]).
    # 5. If handoff requested OR intent in HANDOFF_ONLY_INTENTS →
    #    celery_app.send_task('lawyer_match.run', args=[call_uuid]).
    # 6. Emit ClickHouse event.
    # 7. On exception: self.retry(exc=exc).

    return {"ok": True, "call_uuid": call_uuid}
