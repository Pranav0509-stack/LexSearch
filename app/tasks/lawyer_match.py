"""Lawyer-match async task (plan §4)."""

import logging

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="lawyer_match.run", bind=True, max_retries=3, default_retry_delay=60)
def lawyer_match_run(self, call_uuid: str) -> dict:
    logger.info("lawyer_match start call_uuid=%s", call_uuid)

    # TODO:
    # 1. Load Call + User row for domain/city/language.
    # 2. await initiate_handoff(...) → proposed assignments.
    # 3. SMS user the top-3 options + payment links.
    # 4. Wait for Razorpay webhook to flip status to 'paid'.

    return {"ok": True, "call_uuid": call_uuid}
