"""Document generation async task (plan §3.2)."""

import logging

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="doc_generation.run", bind=True, max_retries=3, default_retry_delay=60)
def doc_generation_run(self, call_uuid: str) -> dict:
    logger.info("doc_generation start call_uuid=%s", call_uuid)

    # TODO:
    # 1. Load Call row + session slots.
    # 2. Resolve template_key from intent.
    # 3. LLM normalize slots against template schema.
    # 4. render_pdf(template_key, slots, language).
    # 5. Upload to S3 under docs/{user_id}/{doc_id}/unsigned.pdf.
    # 6. Insert Document row (status='awaiting_sig').
    # 7. SMS OTP-link to user for eSignature.

    return {"ok": True, "call_uuid": call_uuid}
