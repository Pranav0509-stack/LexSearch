"""Payments reconciliation task — reconcile Razorpay Route transfers nightly."""

import logging

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="payments_recon.run", bind=True)
def payments_recon_run(self) -> dict:
    logger.info("payments_recon start")
    # TODO: fetch settled transfers from Razorpay, mark LawyerAssignment rows.
    return {"ok": True}
