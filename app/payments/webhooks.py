"""Razorpay webhook receiver — captures payment.captured, refund events."""

import logging

from fastapi import APIRouter, Header, HTTPException, Request

from app.payments.razorpay_client import verify_webhook_signature

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("")
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: str = Header(default=""),
) -> dict:
    body = await request.body()
    if not verify_webhook_signature(body, x_razorpay_signature):
        raise HTTPException(status_code=400, detail="invalid signature")

    payload = await request.json()
    event = payload.get("event", "")

    # TODO: dispatch to handler per event (payment.captured, refund.created, etc.)
    logger.info("razorpay webhook: event=%s", event)
    return {"ok": True}
