"""Razorpay SDK wrapper."""

from functools import lru_cache
from typing import Any

import razorpay

from app.config import get_settings


@lru_cache
def client() -> razorpay.Client:
    settings = get_settings()
    c = razorpay.Client(auth=(settings.razorpay_key_id, settings.razorpay_key_secret))
    return c


def create_order(
    *,
    amount_paise: int,
    currency: str = "INR",
    notes: dict[str, Any] | None = None,
) -> dict:
    return client().order.create(
        {
            "amount": amount_paise,
            "currency": currency,
            "notes": notes or {},
        }
    )


def verify_webhook_signature(payload_bytes: bytes, signature: str) -> bool:
    settings = get_settings()
    try:
        client().utility.verify_webhook_signature(
            payload_bytes.decode(), signature, settings.razorpay_webhook_secret
        )
        return True
    except razorpay.errors.SignatureVerificationError:
        return False
