"""Mock partner adapter — used in dev + CI, and in email-handoff fallback mode."""

import uuid as uuidlib
from typing import Optional

from app.lawyers.adapter import (
    BookingEvent,
    BookingRef,
    BookingStatus,
    LawyerProfile,
)


class MockPartnerAdapter:
    partner_key = "mock"

    async def search_lawyers(
        self, domain: str, city: str, language: str, limit: int = 5
    ) -> list[LawyerProfile]:
        samples = [
            LawyerProfile(
                partner="mock",
                partner_lawyer_id=f"mock-{i}",
                name=f"Adv. Demo {i}",
                specializations=[domain],
                languages=[language, "en-IN"],
                city=city or "Delhi",
                rating=4.2 + (i % 3) * 0.2,
            )
            for i in range(limit)
        ]
        return samples

    async def create_booking(
        self,
        *,
        lawyer_id: str,
        user_phone_e164: str,
        preferred_slot: Optional[str] = None,
        lead_package: Optional[dict] = None,
    ) -> BookingRef:
        return BookingRef(
            partner="mock",
            booking_id=str(uuidlib.uuid4()),
            payment_link="https://rzp.io/l/mock",
            scheduled_at=preferred_slot,
        )

    async def get_booking_status(self, booking_id: str) -> BookingStatus:
        return BookingStatus(booking_id=booking_id, status="proposed")

    async def webhook_handler(self, payload: dict) -> BookingEvent:
        return BookingEvent(
            booking_id=payload.get("booking_id", ""),
            event=payload.get("event", "paid"),
            raw=payload,
        )
