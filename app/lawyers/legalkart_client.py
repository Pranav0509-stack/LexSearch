"""LegalKart adapter — real API shape comes after BD deal closes (plan §4.1)."""

from typing import Optional

from app.lawyers.adapter import (
    BookingEvent,
    BookingRef,
    BookingStatus,
    LawyerProfile,
)


class LegalKartAdapter:
    partner_key = "legalkart"

    async def search_lawyers(
        self, domain: str, city: str, language: str, limit: int = 5
    ) -> list[LawyerProfile]:
        raise NotImplementedError("BD deal pending — use MockPartnerAdapter or email-handoff")

    async def create_booking(
        self,
        *,
        lawyer_id: str,
        user_phone_e164: str,
        preferred_slot: Optional[str] = None,
        lead_package: Optional[dict] = None,
    ) -> BookingRef:
        raise NotImplementedError

    async def get_booking_status(self, booking_id: str) -> BookingStatus:
        raise NotImplementedError

    async def webhook_handler(self, payload: dict) -> BookingEvent:
        raise NotImplementedError
