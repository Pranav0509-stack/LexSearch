"""Abstract lawyer-partner adapter (plan §4.1)."""

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class LawyerProfile:
    partner: str
    partner_lawyer_id: str
    name: str
    specializations: list[str]
    languages: list[str]
    city: str
    rating: float          # 0..5
    profile_url: Optional[str] = None


@dataclass
class BookingRef:
    partner: str
    booking_id: str
    payment_link: Optional[str] = None
    scheduled_at: Optional[str] = None    # ISO-8601


@dataclass
class BookingStatus:
    booking_id: str
    status: str            # 'proposed' | 'paid' | 'consulted' | 'cancelled' | 'refunded'
    consulted_at: Optional[str] = None


@dataclass
class BookingEvent:
    booking_id: str
    event: str             # 'paid' | 'consulted' | 'cancelled' | 'refunded'
    raw: dict


class LawyerPartnerAdapter(Protocol):
    partner_key: str

    async def search_lawyers(
        self, domain: str, city: str, language: str, limit: int = 5
    ) -> list[LawyerProfile]: ...

    async def create_booking(
        self,
        *,
        lawyer_id: str,
        user_phone_e164: str,
        preferred_slot: Optional[str] = None,
        lead_package: Optional[dict] = None,
    ) -> BookingRef: ...

    async def get_booking_status(self, booking_id: str) -> BookingStatus: ...

    async def webhook_handler(self, payload: dict) -> BookingEvent: ...
