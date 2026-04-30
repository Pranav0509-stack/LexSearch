"""
Lawyer-handoff orchestration (plan §4.3–4.4).

Flow: match → create booking (partner API or email-fallback) → Razorpay Route
order → SMS/WhatsApp link to user → record LawyerAssignment row.
"""

import uuid as uuidlib
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import LawyerAssignment
from app.lawyers.matcher import propose_lawyers
from app.lawyers.mock_adapter import MockPartnerAdapter


async def initiate_handoff(
    db: AsyncSession,
    *,
    call_id: uuidlib.UUID,
    user_id: uuidlib.UUID,
    domain: str,
    city: str,
    language: str,
    user_phone_e164: str,
    lead_package: Optional[dict] = None,
) -> list[LawyerAssignment]:
    """
    Returns a list of *proposed* assignments (not yet paid). User picks one
    via SMS/IVR; on pick we move that row to status='booked' and generate the
    Razorpay Route order.
    """
    picks = await propose_lawyers(domain=domain, city=city, language=language, limit=3)

    # v1: single MockAdapter. Pluggable in month 4 once BD deals close.
    adapter = MockPartnerAdapter()

    rows: list[LawyerAssignment] = []
    for cand in picks["candidates"]:
        booking = await adapter.create_booking(
            lawyer_id=cand["partner_lawyer_id"],
            user_phone_e164=user_phone_e164,
            lead_package=lead_package,
        )
        row = LawyerAssignment(
            call_id=call_id,
            user_id=user_id,
            partner=cand["partner"],
            partner_lawyer_id=cand["partner_lawyer_id"],
            lawyer_profile=cand,
            domain=domain,
            city=city,
            language=language,
            status="proposed",
        )
        db.add(row)
        rows.append(row)
        _ = booking  # real impl stashes booking.payment_link on row

    await db.flush()
    return rows
