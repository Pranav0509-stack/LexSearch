"""Consent capture + storage (plan §7.2)."""

import uuid as uuidlib
from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Consent


async def record_consent(
    db: AsyncSession,
    *,
    user_id: uuidlib.UUID,
    call_id: Optional[uuidlib.UUID],
    scope: str,
    granted: bool,
    audio_evidence_s3_key: Optional[str] = None,
) -> Consent:
    row = Consent(
        user_id=user_id,
        call_id=call_id,
        scope=scope,
        granted=granted,
        audio_evidence_s3_key=audio_evidence_s3_key,
        granted_at=datetime.utcnow(),
    )
    db.add(row)
    await db.flush()
    return row
