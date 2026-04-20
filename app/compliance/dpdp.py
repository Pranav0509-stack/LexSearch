"""DPDP Act 2023 data-subject endpoints (plan §7.2)."""

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/data-export")
async def data_export(phone_e164: str) -> dict:
    """Enqueue a data-export job. SLA 30 days."""
    # TODO: enqueue Celery task that gathers all call, doc, payment rows,
    # zips into an S3 export, emails a signed link to the user.
    if not phone_e164.startswith("+"):
        raise HTTPException(400, "phone_e164 must be E.164-formatted")
    return {"status": "queued", "phone_e164": phone_e164}


@router.post("/delete")
async def delete_me(phone_e164: str) -> dict:
    """
    Enqueue a data-deletion job. Tombstones user row, scrubs transcripts + audio
    from S3, PII-redacts retained financial/legal records. SLA 30 days.
    """
    if not phone_e164.startswith("+"):
        raise HTTPException(400, "phone_e164 must be E.164-formatted")
    return {"status": "queued", "phone_e164": phone_e164}
