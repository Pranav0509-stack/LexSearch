"""
SMS-OTP eSignature flow (plan §3.3).

IT Act 2000 §3A-compliant for non-registered documents (legal notices, RTI,
demand letters). Not for registered sale deeds — those need Aadhaar eSign.
"""

import hashlib
import io
import secrets
from datetime import datetime
from typing import Optional

from pypdf import PdfReader, PdfWriter


def generate_otp() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def signature_certificate_text(
    *,
    signer_name: str,
    phone_e164: str,
    call_uuid: str,
    otp_reference: str,
    ip: Optional[str],
    device_fingerprint: Optional[str],
    signed_at: datetime,
    transcript_hash: str,
) -> str:
    return (
        "SIGNATURE CERTIFICATE\n\n"
        f"Signer: {signer_name}\n"
        f"Phone: {phone_e164}\n"
        f"Signed: {signed_at.isoformat()}Z\n"
        f"OTP ref: {otp_reference}\n"
        f"IP: {ip or 'n/a'}\n"
        f"Device: {device_fingerprint or 'n/a'}\n"
        f"Call UUID: {call_uuid}\n"
        f"Transcript SHA-256: {transcript_hash}\n\n"
        "This electronic signature is generated under Section 3A of the IT Act, 2000. "
        "This document (together with the audit trail above) constitutes valid "
        "electronic evidence."
    )


def compute_transcript_hash(transcript_json: str) -> str:
    return hashlib.sha256(transcript_json.encode()).hexdigest()


def stamp_pdf_with_signature(
    unsigned_pdf_bytes: bytes,
    *,
    signer_name: str,
    certificate_text: str,
) -> bytes:
    """
    Stamp the signer's typed name into the unsigned PDF and append a
    signature-certificate page. Real impl uses reportlab to build the cert
    page and pypdf to merge. Month-1 stub returns unsigned bytes.
    """
    # TODO: real stamping. For now return input unchanged so pipeline runs end-to-end.
    reader = PdfReader(io.BytesIO(unsigned_pdf_bytes))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    out = io.BytesIO()
    writer.write(out)
    _ = signer_name, certificate_text
    return out.getvalue()
