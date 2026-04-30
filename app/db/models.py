"""SQLAlchemy 2.0 models — schema from plan §5."""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    phone_e164: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    primary_language: Mapped[str] = mapped_column(String(10), default="hi-IN")
    preferred_languages: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    city: Mapped[Optional[str]] = mapped_column(String(128))
    state: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    last_call_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    dpdp_consent_recorded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    pii_deletion_requested_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    calls: Mapped[list["Call"]] = relationship(back_populates="user")
    documents: Mapped[list["Document"]] = relationship(back_populates="user")


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    plivo_call_uuid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer)
    language: Mapped[Optional[str]] = mapped_column(String(10))
    intent: Mapped[Optional[str]] = mapped_column(String(64))
    intent_confidence: Mapped[Optional[float]] = mapped_column(Numeric(3, 2))
    outcome: Mapped[Optional[str]] = mapped_column(
        String(32)
    )  # 'doc_sent' | 'lawyer_assigned' | 'both' | 'abandoned' | 'refused'
    transcript_s3_key: Mapped[Optional[str]] = mapped_column(Text)
    audio_s3_key: Mapped[Optional[str]] = mapped_column(Text)
    rag_citations: Mapped[Optional[list[dict]]] = mapped_column(JSONB)
    disclaimer_ack: Mapped[bool] = mapped_column(Boolean, default=False)
    recording_consent: Mapped[bool] = mapped_column(Boolean, default=False)
    cost_paise: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="calls")
    documents: Mapped[list["Document"]] = relationship(back_populates="call")
    lawyer_assignments: Mapped[list["LawyerAssignment"]] = relationship(back_populates="call")

    __table_args__ = (Index("ix_calls_user_started", "user_id", "started_at"),)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    call_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("calls.id"), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    template_key: Mapped[str] = mapped_column(String(64), nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="en-IN")
    slots: Mapped[dict] = mapped_column(JSONB, default=dict)  # PII; encrypt at rest
    unsigned_pdf_s3_key: Mapped[Optional[str]] = mapped_column(Text)
    signed_pdf_s3_key: Mapped[Optional[str]] = mapped_column(Text)
    signature_cert_s3_key: Mapped[Optional[str]] = mapped_column(Text)
    signature_method: Mapped[Optional[str]] = mapped_column(String(32))  # 'otp_v1', 'aadhaar_esign'
    signed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    sent_to_email: Mapped[Optional[str]] = mapped_column(String(256))
    ses_message_id: Mapped[Optional[str]] = mapped_column(String(128))
    delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    opened_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32), default="draft")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    call: Mapped[Call] = relationship(back_populates="documents")
    user: Mapped[User] = relationship(back_populates="documents")

    __table_args__ = (Index("ix_documents_status_created", "status", "created_at"),)


class LawyerAssignment(Base):
    __tablename__ = "lawyer_assignments"

    id: Mapped[uuid.UUID] = _uuid_pk()
    call_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("calls.id"), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    partner: Mapped[str] = mapped_column(String(32), nullable=False)  # 'legalkart', 'vakilsearch', ...
    partner_lawyer_id: Mapped[Optional[str]] = mapped_column(String(128))
    lawyer_profile: Mapped[Optional[dict]] = mapped_column(JSONB)
    domain: Mapped[Optional[str]] = mapped_column(String(64))
    city: Mapped[Optional[str]] = mapped_column(String(128))
    language: Mapped[Optional[str]] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(32), default="proposed")
    booking_fee_paise: Mapped[Optional[int]] = mapped_column(Integer)
    our_commission_paise: Mapped[Optional[int]] = mapped_column(Integer)
    partner_payout_paise: Mapped[Optional[int]] = mapped_column(Integer)
    razorpay_order_id: Mapped[Optional[str]] = mapped_column(String(64))
    razorpay_transfer_id: Mapped[Optional[str]] = mapped_column(String(64))
    paid_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    consulted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    call: Mapped[Call] = relationship(back_populates="lawyer_assignments")

    __table_args__ = (Index("ix_lawyer_assignments_status_created", "status", "created_at"),)


class Payment(Base):
    __tablename__ = "payments"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    related_type: Mapped[str] = mapped_column(String(32))  # 'lawyer_assignment' | 'premium_doc'
    related_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    amount_paise: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="INR")
    razorpay_order_id: Mapped[Optional[str]] = mapped_column(String(64))
    razorpay_payment_id: Mapped[Optional[str]] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="created")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    captured_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class Consent(Base):
    __tablename__ = "consents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    call_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("calls.id"))
    scope: Mapped[str] = mapped_column(String(32))  # 'recording' | 'data_processing' | 'marketing'
    granted: Mapped[bool] = mapped_column(Boolean, default=False)
    audio_evidence_s3_key: Mapped[Optional[str]] = mapped_column(Text)
    granted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class RateLimit(Base):
    """Audit mirror of Redis rate-limit counters (for dispute resolution + compliance)."""

    __tablename__ = "rate_limits"

    id: Mapped[uuid.UUID] = _uuid_pk()
    phone_e164: Mapped[str] = mapped_column(String(20), nullable=False)
    bucket: Mapped[str] = mapped_column(String(32), nullable=False)  # 'day' | 'month'
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    count: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        UniqueConstraint("phone_e164", "bucket", "window_start", name="uq_rate_limits_window"),
    )
