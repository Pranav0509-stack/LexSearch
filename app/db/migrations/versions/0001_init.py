"""baseline schema — users, calls, documents, lawyer_assignments, payments, consents, rate_limits

Revision ID: 0001_init
Revises:
Create Date: 2026-04-20
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("phone_e164", sa.String(20), unique=True, nullable=False),
        sa.Column("primary_language", sa.String(10), server_default="hi-IN"),
        sa.Column("preferred_languages", postgresql.ARRAY(sa.String())),
        sa.Column("city", sa.String(128)),
        sa.Column("state", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("last_call_at", sa.DateTime(timezone=True)),
        sa.Column("dpdp_consent_recorded_at", sa.DateTime(timezone=True)),
        sa.Column("pii_deletion_requested_at", sa.DateTime(timezone=True)),
    )

    op.create_table(
        "calls",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("plivo_call_uuid", sa.String(64), unique=True, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Column("duration_seconds", sa.Integer),
        sa.Column("language", sa.String(10)),
        sa.Column("intent", sa.String(64)),
        sa.Column("intent_confidence", sa.Numeric(3, 2)),
        sa.Column("outcome", sa.String(32)),
        sa.Column("transcript_s3_key", sa.Text),
        sa.Column("audio_s3_key", sa.Text),
        sa.Column("rag_citations", postgresql.JSONB),
        sa.Column("disclaimer_ack", sa.Boolean, server_default=sa.false()),
        sa.Column("recording_consent", sa.Boolean, server_default=sa.false()),
        sa.Column("cost_paise", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_calls_user_started", "calls", ["user_id", "started_at"])

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("call_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("calls.id"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("template_key", sa.String(64), nullable=False),
        sa.Column("language", sa.String(10), server_default="en-IN"),
        sa.Column("slots", postgresql.JSONB),
        sa.Column("unsigned_pdf_s3_key", sa.Text),
        sa.Column("signed_pdf_s3_key", sa.Text),
        sa.Column("signature_cert_s3_key", sa.Text),
        sa.Column("signature_method", sa.String(32)),
        sa.Column("signed_at", sa.DateTime(timezone=True)),
        sa.Column("sent_to_email", sa.String(256)),
        sa.Column("ses_message_id", sa.String(128)),
        sa.Column("delivered_at", sa.DateTime(timezone=True)),
        sa.Column("opened_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(32), server_default="draft"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_documents_status_created", "documents", ["status", "created_at"])

    op.create_table(
        "lawyer_assignments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("call_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("calls.id"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("partner", sa.String(32), nullable=False),
        sa.Column("partner_lawyer_id", sa.String(128)),
        sa.Column("lawyer_profile", postgresql.JSONB),
        sa.Column("domain", sa.String(64)),
        sa.Column("city", sa.String(128)),
        sa.Column("language", sa.String(10)),
        sa.Column("status", sa.String(32), server_default="proposed"),
        sa.Column("booking_fee_paise", sa.Integer),
        sa.Column("our_commission_paise", sa.Integer),
        sa.Column("partner_payout_paise", sa.Integer),
        sa.Column("razorpay_order_id", sa.String(64)),
        sa.Column("razorpay_transfer_id", sa.String(64)),
        sa.Column("paid_at", sa.DateTime(timezone=True)),
        sa.Column("consulted_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_lawyer_assignments_status_created", "lawyer_assignments", ["status", "created_at"]
    )

    op.create_table(
        "payments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("related_type", sa.String(32)),
        sa.Column("related_id", postgresql.UUID(as_uuid=True)),
        sa.Column("amount_paise", sa.Integer, nullable=False),
        sa.Column("currency", sa.String(3), server_default="INR"),
        sa.Column("razorpay_order_id", sa.String(64)),
        sa.Column("razorpay_payment_id", sa.String(64)),
        sa.Column("status", sa.String(32), server_default="created"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("captured_at", sa.DateTime(timezone=True)),
    )

    op.create_table(
        "consents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("call_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("calls.id")),
        sa.Column("scope", sa.String(32)),
        sa.Column("granted", sa.Boolean, server_default=sa.false()),
        sa.Column("audio_evidence_s3_key", sa.Text),
        sa.Column("granted_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "rate_limits",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("phone_e164", sa.String(20), nullable=False),
        sa.Column("bucket", sa.String(32), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("count", sa.Integer, server_default="0"),
        sa.UniqueConstraint("phone_e164", "bucket", "window_start", name="uq_rate_limits_window"),
    )


def downgrade() -> None:
    op.drop_table("rate_limits")
    op.drop_table("consents")
    op.drop_table("payments")
    op.drop_index("ix_lawyer_assignments_status_created", table_name="lawyer_assignments")
    op.drop_table("lawyer_assignments")
    op.drop_index("ix_documents_status_created", table_name="documents")
    op.drop_table("documents")
    op.drop_index("ix_calls_user_started", table_name="calls")
    op.drop_table("calls")
    op.drop_table("users")
