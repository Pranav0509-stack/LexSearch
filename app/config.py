"""Central settings via pydantic-settings. All env vars land here."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"

    # Postgres
    database_url: str = "postgresql+asyncpg://nyayasathi:nyayasathi@localhost:5432/nyayasathi"
    database_url_sync: str = "postgresql+psycopg2://nyayasathi:nyayasathi@localhost:5432/nyayasathi"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Plivo
    plivo_auth_id: str = ""
    plivo_auth_token: str = ""
    plivo_phone_number: str = ""
    plivo_webhook_secret: str = ""

    # Sarvam
    sarvam_api_key: str = ""
    sarvam_asr_model: str = "saaras:v2"
    sarvam_tts_model: str = "bulbul:v2"
    sarvam_llm_model: str = "sarvam-m"

    # AWS
    aws_region: str = "ap-south-1"
    s3_bucket_static: str = "nyayasathi-static"
    s3_bucket_prod: str = "nyayasathi-prod"
    ses_sender: str = "no-reply@nyayasathi.in"
    kms_key_id: str = ""

    # Razorpay
    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""
    razorpay_webhook_secret: str = ""

    # Existing LexSearch judgment corpus (kept as RAG source)
    hc_bucket: str = "indian-high-court-judgments"
    sc_bucket: str = "indian-supreme-court-judgments"

    # Rate limits (per-phone)
    rate_limit_calls_per_day: int = 5
    rate_limit_calls_per_month: int = 20

    # Retention (days)
    retention_audio_days: int = 90
    retention_transcript_days: int = 730
    retention_document_days: int = 2555

    # Observability
    sentry_dsn: str = ""

    # Fallback providers (feature-flagged)
    enable_asr_fallback: bool = False
    enable_llm_fallback: bool = False
    enable_tts_fallback: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
