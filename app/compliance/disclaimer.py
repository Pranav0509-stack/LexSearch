"""
Pre-recorded disclaimer selector (plan §7.1). We serve a stable audio clip
(not live TTS) per language for audit stability.

S3 layout: s3://nyayasathi-static/disclaimer/{lang}.wav
"""

from app.config import get_settings

SUPPORTED_LANGUAGES = (
    "hi-IN", "en-IN", "ta-IN", "te-IN", "bn-IN",
    "mr-IN", "gu-IN", "kn-IN", "ml-IN", "pa-IN",
)


def disclaimer_s3_key(language: str) -> str:
    lang = language if language in SUPPORTED_LANGUAGES else "hi-IN"
    return f"disclaimer/{lang}.wav"


def disclaimer_public_url(language: str) -> str:
    settings = get_settings()
    return f"https://{settings.s3_bucket_static}.s3.{settings.aws_region}.amazonaws.com/{disclaimer_s3_key(language)}"


DISCLAIMER_SCRIPTS: dict[str, str] = {
    "hi-IN": (
        "Namaste. Main NyayaSathi hoon — ek AI assistant. Main vakeel nahi hoon, "
        "aur jo main bolunga woh kanooni salah nahi hai — balki general legal "
        "information aur document drafting hai. Is call ko record kiya jaa raha "
        "hai. Kya aap agree karte hain? Haan ya na boliye."
    ),
    "en-IN": (
        "Hello, I'm NyayaSathi — an AI assistant. I am not a lawyer, and what I "
        "say is not legal advice — it is legal information and document drafting. "
        "This call is being recorded. Do you consent? Please say yes or no."
    ),
    # Month 2+: record the other 8 languages.
}
