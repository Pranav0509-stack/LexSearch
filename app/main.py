"""FastAPI app factory. `server.py` imports `create_app` from here."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(
        title="NyayaSathi",
        version="0.1.0",
        description="Voice-first legal access layer for India.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Legacy LexSearch routes (judgment search — kept as public tool + RAG source).
    from app.api.lexsearch import router as lexsearch_router
    app.include_router(lexsearch_router)

    # Voice gateway (Plivo webhooks + media WS).
    from app.voice.plivo_webhook import router as plivo_router
    app.include_router(plivo_router, prefix="/voice/plivo", tags=["voice"])

    from app.voice.plivo_media_ws import router as plivo_ws_router
    app.include_router(plivo_ws_router, prefix="/voice/plivo", tags=["voice"])

    # eSignature web page.
    from app.api.sign_page import router as sign_router
    app.include_router(sign_router, prefix="/s", tags=["esign"])

    # DPDP data-subject endpoints.
    from app.compliance.dpdp import router as dpdp_router
    app.include_router(dpdp_router, prefix="/api/user", tags=["compliance"])

    # Razorpay webhooks.
    from app.payments.webhooks import router as razorpay_router
    app.include_router(razorpay_router, prefix="/webhooks/razorpay", tags=["payments"])

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "env": settings.env, "version": app.version}

    return app
