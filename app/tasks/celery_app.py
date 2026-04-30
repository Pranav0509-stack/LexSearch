"""Celery application — async pipelines (plan §1.3)."""

from celery import Celery

from app.config import get_settings


def make_celery() -> Celery:
    settings = get_settings()
    app = Celery(
        "nyayasathi",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=[
            "app.tasks.post_call",
            "app.tasks.doc_generation",
            "app.tasks.lawyer_match",
            "app.tasks.payments_recon",
        ],
    )
    app.conf.update(
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_time_limit=600,
        task_soft_time_limit=540,
        timezone="Asia/Kolkata",
        enable_utc=True,
    )
    return app


celery_app = make_celery()
