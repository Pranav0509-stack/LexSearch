"""
NyayaSathi entry point.

Keeps the historical `server:app` symbol so existing Render/Docker deploy
configs keep working. All routes live under `app.main:create_app`.
"""

from app.main import create_app

app = create_app()
