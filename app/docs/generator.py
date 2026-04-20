"""
Document generator: slot-dict → Jinja → WeasyPrint → PDF → S3 (plan §3).

Template layout:
  app/docs/templates/{template_key}.jinja
  app/docs/templates/{template_key}.schema.json
"""

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path(__file__).parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml", "jinja"]),
)


def load_schema(template_key: str) -> dict:
    path = TEMPLATES_DIR / f"{template_key}.schema.json"
    if not path.exists():
        raise FileNotFoundError(f"schema not found: {template_key}")
    return json.loads(path.read_text())


def render_html(template_key: str, slots: dict[str, Any], language: str = "en-IN") -> str:
    tpl = _env.get_template(f"{template_key}.jinja")
    return tpl.render(slots=slots, language=language)


def render_pdf(template_key: str, slots: dict[str, Any], language: str = "en-IN") -> bytes:
    """Render HTML → PDF via WeasyPrint. Returns PDF bytes."""
    from weasyprint import HTML  # lazy import (heavy system deps)

    html = render_html(template_key, slots, language)
    return HTML(string=html, base_url=str(TEMPLATES_DIR)).write_pdf()
