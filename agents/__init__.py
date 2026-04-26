"""Sanhita agents — tool-using legal research/drafting orchestrators.

The legal_agent module is the entry point. See `legal_agent.run` for the
public surface used by `brief_service.answer_agent` + the
`POST /api/brief/agent` endpoint.
"""

from . import legal_agent  # noqa: F401
