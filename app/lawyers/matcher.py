"""Lawyer match scoring + pool selection (plan §4.2)."""

from app.lawyers.adapter import LawyerProfile
from app.lawyers.mock_adapter import MockPartnerAdapter


def score_lawyer(
    profile: LawyerProfile,
    *,
    domain: str,
    city: str,
    language: str,
    partner_weight: float = 0.0,
) -> float:
    language_match = 1.0 if language in profile.languages else 0.0
    city_match = 1.0 if profile.city.lower() == (city or "").lower() else 0.3
    domain_match = 1.0 if domain in profile.specializations else 0.2
    rating_norm = profile.rating / 5.0
    return (
        0.40 * language_match
        + 0.25 * city_match
        + 0.20 * domain_match
        + 0.10 * rating_norm
        + 0.05 * partner_weight
    )


async def propose_lawyers(
    domain: str,
    city: str = "",
    language: str = "hi-IN",
    limit: int = 5,
) -> dict:
    """
    Queries all active partner adapters, scores, returns top `limit`.
    v1: Mock only; real adapters wire in after BD deals.
    """
    adapters = [MockPartnerAdapter()]

    candidates: list[LawyerProfile] = []
    for a in adapters:
        try:
            lawyers = await a.search_lawyers(domain, city, language, limit=limit * 2)
            candidates.extend(lawyers)
        except NotImplementedError:
            continue

    ranked = sorted(
        candidates,
        key=lambda p: score_lawyer(p, domain=domain, city=city, language=language),
        reverse=True,
    )[:limit]

    return {"candidates": [c.__dict__ for c in ranked]}
