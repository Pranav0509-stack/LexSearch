"""Intent taxonomy — plan §2.1."""

from typing import Literal

Intent = Literal[
    "cheque_bounce",
    "consumer_complaint",
    "landlord_tenant",
    "employment",
    "rti",
    "domestic_violence",
    "family_divorce_maintenance",
    "property_dispute",
    "criminal_fir_complaint",
    "defamation",
    "general_legal_notice",
    "info_only",
    "out_of_scope",
]

LEGAL_INTENTS: tuple[Intent, ...] = (
    "cheque_bounce",
    "consumer_complaint",
    "landlord_tenant",
    "employment",
    "rti",
    "domestic_violence",
    "family_divorce_maintenance",
    "property_dispute",
    "criminal_fir_complaint",
    "defamation",
    "general_legal_notice",
)

# Intents that MUST route to human lawyer (no unilateral drafting).
HANDOFF_ONLY_INTENTS: tuple[Intent, ...] = (
    "domestic_violence",
    "family_divorce_maintenance",
    "criminal_fir_complaint",
    "property_dispute",
)


TEMPLATE_FOR_INTENT: dict[str, str] = {
    "cheque_bounce": "cheque_bounce_138",
    "consumer_complaint": "consumer_complaint",
    "landlord_tenant": "landlord_notice",
    "employment": "employer_notice",
    "rti": "rti_form_a",
    "defamation": "defamation_notice",
    "criminal_fir_complaint": "police_complaint_fir",
    "general_legal_notice": "general_notice",
}
