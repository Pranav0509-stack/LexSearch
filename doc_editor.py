"""
Sanhita — Legal Document Editor Service.

A Google-Docs-like backend for drafting legal documents with:
- Real-time AI writing assistance (complete, improve, rephrase)
- Inline citation insertion from the 31.9M corpus
- Document templates (bail application, writ petition, plaint, etc.)
- Version history (auto-save every change)
- Export to DOCX/PDF

Documents are stored in the SQLite auth DB under `legal_documents` table.
Each doc has a title, content (markdown/rich text), doc_type, and citations.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from llm import router

logger = logging.getLogger(__name__)


# ── Document types and their AI system prompts ────────────────────────────
DOC_TYPES = {
    "bail_application": {
        "label": "Bail Application",
        "description": "Application for regular/interim bail under Section 437/439 CrPC / Section 480/483 BNSS",
        "icon": "⚖️",
        "template": """IN THE HON'BLE [COURT NAME]
[CASE NUMBER]

IN THE MATTER OF:
[APPLICANT NAME] ... Applicant/Accused
VERSUS
STATE OF [STATE] ... Respondent

APPLICATION FOR BAIL UNDER SECTION 437/439 CR.P.C.
(Section 480/483 BNSS, 2023)

RESPECTFULLY SHOWETH:

1. That the applicant has been arrested on [DATE] in connection with FIR No. [FIR NUMBER] dated [DATE] registered at [POLICE STATION], for offence(s) punishable under [SECTIONS].

2. BRIEF FACTS:
[Insert brief facts of the case]

3. GROUNDS FOR BAIL:
(i) [Ground 1 with citation]
(ii) [Ground 2 with citation]
(iii) [Ground 3 with citation]

4. UNDERTAKING:
The applicant undertakes to abide by all conditions imposed by this Hon'ble Court.

PRAYER:
It is, therefore, most respectfully prayed that this Hon'ble Court may be pleased to:
(a) Release the applicant on bail;
(b) Pass such other orders as this Hon'ble Court deems fit.

Filed by:
[ADVOCATE NAME]
Advocate for Applicant
Bar Council No.: [NUMBER]
Date: [DATE]
Place: [PLACE]""",
    },
    "anticipatory_bail": {
        "label": "Anticipatory Bail Application",
        "description": "Application for anticipatory bail under Section 438 CrPC / Section 482 BNSS",
        "icon": "🛡️",
        "template": """IN THE HON'BLE [COURT NAME]
CRIMINAL MISC. APPLICATION NO. _____ OF [YEAR]

IN THE MATTER OF:
[APPLICANT NAME] ... Applicant
VERSUS
STATE OF [STATE] ... Respondent

APPLICATION FOR ANTICIPATORY BAIL UNDER SECTION 438 CR.P.C.
(Section 482 BNSS, 2023)

MOST RESPECTFULLY SHOWETH:

1. That the applicant apprehends arrest in connection with [MATTER/FIR] for offences under [SECTIONS].

2. PERSONAL BACKGROUND OF APPLICANT:
[Insert background — employment, family, no prior criminal record, ties to community]

3. GROUNDS:
(i) The apprehension of arrest is based on: [basis]
(ii) The applicant has not committed the alleged offences: [brief denial]
(iii) [Additional grounds with case law citations]

4. RELEVANT CASE LAW:
[Insert citations]

PRAYER:
Direct the police not to arrest the applicant, or in the alternative, release the applicant on bail in the event of arrest.

Date:
Place:""",
    },
    "writ_petition": {
        "label": "Writ Petition",
        "description": "Writ petition under Article 226 (High Court) or Article 32 (Supreme Court)",
        "icon": "📜",
        "template": """IN THE HON'BLE HIGH COURT OF [STATE]
AT [BENCH]

WRIT PETITION (CIVIL/CRIMINAL) NO. _____ OF [YEAR]

IN THE MATTER OF:
[PETITIONER NAME] ... Petitioner
VERSUS
[RESPONDENT 1], [RESPONDENT 2] ... Respondents

WRIT PETITION UNDER ARTICLE 226 OF THE CONSTITUTION OF INDIA

MOST RESPECTFULLY SHOWETH:

A. FACTS OF THE CASE:
1. [Background facts]
2. [Impugned action/order]

B. QUESTIONS OF LAW:
1. [Legal question 1]
2. [Legal question 2]

C. GROUNDS:
I. [Ground 1: Constitutional violation]
II. [Ground 2: Statutory violation]
III. [Ground 3: Natural justice / due process]

D. RELEVANT CASE LAW:
[Insert citations with holdings]

PRAYER:
This Hon'ble Court may be pleased to:
(a) Issue a Writ of [Mandamus/Certiorari/Prohibition/Quo Warranto/Habeas Corpus];
(b) [Specific relief sought];
(c) Interim relief: [Stay/Injunction pending disposal]

Filed by:
[ADVOCATE NAME]
Counsel for Petitioner""",
    },
    "legal_notice": {
        "label": "Legal Notice",
        "description": "Legal notice under Section 138 NI Act, property disputes, service matters, consumer complaints",
        "icon": "📋",
        "template": """LEGAL NOTICE

To,
[RECIPIENT NAME]
[ADDRESS]

Date: [DATE]

NOTICE UNDER [SECTION/ACT]

I am instructed by my client [CLIENT NAME], [ADDRESS], to address this Legal Notice to you as under:

1. BACKGROUND:
[Brief background of the matter]

2. CAUSE OF ACTION:
[Specific act/omission giving rise to the notice]

3. LEGAL BASIS:
[Applicable law, provisions, and case law citations]

4. DEMAND:
You are hereby called upon to [specific demand] within [NUMBER] days from the date of receipt of this notice.

5. CONSEQUENCES OF NON-COMPLIANCE:
In the event of failure, my client shall be constrained to initiate appropriate legal proceedings before the competent court/authority for recovery/relief, costs of which shall be recoverable from you.

This notice is without prejudice to all other rights and remedies available to my client.

[ADVOCATE NAME]
Counsel for [CLIENT NAME]

Sent via: [Registered Post/E-mail/Courier]""",
    },
    "plaint": {
        "label": "Plaint / Civil Suit",
        "description": "Plaint for money recovery, specific performance, injunction, declaration",
        "icon": "🏛️",
        "template": """IN THE COURT OF THE [JUDGE DESIGNATION]
[COURT NAME]
[CITY]

CIVIL SUIT NO. _____ OF [YEAR]

IN THE MATTER OF:
[PLAINTIFF NAME], [AGE], [OCCUPATION], [ADDRESS] ... Plaintiff
VERSUS
[DEFENDANT NAME], [AGE], [OCCUPATION], [ADDRESS] ... Defendant

PLAINT

The Plaintiff most respectfully submits as under:

1. JURISDICTION:
This Hon'ble Court has jurisdiction to try and decide this suit as [basis of jurisdiction — territorial, pecuniary, subject matter].

2. LIMITATION:
The suit is within limitation as [basis].

3. FACTS:
[Numbered paragraphs of material facts]

4. CAUSE OF ACTION:
The cause of action arose on [DATE] when [event].

5. RELIEF CLAIMED:
The Plaintiff prays that this Hon'ble Court may be pleased to pass a decree:
(a) For [monetary amount / specific performance / injunction / declaration]
(b) For costs of this suit
(c) For such further relief as this Court deems fit

6. VALUATION AND COURT FEES:
The suit is valued at Rs. [AMOUNT] for the purpose of jurisdiction and court fees of Rs. [AMOUNT] is paid accordingly.

VERIFICATION:
I, [PLAINTIFF NAME], do hereby verify that the contents of paragraphs 1 to [NUMBER] of this Plaint are true and correct to the best of my knowledge and belief.

Verified at [CITY] on [DATE].

[SIGNATURE OF PLAINTIFF]

Through:
[ADVOCATE NAME]
Counsel for Plaintiff""",
    },
    "vakalatnama": {
        "label": "Vakalatnama",
        "description": "Power of attorney / authority to appear",
        "icon": "🖊️",
        "template": """VAKALATNAMA

IN THE HON'BLE [COURT NAME]
[CASE TITLE AND NUMBER]

I/We, [CLIENT NAME], [ADDRESS], do hereby appoint and retain:

[ADVOCATE NAME], Advocate, Bar Council Enrolment No. [NUMBER],
[ADDRESS]

to be my/our Advocate in the above matter and on my/our behalf to appear, plead, act and to do all acts, deeds and things as may be necessary for the proper conduct of the said case.

I/We agree to ratify all acts, deeds and things done by my/our said Advocate.

Date: [DATE]
Place: [PLACE]

Signature of Client: ________________
[CLIENT NAME]

Accepted:
Signature of Advocate: ________________
[ADVOCATE NAME]""",
    },
    "written_statement": {
        "label": "Written Statement",
        "description": "Written statement / reply in civil suit under Order VIII CPC",
        "icon": "📄",
        "template": """IN THE COURT OF [COURT NAME]

CIVIL SUIT NO. _____ OF [YEAR]

[PLAINTIFF NAME] ... Plaintiff
VERSUS
[DEFENDANT NAME] ... Defendant

WRITTEN STATEMENT ON BEHALF OF THE DEFENDANT

The Defendant most respectfully submits as under:

PRELIMINARY OBJECTIONS:
1. The suit is not maintainable in law and on facts.
2. The suit is barred by limitation under [applicable section].
3. [Other preliminary objections]

REPLY ON MERITS:
Para 1: [Response to plaintiff's para 1]
Para 2: [Response to plaintiff's para 2]

ADDITIONAL FACTS/COUNTER-CLAIM:
[If any additional facts or counter-claim]

PRAYER:
The Defendant prays that this Hon'ble Court may be pleased to:
(a) Dismiss the suit with costs;
(b) [Any other relief]

Verification: [Standard verification clause]

Through:
[ADVOCATE NAME]
Counsel for Defendant""",
    },
    "consumer_complaint": {
        "label": "Consumer Complaint",
        "description": "Complaint under Consumer Protection Act, 2019 before DCDRC/SCDRC/NCDRC",
        "icon": "🛒",
        "template": """BEFORE THE [DISTRICT/STATE/NATIONAL] CONSUMER DISPUTES REDRESSAL COMMISSION
[LOCATION]

CONSUMER COMPLAINT NO. _____ OF [YEAR]

[COMPLAINANT NAME], [ADDRESS] ... Complainant
VERSUS
[OPPOSITE PARTY/SERVICE PROVIDER], [ADDRESS] ... Opposite Party

CONSUMER COMPLAINT UNDER SECTION 35 OF THE CONSUMER PROTECTION ACT, 2019

RESPECTFULLY SHOWETH:

1. The complainant is a consumer as defined under Section 2(7) of the Consumer Protection Act, 2019, having availed [goods/services] from the Opposite Party.

2. FACTS:
[Detailed facts of the transaction and deficiency in service/defective goods]

3. DEFICIENCY IN SERVICE / DEFECT IN GOODS:
[Specific deficiencies under Section 2(11)/Section 2(10) of the CP Act]

4. COMPENSATION CLAIMED:
(a) Refund: Rs. [AMOUNT]
(b) Compensation for mental agony: Rs. [AMOUNT]
(c) Litigation costs: Rs. [AMOUNT]
Total: Rs. [AMOUNT]

ANNEXURES:
1. [Receipt/Invoice]
2. [Correspondence]
3. [Evidence of deficiency]

PRAYER:
[Relief sought]""",
    },
    "affidavit": {
        "label": "Affidavit",
        "description": "General affidavit / supporting affidavit",
        "icon": "✍️",
        "template": """AFFIDAVIT

I, [DEPONENT NAME], [AGE], [OCCUPATION], [ADDRESS], do hereby solemnly affirm and state on oath as under:

1. I am the [applicant/petitioner/plaintiff/defendant] in the above-captioned matter and am fully conversant with the facts stated herein.

2. [Substantive averments numbered]

3. I state that the facts averred herein are true and correct to the best of my knowledge and belief and nothing material has been concealed.

DEPONENT

VERIFICATION:
Verified at [CITY] on [DATE] that the contents of this Affidavit are true and correct to the best of my knowledge and belief.

DEPONENT

SWORN BEFORE ME / NOTARY / OATH COMMISSIONER

[SIGNATURE AND SEAL]""",
    },
    "memo_of_appeal": {
        "label": "Memo of Appeal",
        "description": "Memorandum of appeal — criminal or civil",
        "icon": "📑",
        "template": """IN THE HON'BLE [APPELLATE COURT]

CRIMINAL/CIVIL APPEAL NO. _____ OF [YEAR]

IN THE MATTER OF:
[APPELLANT NAME] ... Appellant
VERSUS
[RESPONDENT NAME] ... Respondent

MEMORANDUM OF APPEAL AGAINST THE JUDGMENT/ORDER
DATED [DATE] PASSED BY [LOWER COURT] IN [CASE NUMBER]

GROUNDS OF APPEAL:

1. [Ground 1 — Factual error / Perversity of findings]

2. [Ground 2 — Erroneous appreciation of evidence]

3. [Ground 3 — Legal error / wrong interpretation of statute]

4. [Ground 4 — Sentence disproportionate / Reliefs inadequate]

PRAYER:
Allow the appeal, set aside the impugned judgment/order, and [specific relief].

Through:
[ADVOCATE NAME]
Counsel for Appellant""",
    },
}


# ── AI writing prompts ─────────────────────────────────────────────────────
_EDITOR_SYSTEM = """You are an expert Indian legal document drafter with 20+ years of experience.
You assist advocates in drafting precise, court-ready legal documents.

Rules:
1. Use formal legal English appropriate for Indian courts.
2. Follow Indian court formatting conventions.
3. When citing cases, use [CASE: Title | Court | Year] format so they can be linked.
4. Suggest applicable sections, rules, and precedents.
5. Be concise but complete — courts value precision.
6. Where you see [PLACEHOLDER] text, help fill it in based on context.
7. Never invent case citations — use generic placeholders like [cite authority] instead.
8. Always end legal clauses with proper legal language ("respectfully prays", "humbly submits", etc.)."""

_IMPROVE_SYSTEM = """You are an expert Indian legal document editor.
Improve the given legal text: fix grammar, improve legal precision, strengthen arguments,
ensure proper legal formatting. Keep the same structure and meaning but make it court-ready.
Return only the improved text, no explanation."""

_COMPLETE_SYSTEM = """You are an expert Indian legal document drafter.
Complete the legal text naturally, maintaining the document's style and legal accuracy.
Continue from exactly where the text ends. Return only the completion, not the original text."""

_SUGGEST_CASES_SYSTEM = """You are an Indian legal research assistant.
Given a legal argument or claim, suggest what TYPE of case law would support it.
Return a JSON object: {"search_query": "...", "explanation": "..."}
The search_query should be optimized for searching 31.9M Indian court judgments."""


def ai_complete(text: str, doc_type: str = "", cursor_text: str = "") -> str:
    """Continue writing from where the cursor is."""
    context = f"Document type: {DOC_TYPES.get(doc_type, {}).get('label', 'Legal Document')}\n\n"
    prompt = f"{context}Complete this legal text:\n\n{cursor_text or text[-500:]}"
    try:
        resp = router.generate(_COMPLETE_SYSTEM, prompt, temperature=0.3, max_tokens=400)
        return resp.text
    except Exception as e:
        logger.error("ai_complete failed: %s", e)
        return ""


def ai_improve(text: str, doc_type: str = "") -> str:
    """Improve selected text."""
    context = f"Document type: {DOC_TYPES.get(doc_type, {}).get('label', 'Legal Document')}\n\n"
    prompt = f"{context}Improve this legal text:\n\n{text}"
    try:
        resp = router.generate(_IMPROVE_SYSTEM, prompt, temperature=0.2, max_tokens=600)
        return resp.text
    except Exception as e:
        logger.error("ai_improve failed: %s", e)
        return text


def ai_write_section(instruction: str, doc_type: str = "", context: str = "") -> str:
    """Write a complete section based on instruction."""
    doc_label = DOC_TYPES.get(doc_type, {}).get("label", "Legal Document")
    prompt = f"""Document type: {doc_label}
Document context: {context[:400] if context else 'Not provided'}

Write this section: {instruction}"""
    try:
        resp = router.generate(_EDITOR_SYSTEM, prompt, temperature=0.25, max_tokens=800)
        return resp.text
    except Exception as e:
        logger.error("ai_write_section failed: %s", e)
        return ""


def ai_suggest_case_search(argument: str) -> dict[str, str]:
    """Given a legal argument, suggest what case law to search for."""
    try:
        resp = router.generate(
            _SUGGEST_CASES_SYSTEM,
            f"Argument: {argument}",
            temperature=0.2,
            max_tokens=150,
            prefer="openai",
        )
        m = re.search(r'\{.*\}', resp.text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    # Fallback
    words = argument.split()[:8]
    return {"search_query": " ".join(words), "explanation": "Search based on argument keywords"}


def ai_insert_citation(case: dict[str, Any], context: str = "") -> str:
    """Format a case citation for insertion into a document."""
    title = case.get("title", "")
    court = case.get("court", "")
    year = case.get("year", "")
    citation = case.get("citation", "")
    verdict = case.get("verdict", "")
    excerpt = (case.get("excerpt") or "")[:200]

    # Build citation string
    cite_str = title
    if court:
        cite_str += f", {court}"
    if year:
        cite_str += f" ({year})"
    if citation and citation != title:
        cite_str += f", {citation}"

    return cite_str


def format_document_for_export(content: str, doc_type: str, title: str) -> str:
    """Clean up document content for export."""
    return content


def list_doc_types() -> list[dict[str, Any]]:
    """Return all document types with metadata."""
    return [
        {
            "id": k,
            "label": v["label"],
            "description": v["description"],
            "icon": v["icon"],
        }
        for k, v in DOC_TYPES.items()
    ]


def get_template(doc_type: str) -> str:
    """Return the template for a document type."""
    return DOC_TYPES.get(doc_type, {}).get("template", "")
