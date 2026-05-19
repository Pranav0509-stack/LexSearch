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
You assist advocates in drafting precise, court-ready legal documents that lawyers can FILE without redoing the citation work.

GROUNDING RULES (these are non-negotiable — lawyers will not trust your output otherwise):
1. If the user prompt includes a "GROUNDING SOURCES" block with [E1] [E2] ... entries:
   - EVERY factual or legal claim in your output must end with a source marker like [E1] or [E2].
   - You may only cite cases, statutes, or Sections that are explicitly in the GROUNDING SOURCES or in the input CONTEXT.
   - If a claim cannot be grounded, write "(not in corpus)" at the end of that sentence — do NOT invent.
   - Indian-law landmark cases (Kesavananda Bharati, Maneka Gandhi, Vishaka, Puttaswamy, Sanjay Chandra, Niranjan Shankar Golikari, Hakam Singh) may be cited without [E*] markers but only when directly relevant.
2. If no GROUNDING SOURCES block is provided, restrict yourself strictly to the input CONTEXT — paraphrase, summarise, extract, classify, but DO NOT introduce facts the context does not contain.

STYLE RULES:
3. Use formal legal English appropriate for Indian courts.
4. Follow Indian court formatting conventions (cause-title for petitions, "Re:" lines for notices, sworn-affidavit endings, etc.).
5. Be concise — Indian benches value precision over verbosity.
6. Where the input has [PLACEHOLDER] text, fill it from CONTEXT or leave as-is — do not fabricate values.
7. Do not use AI-style hedging ("I think", "I believe", "as an AI", "in my opinion"). Speak as counsel.
8. End pleadings with proper legal language ("respectfully prays", "humbly submits") where appropriate.

OUTPUT FORMAT:
9. If the user prompt asks for a JSON / table / bullet / ranked-list format, produce exactly that. Do not wrap a table inside a paragraph and do not narrate the table afterward.
10. No preamble like "As an expert lawyer, I will now..." — start with the answer.
"""

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
    """Write a complete section based on instruction.

    Note: context is allowed up to 8K chars (was 400). Workflows pass the full
    contract / FIR / matter brief as context, so a hard 400-char cap was
    truncating the input before the LLM ever saw it. 8K covers ~2K tokens of
    context plus the instruction — comfortably within Gemini Flash's 128K
    window with room left for the response.
    """
    doc_label = DOC_TYPES.get(doc_type, {}).get("label", "Legal Document")
    prompt = f"""Document type: {doc_label}
Document context:
{context[:8000] if context else 'Not provided'}

Task: {instruction}"""
    try:
        resp = router.generate(_EDITOR_SYSTEM, prompt, temperature=0.25, max_tokens=1500)
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


# ── Legal Clause Library ──────────────────────────────────────────────────
# Ready-made clauses that lawyers insert frequently. Google Docs/Word have
# no equivalent — lawyers copy-paste from old files. This is the killer feature.

LEGAL_CLAUSES = {
    "prayer_general": {
        "category": "Prayer",
        "label": "General Prayer Clause",
        "text": "It is, therefore, most respectfully prayed that this Hon'ble Court may be pleased to:\n\n(a) [Primary relief sought];\n(b) [Alternative relief];\n(c) Pass such other and further order(s) as this Hon'ble Court may deem fit and proper in the facts and circumstances of the case.\n\nAnd for this act of kindness the petitioner/applicant shall ever pray.",
    },
    "prayer_bail": {
        "category": "Prayer",
        "label": "Bail Prayer",
        "text": "It is, therefore, most respectfully prayed that this Hon'ble Court may be pleased to:\n\n(a) Enlarge the applicant on bail in connection with Case/FIR No. [NUMBER] registered at P.S. [POLICE STATION];\n(b) Impose such conditions as this Hon'ble Court may deem fit;\n(c) Pass any other order(s) in the interest of justice.\n\nAnd for this act of kindness the applicant shall ever pray.",
    },
    "prayer_writ": {
        "category": "Prayer",
        "label": "Writ Petition Prayer",
        "text": "It is, therefore, most respectfully prayed that this Hon'ble Court may be pleased to:\n\n(a) Issue a Writ of [Mandamus/Certiorari/Quo Warranto/Prohibition/Habeas Corpus];\n(b) Declare the impugned order/action dated [DATE] as null and void, being violative of Article [ARTICLE] of the Constitution;\n(c) Grant interim relief by way of stay of the impugned order;\n(d) Award costs of this petition;\n(e) Pass such other order(s) as this Hon'ble Court may deem fit and proper.",
    },
    "undertaking_bail": {
        "category": "Undertaking",
        "label": "Bail Undertaking",
        "text": "The applicant hereby undertakes:\n\n(i) To cooperate with the investigation and appear before the Investigating Officer as and when required;\n(ii) Not to tamper with evidence or influence/threaten witnesses;\n(iii) Not to leave the jurisdiction of this Hon'ble Court without prior permission;\n(iv) To furnish bail bonds/surety as directed by this Hon'ble Court;\n(v) To mark attendance at the concerned police station as per the directions of this Hon'ble Court.",
    },
    "verification": {
        "category": "Verification",
        "label": "Standard Verification",
        "text": "VERIFICATION\n\nI, [NAME], [S/o/D/o/W/o] [FATHER/HUSBAND NAME], aged [AGE] years, [OCCUPATION], resident of [ADDRESS], do hereby verify that the contents of the above [DOCUMENT TYPE] are true and correct to the best of my knowledge and belief.\n\nVerified at [PLACE] on this [DAY] day of [MONTH], [YEAR].\n\n\nDEPONENT",
    },
    "affidavit_header": {
        "category": "Affidavit",
        "label": "Affidavit Opening",
        "text": "AFFIDAVIT\n\nI, [NAME], [S/o/D/o/W/o] [FATHER/HUSBAND NAME], aged about [AGE] years, [OCCUPATION], resident of [FULL ADDRESS], do hereby solemnly affirm and state on oath as under:\n\n1. That I am the [deponent/applicant/respondent] in the above matter and am fully conversant with the facts and circumstances of the case.\n\n2. That the statements made herein are true to my knowledge and belief and nothing material has been concealed.",
    },
    "non_joinder": {
        "category": "Legal Clause",
        "label": "Non-Joinder of Necessary Party",
        "text": "It is submitted that no necessary party has been left unimpleaded. All persons who are necessary and proper parties to the present proceedings have been made parties hereto. The non-joinder of any party, if any, does not affect the merits of the case.",
    },
    "limitation_clause": {
        "category": "Legal Clause",
        "label": "Limitation Period Compliance",
        "text": "The present [petition/application/suit] is within the period of limitation prescribed under [Section/Article] of the Limitation Act, 1963. The cause of action arose on [DATE] and the present proceedings have been filed within [PERIOD] therefrom. No part of the cause of action is barred by limitation.",
    },
    "cause_of_action": {
        "category": "Legal Clause",
        "label": "Cause of Action",
        "text": "CAUSE OF ACTION:\n\nThe cause of action for filing the present [petition/suit/application] arose on [DATE] when [EVENT], and continues to subsist. The cause of action arose within the territorial jurisdiction of this Hon'ble Court as [REASON FOR JURISDICTION].",
    },
    "jurisdiction_clause": {
        "category": "Legal Clause",
        "label": "Territorial Jurisdiction",
        "text": "JURISDICTION:\n\nThis Hon'ble Court has territorial jurisdiction to entertain and try the present [suit/petition/application] as the cause of action has arisen within the jurisdiction of this Hon'ble Court, the [defendant/respondent] resides/carries on business within the jurisdiction, and the subject matter of the dispute is situated within the jurisdiction of this Hon'ble Court.",
    },
    "no_other_remedy": {
        "category": "Legal Clause",
        "label": "No Other Remedy Available",
        "text": "The [petitioner/applicant] submits that no other equally efficacious alternative remedy is available except to approach this Hon'ble Court under [Article/Section]. The petitioner has no other adequate remedy in the ordinary course of law.",
    },
    "interim_relief": {
        "category": "Legal Clause",
        "label": "Interim Relief / Stay",
        "text": "APPLICATION FOR INTERIM RELIEF / STAY:\n\nPending the hearing and final disposal of this [petition/suit], it is prayed that this Hon'ble Court may be pleased to:\n\n(a) Stay the operation/implementation of the impugned order dated [DATE];\n(b) Direct the respondent(s) to maintain status quo;\n(c) Pass such interim order(s) as this Hon'ble Court may deem fit.\n\nGROUNDS FOR INTERIM RELIEF:\n(i) Prima facie case exists in favour of the applicant;\n(ii) Irreparable injury will be caused if interim relief is not granted;\n(iii) Balance of convenience lies in favour of the applicant.",
    },
    "res_judicata": {
        "category": "Legal Clause",
        "label": "Res Judicata / No Previous Filing",
        "text": "The [petitioner/applicant] states that no previous petition or application has been filed by the petitioner before this Hon'ble Court or any other Court of competent jurisdiction on the same cause of action. The matter is not barred by the principles of res judicata or constructive res judicata.",
    },
    "vakalatnama_text": {
        "category": "Format",
        "label": "Vakalatnama",
        "text": "VAKALATNAMA\n\nI/We [CLIENT NAME], [S/o/D/o/W/o] [PARENT NAME], resident of [ADDRESS], do hereby appoint and authorize [ADVOCATE NAME], Advocate, bearing Enrollment No. [BAR COUNCIL NO.], to appear, plead and act on my/our behalf in [CASE DESCRIPTION] before [COURT NAME] and to do all acts, deeds and things as may be necessary in connection with the above matter.\n\nI/We agree to ratify all acts done by the said Advocate.\n\nDated: [DATE]\nPlace: [PLACE]\n\n\n[Signature of Client]\n[Name of Client]\n\nAccepted:\n[Signature of Advocate]\n[Name of Advocate]\nEnrollment No.: [NUMBER]",
    },
    "memo_of_parties": {
        "category": "Format",
        "label": "Memorandum of Parties",
        "text": "MEMORANDUM OF PARTIES\n\n1. [NAME], [S/o/D/o/W/o] [PARENT],\n   Aged [AGE] years, [OCCUPATION],\n   R/o [ADDRESS]\n   ... Petitioner/Plaintiff No. 1\n\nVERSUS\n\n1. [NAME], [DESIGNATION/OCCUPATION],\n   [OFFICE ADDRESS]\n   ... Respondent/Defendant No. 1\n\n2. [NAME], [DESIGNATION/OCCUPATION],\n   [OFFICE ADDRESS]\n   ... Respondent/Defendant No. 2",
    },
    "court_fee_valuation": {
        "category": "Format",
        "label": "Court Fee & Valuation",
        "text": "COURT FEE AND VALUATION:\n\nThe present suit/petition is valued at Rs. [AMOUNT]/- for the purpose of court fee and jurisdiction.\n\nCourt fee of Rs. [FEE AMOUNT]/- has been affixed on the plaint/petition as per the provisions of the Court Fees Act, 1870 / [STATE] Court Fees Act.",
    },
}


def list_legal_clauses() -> list[dict[str, str]]:
    """Return all legal clauses grouped by category."""
    return [
        {"id": k, "category": v["category"], "label": v["label"]}
        for k, v in LEGAL_CLAUSES.items()
    ]


def get_legal_clause(clause_id: str) -> dict[str, str] | None:
    """Return a specific clause by ID."""
    c = LEGAL_CLAUSES.get(clause_id)
    if not c:
        return None
    return {"id": clause_id, **c}


def ai_generate_cause_title(
    case_type: str, petitioner: str, respondent: str,
    court: str, case_no: str = "", year: str = ""
) -> str:
    """Generate a properly formatted cause title for Indian courts."""
    year_str = year or "[YEAR]"
    case_no_str = case_no or "_____ of " + year_str

    if "criminal" in case_type.lower() or "bail" in case_type.lower():
        prefix = "CRIMINAL MISC. APPLICATION NO."
    elif "writ" in case_type.lower():
        prefix = "WRIT PETITION (CIVIL) NO."
    elif "appeal" in case_type.lower():
        prefix = "CIVIL/CRIMINAL APPEAL NO."
    else:
        prefix = "CIVIL SUIT NO."

    return f"""IN THE HON'BLE {court.upper()}

{prefix} {case_no_str}

IN THE MATTER OF:

{petitioner}
... Petitioner/Applicant

VERSUS

{respondent}
... Respondent"""
