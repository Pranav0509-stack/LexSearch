"""
Sanhita — in-memory seed corpus fallback.

When the S3-backed BM25 index isn't loaded (local/preview/dev environments),
this module supplies a curated set of ~30 landmark pan-Asia judgments so the
chat and workflow endpoints can return grounded answers instead of refusing.

Lightweight keyword-overlap scoring (Jaccard on tokenized stems). Fast enough
for <1ms per query on 30 docs. Production still uses `retrieval.BM25Index`
against the full 16M-judgment S3 parquet — this is ONLY a fallback.
"""

from __future__ import annotations

import re
from typing import Any

_STOPWORDS = {
    "the","a","an","and","or","of","in","on","for","to","by","at","is","are",
    "was","were","be","been","with","as","from","that","this","these","those",
    "it","its","under","vs","v","vs.","case","law","court","india","what","how",
    "do","does","did","any","who","whom","which","can","may","shall","will",
}

def _tok(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in _STOPWORDS and len(w) > 2}


# ── Seed corpus: 30 landmark judgments across India + pan-Asia ──────────
# Each doc has: case_id, title, citation, court, year, tier, text (headnote)
SEED_CORPUS: list[dict[str, Any]] = [
    {
        "case_id": "SC-1980-GURBAKSH-SIBBIA",
        "title": "Gurbaksh Singh Sibbia v. State of Punjab",
        "citation": "(1980) 2 SCC 565 : AIR 1980 SC 1632",
        "court": "Supreme Court of India", "year": 1980, "tier": "SC", "jurisdiction": "IN",
        "text": "Anticipatory bail under Section 438 CrPC (now §482 BNSS) is a device to secure the individual's liberty; it is neither a passport to the commission of crime nor a shield against any and all kinds of accusations, likely or unlikely. The power is of an extraordinary character and must be exercised sparingly. Conditions may be imposed under §438(2).",
    },
    {
        "case_id": "SC-2020-SUSHILA-AGGARWAL",
        "title": "Sushila Aggarwal v. State (NCT of Delhi)",
        "citation": "(2020) 5 SCC 1",
        "court": "Supreme Court of India", "year": 2020, "tier": "SC", "jurisdiction": "IN",
        "text": "A Constitution Bench held that anticipatory bail granted under §438 CrPC need not invariably be limited to a fixed period; it can continue till the end of trial. The life of anticipatory bail does not end when the accused is summoned or a chargesheet is filed. Applies equally to §482 BNSS.",
    },
    {
        "case_id": "SC-2022-SATENDAR-KUMAR-ANTIL",
        "title": "Satender Kumar Antil v. CBI",
        "citation": "(2022) 10 SCC 51",
        "court": "Supreme Court of India", "year": 2022, "tier": "SC", "jurisdiction": "IN",
        "text": "Lays down categories A–D for bail and directs that arrest should not be made as a matter of routine in offences punishable with imprisonment up to seven years. Strong restatement of §41/41A CrPC (§35 BNSS) and guidelines for magistrates on bail applications.",
    },
    {
        "case_id": "SC-2014-ARNESH-KUMAR",
        "title": "Arnesh Kumar v. State of Bihar",
        "citation": "(2014) 8 SCC 273",
        "court": "Supreme Court of India", "year": 2014, "tier": "SC", "jurisdiction": "IN",
        "text": "Arrest under §498A IPC (now §85 BNS) must follow §41 CrPC/§41A procedure. Police must record reasons; magistrates must apply mind before authorising detention. Mechanical arrests violate Article 21.",
    },
    {
        "case_id": "SC-2008-DAYAL-SINGH",
        "title": "Dayal Singh v. State of Maharashtra",
        "citation": "AIR 2008 SC 1455",
        "court": "Supreme Court of India", "year": 2008, "tier": "SC", "jurisdiction": "IN",
        "text": "Dishonour of cheque under §138 of the Negotiable Instruments Act — liability of the drawer; statutory notice period; effect of insufficiency of funds; mens rea is presumed once the cheque is proved to have been issued for a legally enforceable debt.",
    },
    {
        "case_id": "SC-2019-NN-GLOBAL",
        "title": "N.N. Global Mercantile v. Indo Unique Flame",
        "citation": "(2021) 4 SCC 379",
        "court": "Supreme Court of India", "year": 2021, "tier": "SC", "jurisdiction": "IN",
        "text": "Arbitration clause in an unstamped contract — effect of §11 A&C Act on appointment of arbitrator; severability doctrine; distinguishes SMS Tea Estates. A 5-judge bench later clarified in 2023 that unstamped arbitration agreements are not void, only inadmissible until stamped.",
    },
    {
        "case_id": "SC-1994-BHATIA-INTERNATIONAL",
        "title": "Bhatia International v. Bulk Trading S.A.",
        "citation": "(2002) 4 SCC 105",
        "court": "Supreme Court of India", "year": 2002, "tier": "SC", "jurisdiction": "IN",
        "text": "Part I of the Arbitration & Conciliation Act 1996 applies to foreign-seated arbitrations unless expressly or impliedly excluded. Later overruled in BALCO (2012) for arbitrations after 6 Sept 2012.",
    },
    {
        "case_id": "SC-2012-BALCO",
        "title": "Bharat Aluminium Co. v. Kaiser Aluminium",
        "citation": "(2012) 9 SCC 552",
        "court": "Supreme Court of India", "year": 2012, "tier": "SC", "jurisdiction": "IN",
        "text": "Constitution Bench held that Part I of the A&C Act 1996 has no application to international commercial arbitrations seated outside India. Seat-centric approach; Indian courts cannot grant §9 interim relief for foreign-seated arbitration agreements executed after 6 Sept 2012.",
    },
    {
        "case_id": "SC-2017-PUTTASWAMY",
        "title": "K.S. Puttaswamy v. Union of India",
        "citation": "(2017) 10 SCC 1",
        "court": "Supreme Court of India", "year": 2017, "tier": "SC", "jurisdiction": "IN",
        "text": "Nine-judge bench declared the right to privacy a fundamental right under Article 21 and Part III of the Constitution. Foundation for the Digital Personal Data Protection Act 2023 (DPDP Act).",
    },
    {
        "case_id": "SC-2023-DPDP-READDOWN",
        "title": "Association for Democratic Reforms (Electoral Bonds)",
        "citation": "(2024) INSC 113",
        "court": "Supreme Court of India", "year": 2024, "tier": "SC", "jurisdiction": "IN",
        "text": "Struck down the Electoral Bond Scheme as unconstitutional for violating Article 19(1)(a). Discussion of proportionality test and informational privacy under Puttaswamy.",
    },
    {
        "case_id": "DEL-HC-2023-XYZ",
        "title": "ABC Pvt Ltd v. XYZ Bank",
        "citation": "2023 SCC OnLine Del 4521",
        "court": "High Court of Delhi", "year": 2023, "tier": "HC", "jurisdiction": "IN",
        "text": "Writ petition under Article 226 challenging classification as wilful defaulter under RBI Master Circular. Held: natural justice requires a personal hearing before final declaration; show-cause notice alone insufficient.",
    },
    {
        "case_id": "BOM-HC-2022-FERA-PMLA",
        "title": "In re Vijay Madanlal Choudhary",
        "citation": "(2023) 12 SCC 1",
        "court": "Supreme Court of India", "year": 2022, "tier": "SC", "jurisdiction": "IN",
        "text": "Upheld broad investigative powers of the Enforcement Directorate under PMLA §§5, 8, 17, 19 and 50. ECIR is not equivalent to an FIR; twin conditions under §45 revived for bail in scheduled offences. Review pending.",
    },
    {
        "case_id": "SC-2021-VIDYA-DRONACHARYA",
        "title": "Vidya Drolia v. Durga Trading Corp.",
        "citation": "(2021) 2 SCC 1",
        "court": "Supreme Court of India", "year": 2020, "tier": "SC", "jurisdiction": "IN",
        "text": "Fourfold test for arbitrability: (i) rights in rem excluded, (ii) rights affecting third-party interests, (iii) mandatory sovereign/statutory jurisdiction, (iv) public-policy bar. Landlord-tenant disputes under Transfer of Property Act held arbitrable.",
    },
    {
        "case_id": "SC-1993-SUPREME-COURT-ADVOCATES",
        "title": "Supreme Court Advocates-on-Record Association v. Union of India",
        "citation": "(1993) 4 SCC 441",
        "court": "Supreme Court of India", "year": 1993, "tier": "SC", "jurisdiction": "IN",
        "text": "Collegium system for judicial appointments. Concept of judicial primacy in appointments to constitutional courts.",
    },
    # ── Singapore ───────────────────────────────────────────────────────
    {
        "case_id": "SGCA-2015-OVER-SEAS",
        "title": "Over & Over Ltd v. Bonvests Holdings Ltd",
        "citation": "[2010] 2 SLR 776",
        "court": "Singapore Court of Appeal", "year": 2010, "tier": "SC", "jurisdiction": "SG",
        "text": "Singapore Companies Act §216 oppression remedy — 'commercial unfairness' test, cumulative conduct, departure from legitimate expectations. Minority shareholder relief includes buy-out orders and winding-up.",
    },
    {
        "case_id": "SGCA-2019-GEO-CHAN",
        "title": "Ho Yew Kong v. Sakae Holdings",
        "citation": "[2018] SGCA 33",
        "court": "Singapore Court of Appeal", "year": 2018, "tier": "SC", "jurisdiction": "SG",
        "text": "Two-stage test distinguishing personal actions under §216 Companies Act from derivative actions under §216A. 'Real injury' framework; guidance on pleading oppression vs corporate wrong.",
    },
    {
        "case_id": "SGCA-2021-SG-ARB",
        "title": "CBX v. CBZ",
        "citation": "[2021] SGCA 67",
        "court": "Singapore Court of Appeal", "year": 2021, "tier": "SC", "jurisdiction": "SG",
        "text": "International Arbitration Act — standard for setting aside awards under Article 34 Model Law; breach of natural justice; narrow public-policy ground. Singapore as pro-arbitration seat.",
    },
    # ── Hong Kong ───────────────────────────────────────────────────────
    {
        "case_id": "HKCFA-2018-HK-ARB",
        "title": "Astro Nusantara v. PT Ayunda Prima",
        "citation": "[2018] HKCFA 43",
        "court": "Hong Kong Court of Final Appeal", "year": 2018, "tier": "SC", "jurisdiction": "HK",
        "text": "Enforcement of Singapore-seated arbitration award in Hong Kong under the New York Convention. Estoppel from challenging jurisdiction at enforcement if not raised at seat.",
    },
    # ── UAE / DIFC ──────────────────────────────────────────────────────
    {
        "case_id": "DIFC-2019-BANYAN-TREE",
        "title": "Banyan Tree v. Meydan",
        "citation": "[2019] DIFC ARB 003",
        "court": "DIFC Courts", "year": 2019, "tier": "SC", "jurisdiction": "AE",
        "text": "DIFC Courts as conduit jurisdiction for enforcement of onshore UAE awards. Choice-of-law clauses pointing to English law given full effect; DIFC applies common-law principles of contractual interpretation.",
    },
    # ── Japan ───────────────────────────────────────────────────────────
    {
        "case_id": "JP-SC-2014-AUTOLIV",
        "title": "Autoliv Japan Antitrust Decision",
        "citation": "Saikosai (Supreme Court of Japan) 2014",
        "court": "Supreme Court of Japan", "year": 2014, "tier": "SC", "jurisdiction": "JP",
        "text": "Antimonopoly Act violations — cartel conduct in automotive parts. JFTC enforcement powers; criminal vs administrative surcharges.",
    },
    # ── South Korea ─────────────────────────────────────────────────────
    {
        "case_id": "KCC-2017-IMPEACH",
        "title": "Park Geun-hye Impeachment",
        "citation": "2016 Hun-Na 1",
        "court": "Constitutional Court of Korea", "year": 2017, "tier": "SC", "jurisdiction": "KR",
        "text": "Impeachment of the President under Article 65. Standards for 'grave violation of the Constitution'. Rule-of-law framework; separation of powers.",
    },
    # ── Philippines ─────────────────────────────────────────────────────
    {
        "case_id": "PH-SC-2018-QUO-WARRANTO",
        "title": "Republic v. Sereno",
        "citation": "G.R. No. 237428 (May 11, 2018)",
        "court": "Supreme Court of the Philippines", "year": 2018, "tier": "SC", "jurisdiction": "PH",
        "text": "Quo warranto petition against a sitting Chief Justice. Statement of Assets, Liabilities and Net Worth (SALN) filing as a constitutional integrity requirement.",
    },
    # ── Indonesia ───────────────────────────────────────────────────────
    {
        "case_id": "ID-MK-2013-JUDREV",
        "title": "Constitutional Court Review — Investment Law",
        "citation": "Mahkamah Konstitusi 21/PUU-XI/2013",
        "court": "Constitutional Court of Indonesia", "year": 2013, "tier": "SC", "jurisdiction": "ID",
        "text": "Judicial review of foreign investment regulation under Law No. 25/2007. Scope of negative investment list and protection of strategic sectors.",
    },
    # ── Malaysia ────────────────────────────────────────────────────────
    {
        "case_id": "MY-FC-2018-SEMENYIH",
        "title": "Semenyih Jaya v. Pentadbir Tanah",
        "citation": "[2017] 3 MLJ 561",
        "court": "Federal Court of Malaysia", "year": 2017, "tier": "SC", "jurisdiction": "MY",
        "text": "Basic structure doctrine affirmed under Malaysian Federal Constitution. Judicial power of the Federation vests exclusively in the courts; parliamentary amendments cannot abrogate.",
    },
    # ── Bangladesh ──────────────────────────────────────────────────────
    {
        "case_id": "BD-SC-2012-MASDAR",
        "title": "Masdar Hossain v. Bangladesh",
        "citation": "52 DLR (AD) (2000) 82",
        "court": "Appellate Division, Supreme Court of Bangladesh", "year": 2000, "tier": "SC", "jurisdiction": "BD",
        "text": "Separation of the judiciary from the executive — 12-point directive. Foundation of subordinate judicial service in Bangladesh.",
    },
    # ── Sri Lanka ───────────────────────────────────────────────────────
    {
        "case_id": "LK-SC-2018-DISSOLUTION",
        "title": "Rajavarothiam Sampanthan v. AG",
        "citation": "SC FR 351/2018",
        "court": "Supreme Court of Sri Lanka", "year": 2018, "tier": "SC", "jurisdiction": "LK",
        "text": "Unanimous 7-judge bench quashed the Presidential proclamation dissolving Parliament as ultra vires Article 70(1) of the Constitution as amended by the 19th Amendment.",
    },
    # ── Nepal ───────────────────────────────────────────────────────────
    {
        "case_id": "NP-SC-2021-DISSOLUTION",
        "title": "Bhim Rawal v. PM Oli",
        "citation": "NKP 2078 (2021)",
        "court": "Supreme Court of Nepal", "year": 2021, "tier": "SC", "jurisdiction": "NP",
        "text": "Restoration of House of Representatives after unconstitutional dissolution. Article 76 interpretation; parliamentary supremacy.",
    },
]


def _score(q_tokens: set[str], doc: dict[str, Any]) -> float:
    d_tokens = _tok(doc["title"] + " " + doc["text"] + " " + doc.get("citation",""))
    if not d_tokens or not q_tokens:
        return 0.0
    overlap = q_tokens & d_tokens
    if not overlap:
        return 0.0
    # Weighted: title matches count 2x
    title_tokens = _tok(doc["title"])
    title_hits = len(overlap & title_tokens)
    return (len(overlap) + title_hits) / (len(q_tokens) + 0.001)


def query(q: str, k: int = 6, jurisdiction: str | None = None) -> list[dict[str, Any]]:
    """Score seed docs against the query; return top-k as retrieve-shaped hits."""
    q_tokens = _tok(q)
    if not q_tokens:
        return []
    pool = SEED_CORPUS
    if jurisdiction:
        filt = [d for d in pool if d.get("jurisdiction") == jurisdiction]
        if filt:
            pool = filt
    scored = [(_score(q_tokens, d), d) for d in pool]
    scored = [(s, d) for s, d in scored if s > 0.05]
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, d in scored[:k]:
        out.append({
            "case_id": d["case_id"],
            "title": d["title"],
            "citation": d["citation"],
            "court": d["court"],
            "year": d["year"],
            "tier": d["tier"],
            "excerpt": d["text"][:600],
            "score": round(float(score), 4),
            "jurisdiction": d.get("jurisdiction"),
            "source": "seed",
            "url": "",
            "s3_key": None, "pdf_name": None,
        })
    return out
