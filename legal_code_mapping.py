"""
Legal Code Mapping — IPC ↔ BNS · CrPC ↔ BNSS · IEA ↔ BSA
========================================================

The three new criminal codes — Bharatiya Nyaya Sanhita 2023 (BNS),
Bharatiya Nagarik Suraksha Sanhita 2023 (BNSS), and Bharatiya Sakshya
Adhiniyam 2023 (BSA) — came into force on 1 July 2024 and replaced
the IPC 1860, CrPC 1973, and Indian Evidence Act 1872 respectively.

Old offences pending on 1 July 2024 continue under the OLD code; new
FIRs from that date are under the NEW codes. Every Sanhita-product
surface that mentions an old section MUST also show the new equivalent
so lawyers can identify the correct provision when drafting / arguing.

This module is the single source of truth. Both the Python backend
(compliance plugins, doc_editor prompts, server.py renderers) and the
TypeScript frontend (web/src/lib/legal-code-mapping.ts mirror) source
their mappings from this one place.

References:
  · BNS    — https://www.indiacode.nic.in/handle/123456789/20062
  · BNSS   — https://www.indiacode.nic.in/handle/123456789/20063
  · BSA    — https://www.indiacode.nic.in/handle/123456789/20064
  · Ministry of Home Affairs section-by-section comparison tables
    https://www.mha.gov.in/sites/default/files/250708.pdf
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ── IPC → BNS section map ─────────────────────────────────────────────────
#
# Coverage: every section in IPC most commonly cited in litigation
# practice. Not every IPC section maps 1-to-1 — some have been merged,
# split, or repealed. Where a section's number changes, the new BNS
# number is listed. Where the offence has been re-coded with a new
# scope, the new BNS number + a short note appears.
IPC_TO_BNS: dict[str, str] = {
    # General — Chapters I-IV
    "120A": "61(1)",  # criminal conspiracy
    "120B": "61(2)",  # punishment for conspiracy
    "121": "147",     # waging war against Govt of India
    "124A": "152",    # (sedition rebranded as "act endangering sovereignty, unity and integrity of India")
    "141": "189(1)",  # unlawful assembly
    "143": "189(2)",  # punishment for unlawful assembly
    "144": "189(3)",  # joining unlawful assembly armed with deadly weapon
    "146": "190",     # rioting
    "147": "191(2)",  # punishment for rioting
    "148": "191(3)",  # rioting armed with deadly weapon
    # Offences against State
    "153A": "196",    # promoting enmity between groups
    "153B": "197",    # imputations prejudicial to national integration
    "172": "206",     # absconding to avoid summons
    "182": "217",     # false information to public servant
    "186": "221",     # obstructing public servant
    "188": "223",     # disobedience to order
    # Murder / culpable homicide — re-coded with new numbers
    "299": "100",     # culpable homicide (definition)
    "300": "101",     # murder (definition)
    "302": "103",     # punishment for murder
    "304": "105",     # punishment for culpable homicide not amounting to murder
    "304A": "106",    # causing death by negligence
    "304B": "80",     # dowry death
    "306": "108",     # abetment of suicide
    "307": "109",     # attempt to murder
    "308": "110",     # attempt to commit culpable homicide
    # Hurt
    "319": "114",     # hurt (definition)
    "320": "116(1)",  # grievous hurt
    "321": "115",     # voluntarily causing hurt
    "323": "115(2)",  # punishment for voluntarily causing hurt
    "324": "118(1)",  # voluntarily causing hurt by dangerous weapons
    "325": "117(2)",  # voluntarily causing grievous hurt
    "326": "118(2)",  # voluntarily causing grievous hurt by dangerous weapons
    "326A": "124(1)", # acid attack
    "326B": "124(2)", # attempt to throw acid
    # Wrongful restraint / confinement
    "339": "126",
    "340": "127(1)",
    "341": "126(2)",
    "342": "127(2)",
    # Force / assault
    "351": "130",
    "352": "131",
    "354": "74",      # outraging modesty of woman
    "354A": "75",     # sexual harassment
    "354B": "76",     # assault with intent to disrobe
    "354C": "77",     # voyeurism
    "354D": "78",     # stalking
    # Kidnapping / abduction
    "359": "136",
    "360": "138",
    "361": "137",
    "362": "139",
    "363": "139(2)",  # punishment for kidnapping
    "364A": "140(4)", # kidnapping for ransom
    "365": "139(3)",
    "366": "87",      # kidnapping woman to compel marriage
    # Sexual offences — re-coded under BNS Chapter V
    "375": "63",      # rape (definition)
    "376": "64",      # punishment for rape
    "376A": "65(2)",  # rape causing death / vegetative state
    "376AB": "65(2)", # rape on woman under 12 years
    "376B": "67",     # sexual intercourse with wife during separation
    "376C": "68",     # sexual intercourse by person in authority
    "376D": "70(1)",  # gang rape
    "376DA": "70(2)", # gang rape on woman under 16
    "376DB": "70(2)", # gang rape on woman under 12
    "376E": "71",     # repeat offender — rape
    "377": "(repealed — consensual same-sex acts decriminalised per Navtej Singh Johar; non-consensual covered by §63 BNS)",
    # Marriage
    "493": "82",
    "494": "82(2)",   # bigamy
    "495": "82(3)",   # bigamy with concealment
    "497": "(repealed — adultery decriminalised by Joseph Shine v UoI 2018; not in BNS)",
    "498A": "85",     # cruelty by husband / relatives
    # Defamation
    "499": "356(1)",
    "500": "356(2)",  # punishment for defamation
    "501": "356(3)",
    "502": "356(4)",
    # Theft / extortion / robbery / dacoity
    "378": "303(1)",
    "379": "303(2)",  # punishment for theft
    "380": "305",     # theft in dwelling
    "381": "306",     # theft by clerk / servant
    "383": "308(1)",  # extortion (definition)
    "384": "308(2)",  # punishment for extortion
    "390": "309(1)",  # robbery (definition)
    "392": "309(4)",  # punishment for robbery
    "395": "310(2)",  # dacoity
    "396": "310(3)",  # dacoity with murder
    "397": "311",     # robbery / dacoity with deadly weapon
    # Criminal breach of trust
    "405": "316(1)",
    "406": "316(2)",  # punishment for CBT
    "409": "316(5)",  # CBT by public servant / banker
    # Cheating
    "415": "318(1)",  # cheating (definition)
    "417": "318(2)",  # simple cheating
    "418": "318(3)",  # cheating with knowledge of wrongful loss
    "420": "318(4)",  # cheating + dishonestly inducing delivery
    # Forgery
    "463": "336(1)",
    "464": "335",
    "465": "336(2)",  # punishment for forgery
    "467": "338",     # forgery of valuable security
    "468": "336(3)",  # forgery for cheating
    "471": "340",     # using as genuine a forged document
    "474": "342",
    # Mischief
    "425": "324(1)",
    "426": "324(2)",  # punishment for mischief
    # Trespass
    "441": "329(1)",
    "442": "329(2)",
    "447": "329(3)",  # punishment for criminal trespass
    "448": "329(4)",  # punishment for house-trespass
    "452": "331(4)",  # house-trespass after preparation for hurt
    # Public servants / contempt
    "499": "356(1)",  # already listed; defamation
    # Abetment
    "107": "45",
    "108": "46",
    "109": "49",
    "110": "50",
    "115": "55",
    "120": "60",
    # Attempt
    "511": "62",
}


# ── CrPC → BNSS section map ───────────────────────────────────────────────
CRPC_TO_BNSS: dict[str, str] = {
    # Information to police / arrest
    "41": "35",       # when police may arrest without warrant
    "41A": "35(3)",   # notice of appearance
    "41B": "36",      # arrest procedure
    "41C": "37",      # control room / arrest records
    "50": "47",       # arrested person's rights
    "50A": "48",      # notifying family of arrest
    "57": "58",       # produce before magistrate within 24 hrs
    # Investigation
    "154": "173",     # FIR for cognizable offence
    "155": "174",     # info in non-cognizable case
    "156": "175",     # police's power to investigate cognizable offences
    "156(3)": "175(3)", # magistrate's power to direct investigation
    "161": "180",     # statements during investigation
    "164": "183",     # confession before magistrate
    "164A": "184",    # medical exam of rape victim
    "167": "187",     # remand procedure (90/60 day limit + default bail)
    "173": "193",     # final report (chargesheet)
    "176": "196",     # magisterial inquiry into custodial death
    # Trial / bail / appeal
    "190": "210",     # cognizance by magistrate
    "200": "223",     # examination of complainant
    "202": "225",     # postponement of issue of process
    "204": "227",     # issue of process
    "205": "228",     # dispensing with personal attendance
    "227": "250",     # discharge in sessions case
    "228": "251",     # framing of charge in sessions case
    "239": "262",     # discharge in warrant case (on police report)
    "240": "263",     # framing of charge in warrant case
    "245": "268",     # discharge in warrant case (other than police report)
    "246": "269",     # framing of charge
    "313": "351",     # examination of accused
    "319": "358",     # power to proceed against persons appearing guilty
    "320": "359",     # compounding of offences
    "356": "395",     # judgement of competent magistrate
    "374": "415",     # appeals from convictions
    "397": "438",     # revisional powers — calling for record
    "401": "442",     # High Court's powers of revision
    "436": "478",     # bail in bailable offences
    "437": "480",     # bail in non-bailable offences
    "438": "482",     # anticipatory bail
    "439": "483",     # special powers of HC / SC re bail
    "446A": "490",    # cancellation of bond
    "482": "528",     # inherent powers of High Court (quashing)
}


# ── Indian Evidence Act → Bharatiya Sakshya Adhiniyam section map ─────────
IEA_TO_BSA: dict[str, str] = {
    # Definitions / preliminary
    "1":  "1",
    "2":  "2",
    "3":  "2(1)(a)-(n)",    # definitions consolidated
    "4":  "3",
    # Relevancy
    "5":  "4",
    "6":  "4",              # res gestae
    "7":  "5",              # occasion / cause / effect
    "8":  "6",              # motive / preparation / conduct
    "9":  "7",
    "10": "8",              # things said / done by conspirator
    "11": "9",              # facts inconsistent / making fact highly probable
    # Admissions / confessions
    "17": "15",
    "18": "16",
    "19": "17",
    "20": "18",
    "21": "19",
    "22": "20",
    "24": "22",             # confession by inducement
    "25": "23(1)",          # confession to police inadmissible
    "26": "23(2)",          # confession in police custody
    "27": "23(2) Proviso",  # how much info from accused admissible
    "29": "26",
    "30": "27",
    "32": "29",             # statements of persons who cannot be called
    # Privilege
    "122": "128",           # communications during marriage
    "123": "129",           # affairs of state
    "126": "132",           # professional communications (advocate-client)
    "127": "133",
    "128": "134",
    "129": "135",
    # Witness / examination
    "118": "124",           # who may testify
    "120": "126",           # parties to civil suit / spouses
    "133": "138",           # accomplice
    "135": "140",           # order of production / examination
    "137": "142",           # examination-in-chief, cross, re-examination
    "138": "143",           # order of examinations
    "141": "146",           # leading questions
    "145": "148",           # cross-examination as to previous statements
    "146": "149",           # questions in cross-examination
    "154": "157",           # questions by party to his own witness
    # Burden of proof
    "101": "104",
    "102": "105",
    "103": "106",
    "104": "107",
    "105": "108",
    "106": "109",
    "112": "115",           # legitimacy of child during marriage
    # Documentary
    "61": "56",             # proof of contents of documents
    "62": "57",             # primary evidence
    "63": "58",             # secondary evidence
    "65": "60",             # when secondary evidence allowed
    "65A": "61",            # electronic evidence — special provisions
    "65B": "63",            # admissibility of electronic record (CERT)
    "67": "67",             # proof of signature / handwriting
    "73": "73",             # comparison of signature / handwriting
    # Presumptions
    "79": "79",
    "80": "80",
    "81": "81",
    "85B": "85",            # presumption as to electronic records
    "90": "90",             # documents 30 years old
    "114": "118",           # court may presume existence of certain facts
    "114A": "119",          # presumption as to absence of consent
}


# ── Helpers ───────────────────────────────────────────────────────────────

# Inverse maps (built lazily) so we can answer "what's the IPC equivalent of
# BNS §103?" — useful when a lawyer pastes a new chargesheet and wants the
# old number for precedent search.
BNS_TO_IPC = {v: k for k, v in IPC_TO_BNS.items() if not v.startswith("(")}
BNSS_TO_CRPC = {v: k for k, v in CRPC_TO_BNSS.items()}
BSA_TO_IEA  = {v: k for k, v in IEA_TO_BSA.items()}


@dataclass
class SectionRef:
    """A single section reference in any of the six legal codes."""
    code: str          # "IPC" | "CrPC" | "IEA" | "BNS" | "BNSS" | "BSA"
    section: str       # "138", "498A", "302" etc.
    old_or_new: str    # "old" if IPC/CrPC/IEA; "new" if BNS/BNSS/BSA

    @property
    def equivalent(self) -> "SectionRef | None":
        """Return the equivalent in the other code (old ↔ new), or None."""
        if self.code == "IPC":
            mapped = IPC_TO_BNS.get(self.section)
            return SectionRef("BNS", mapped, "new") if mapped and not mapped.startswith("(") else None
        if self.code == "CrPC":
            mapped = CRPC_TO_BNSS.get(self.section)
            return SectionRef("BNSS", mapped, "new") if mapped else None
        if self.code == "IEA":
            mapped = IEA_TO_BSA.get(self.section)
            return SectionRef("BSA", mapped, "new") if mapped else None
        if self.code == "BNS":
            mapped = BNS_TO_IPC.get(self.section)
            return SectionRef("IPC", mapped, "old") if mapped else None
        if self.code == "BNSS":
            mapped = BNSS_TO_CRPC.get(self.section)
            return SectionRef("CrPC", mapped, "old") if mapped else None
        if self.code == "BSA":
            mapped = BSA_TO_IEA.get(self.section)
            return SectionRef("IEA", mapped, "old") if mapped else None
        return None

    def display(self) -> str:
        """E.g. 'Section 420 IPC ⇄ Section 318(4) BNS' (with both sides)."""
        eq = self.equivalent
        if eq:
            return f"§{self.section} {self.code} ⇄ §{eq.section} {eq.code}"
        return f"§{self.section} {self.code}"


# Pattern to recognise any section reference in free text:
#   "Section 138 NI Act"       — already-modern, leave alone
#   "Section 420 IPC"          — annotate with BNS equivalent
#   "Sec. 482 CrPC"            — annotate with BNSS equivalent
#   "S. 65B IEA" / "65B Indian Evidence Act" — annotate with BSA
SECTION_PATTERN = re.compile(
    r"(?:section|sec\.?|s\.?)\s*(\d+[A-Z]*(?:\(\d+\))?)\s+"
    r"(?:of\s+(?:the\s+)?)?"
    r"(IPC|CrPC|Cr\.P\.C\.?|Indian Penal Code|Code of Criminal Procedure|"
    r"IEA|Indian Evidence Act|BNS|BNSS|BSA|"
    r"Bharatiya Nyaya Sanhita|Bharatiya Nagarik Suraksha Sanhita|Bharatiya Sakshya Adhiniyam)",
    re.IGNORECASE,
)

_CODE_ALIAS = {
    "ipc": "IPC", "indian penal code": "IPC",
    "crpc": "CrPC", "cr.p.c": "CrPC", "cr.p.c.": "CrPC",
    "code of criminal procedure": "CrPC",
    "iea": "IEA", "indian evidence act": "IEA",
    "bns": "BNS", "bharatiya nyaya sanhita": "BNS",
    "bnss": "BNSS", "bharatiya nagarik suraksha sanhita": "BNSS",
    "bsa": "BSA", "bharatiya sakshya adhiniyam": "BSA",
}


def annotate_text(text: str) -> str:
    """Walk free text; every "Section X CODE" gets the cross-code equivalent
    appended in parentheses. Idempotent — running twice has no further effect.

    Example:
        in:   "Charged under Section 420 IPC and Section 498A IPC."
        out:  "Charged under Section 420 IPC (BNS §318(4)) and Section 498A IPC (BNS §85)."
    """
    if "BNS §" in text and "IPC" in text and "(BNS §" in text:
        return text   # already annotated

    def _repl(m: re.Match[str]) -> str:
        sec = m.group(1)
        code_token = m.group(2).lower()
        code = _CODE_ALIAS.get(code_token, code_token.upper())
        ref = SectionRef(code, sec, "old" if code in ("IPC", "CrPC", "IEA") else "new")
        eq = ref.equivalent
        if not eq:
            return m.group(0)
        # Don't double-annotate
        tail = text[m.end():m.end() + 20].lower()
        if eq.code.lower() in tail:
            return m.group(0)
        return f"{m.group(0)} ({eq.code} §{eq.section})"

    return SECTION_PATTERN.sub(_repl, text)


# Convenient one-liners for callers
def ipc_to_bns(section: str) -> str | None:
    return IPC_TO_BNS.get(section)


def crpc_to_bnss(section: str) -> str | None:
    return CRPC_TO_BNSS.get(section)


def iea_to_bsa(section: str) -> str | None:
    return IEA_TO_BSA.get(section)


# Smoke / CLI
if __name__ == "__main__":
    samples = [
        "Charged under Section 420 IPC and Section 498A IPC.",
        "Bail application under Section 439 CrPC, triple test of Sanjay Chandra.",
        "Confession is inadmissible under Section 25 of the Indian Evidence Act.",
        "FIR registered under Sec. 302 IPC read with 120B IPC.",
        "Section 65B Indian Evidence Act electronic record certification.",
    ]
    for s in samples:
        print(f"  IN : {s}")
        print(f"  OUT: {annotate_text(s)}")
        print()
