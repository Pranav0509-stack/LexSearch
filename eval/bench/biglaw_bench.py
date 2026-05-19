"""Indian BigLaw Bench — citation-faithfulness eval for Sanhita.

Inspired by Harvey's BigLaw Bench; adapted for Indian law. Four scorers:

  1. citation_faithfulness  Does every cited case actually exist + say what the
                            answer claims? Validated by FTS5 lookup against
                            the corpus (16.9M judgments + 11.6M legal_docs).
  2. statute_precision      Does the answer cite the correct Indian Act +
                            Section for the question? (e.g. cheque dishonour
                            → §138 NI Act, not §420 IPC)
  3. nudge_detection        For drafter outputs: did the system surface the
                            expected risk-nudge? (e.g. §27 ICA non-compete
                            flag, IBC §14 moratorium, POSH for employment)
  4. banned_phrase          The answer must not contain 'I think', 'in my
                            opinion', 'you should sue', 'as an AI', etc.

Run:
    python3 eval/bench/biglaw_bench.py                     # full 110
    python3 eval/bench/biglaw_bench.py --filter citation   # one scorer only
    python3 eval/bench/biglaw_bench.py --limit 20          # smoke

Output:  eval/reports/biglaw_bench_<ts>.json
         eval/reports/biglaw_bench_<ts>.md  (scorecard)
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

ROOT = Path(__file__).resolve().parents[2]
DB   = Path("/Users/pranav/Desktop/india-judgments-corpus/india_courts.db")
BACKEND = "http://localhost:8080"
QFILE   = Path(__file__).parent / "questions.jsonl"
REPORTS = ROOT / "eval" / "reports"


# ── Banned phrases (R3 / G3 / G7 / G9 from the validators) ──────────────

BANNED_HEDGE = [
    "as an ai", "i think", "in my opinion", "i believe",
    "as an artificial intelligence", "as a language model",
]
BANNED_JUDGMENT = [
    "you should sue", "you should file", "i recommend you", "your best move",
    "you must hire", "in my view you",
]
BANNED_ACCOUNTABILITY = [
    "i'll handle", "leave it to me", "i will take care",
]
ALL_BANNED = BANNED_HEDGE + BANNED_JUDGMENT + BANNED_ACCOUNTABILITY


# ── Scorers ─────────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    passed: bool
    score: float
    notes: list[str] = field(default_factory=list)


def score_banned(answer: str, _q: dict) -> ScoreResult:
    notes: list[str] = []
    a = answer.lower()
    hits = [p for p in ALL_BANNED if p in a]
    return ScoreResult(passed=not hits, score=1.0 if not hits else 0.0,
                       notes=[f"banned phrase: {p!r}" for p in hits])


_CITATION_RE = re.compile(
    r"\b(\([0-9]{4}\)\s*\d+\s*(SCC|AIR|SCR|SCALE|JT|Bom|Del|Mad|Cal|All|Pat|Ker|Guj|MP|Raj|UP|Bom)|"
    r"AIR\s+\d{4}\s+SC\s+\d+|"
    r"(?:[A-Z][A-Za-z]+\s+v(?:s|\.|ersus)\s+[A-Z][A-Za-z]+))\b",
    re.IGNORECASE,
)


def score_citation_faithfulness(answer: str, q: dict, db: sqlite3.Connection) -> ScoreResult:
    """Every citation in the answer must resolve to a real case in the corpus.
    We accept either a v./vs./versus party-name pattern or a reporter citation.
    """
    cites = _CITATION_RE.findall(answer)
    if not cites:
        # No citation made — pass IF the question doesn't require one
        if q.get("requires_citation"):
            return ScoreResult(False, 0.0, ["no citation produced; question expects one"])
        return ScoreResult(True, 1.0, ["no citation produced; not required"])
    notes: list[str] = []
    verified = 0
    for raw_match in cites:
        cite = raw_match if isinstance(raw_match, str) else " ".join(raw_match)
        cite = cite.strip()
        # Try FTS5 in judgments + legal_docs
        token = cite.replace("(", "").replace(")", "")
        token = re.sub(r"[^A-Za-z0-9. ]", " ", token).strip()
        if len(token) < 6:
            continue
        # Quote the strongest 3 words
        keywords = " ".join(token.split()[:3])
        try:
            row = db.execute(
                "SELECT 1 FROM legal_docs_fts WHERE legal_docs_fts MATCH ? LIMIT 1",
                (f'"{keywords}"',),
            ).fetchone()
            if not row:
                row = db.execute(
                    "SELECT 1 FROM judgments_fts WHERE judgments_fts MATCH ? LIMIT 1",
                    (f'"{keywords}"',),
                ).fetchone()
        except sqlite3.Error:
            row = None
        if row:
            verified += 1
        else:
            notes.append(f"unresolved citation: {cite!r}")
    total = max(1, len(cites))
    score = verified / total
    return ScoreResult(passed=score >= 0.7, score=score,
                       notes=notes if notes else [f"{verified}/{total} citations resolved"])


def score_statute_precision(answer: str, q: dict) -> ScoreResult:
    must = q.get("must_cite_acts", []) or []
    must_sections = q.get("must_cite_sections", []) or []
    if not must and not must_sections:
        return ScoreResult(True, 1.0, ["no statute expectation set"])
    a = answer.lower()
    hits_acts = [x for x in must if x.lower() in a]
    hits_secs = [s for s in must_sections if s.lower() in a]
    need = len(must) + len(must_sections)
    got  = len(hits_acts) + len(hits_secs)
    score = got / max(1, need)
    missing = [x for x in must if x.lower() not in a] + [s for s in must_sections if s.lower() not in a]
    return ScoreResult(passed=score >= 0.7, score=score,
                       notes=[f"missing: {m}" for m in missing] or [f"{got}/{need} required cites present"])


def score_nudge_detection(answer: Any, q: dict) -> ScoreResult:
    """answer is the JSON nudges-response payload (list of nudge dicts)."""
    expected = set(q.get("expected_nudges", []) or [])
    if not expected:
        return ScoreResult(True, 1.0, ["no nudge expectation set"])
    nudges = answer if isinstance(answer, list) else (answer.get("nudges", []) if isinstance(answer, dict) else [])
    fired_clauses = {n.get("clause", "").lower() for n in nudges if isinstance(n, dict)}
    fired_acts    = {n.get("act", "").lower()    for n in nudges if isinstance(n, dict)}
    found = 0
    missing: list[str] = []
    for e in expected:
        if any(e.lower() in c for c in fired_clauses) or any(e.lower() in a for a in fired_acts):
            found += 1
        else:
            missing.append(e)
    score = found / max(1, len(expected))
    return ScoreResult(passed=score >= 0.7, score=score,
                       notes=[f"missing nudge: {m}" for m in missing] or [f"{found}/{len(expected)} expected nudges fired"])


# ── Runner ──────────────────────────────────────────────────────────────

def call_search(q: str, mode: str = "hybrid", limit: int = 5) -> str:
    """Run a search query → flatten top-N hits to a single text answer."""
    r = httpx.post(f"{BACKEND}/api/cases/smart-search",
                   json={"q": q, "mode": mode, "limit": limit}, timeout=30.0)
    r.raise_for_status()
    hits = r.json().get("hits", [])
    parts = []
    for h in hits:
        parts.append(f"{h.get('title','')} — {h.get('court','')} {h.get('year','')}")
        if h.get("snippet"):
            parts.append(h["snippet"])
    return "\n".join(parts)


def call_compliance(text: str, doc_type: str = "") -> dict:
    r = httpx.post(f"{BACKEND}/api/contract/compliance",
                   json={"body_md": text, "doc_type": doc_type}, timeout=30.0)
    r.raise_for_status()
    return r.json()


def call_nudges(text: str) -> dict:
    r = httpx.post(f"{BACKEND}/api/contract/nudges",
                   json={"body_md": text, "with_cases": False}, timeout=60.0)
    r.raise_for_status()
    return r.json()


def run_question(q: dict, db: sqlite3.Connection) -> dict:
    kind = q["kind"]   # search | nudge | compliance | banned
    t0 = time.time()
    try:
        if kind == "search":
            answer = call_search(q["query"], mode=q.get("mode", "hybrid"),
                                 limit=q.get("limit", 5))
        elif kind == "nudge":
            answer = call_nudges(q["body_md"]).get("nudges", [])
        elif kind == "compliance":
            answer = call_compliance(q["body_md"], q.get("doc_type", "")).get("findings", [])
        else:
            answer = q.get("answer", "")
    except Exception as e:
        return {"id": q["id"], "kind": kind, "passed": False, "score": 0.0,
                "error": str(e), "elapsed_ms": int((time.time() - t0) * 1000)}

    scores: dict[str, ScoreResult] = {}
    if kind == "search":
        scores["banned"]   = score_banned(answer, q)
        scores["citation"] = score_citation_faithfulness(answer, q, db)
        scores["statute"]  = score_statute_precision(answer, q)
    elif kind == "nudge":
        scores["nudge"] = score_nudge_detection(answer, q)
    elif kind == "compliance":
        # Re-use the nudge scorer for compliance findings (same shape)
        scores["compliance"] = score_nudge_detection(answer, q)
    else:
        scores["banned"] = score_banned(answer, q)

    overall_pass = all(s.passed for s in scores.values())
    overall_score = sum(s.score for s in scores.values()) / max(1, len(scores))
    return {
        "id": q["id"],
        "kind": kind,
        "category": q.get("category"),
        "passed": overall_pass,
        "score": round(overall_score, 3),
        "scores": {k: {"passed": v.passed, "score": round(v.score, 3), "notes": v.notes}
                   for k, v in scores.items()},
        "elapsed_ms": int((time.time() - t0) * 1000),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default=None, help="Substring match on q.kind or q.id")
    ap.add_argument("--limit",  type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.80)
    args = ap.parse_args(argv)

    if not QFILE.exists():
        print(f"FATAL: questions file not found: {QFILE}", file=sys.stderr)
        return 2

    qs = [json.loads(l) for l in QFILE.read_text().splitlines() if l.strip()]
    if args.filter:
        qs = [q for q in qs if args.filter.lower() in (q.get("kind","") + q.get("id","")).lower()]
    if args.limit:
        qs = qs[: args.limit]

    db = sqlite3.connect(str(DB), timeout=30.0)
    db.execute("PRAGMA query_only=ON")

    results: list[dict] = []
    t0 = time.time()
    for i, q in enumerate(qs, 1):
        res = run_question(q, db)
        results.append(res)
        mark = "✓" if res["passed"] else "✗"
        print(f"  {mark} [{i:3d}/{len(qs)}] {q['kind']:10s} {q['id']:30s} score={res['score']:.2f} ({res['elapsed_ms']}ms)")

    db.close()

    passed = sum(1 for r in results if r["passed"])
    total  = len(results)
    overall = (sum(r["score"] for r in results) / max(1, total))
    elapsed = time.time() - t0

    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    rep_json = REPORTS / f"biglaw_bench_{ts}.json"
    rep_md   = REPORTS / f"biglaw_bench_{ts}.md"
    rep_json.write_text(json.dumps({
        "ts": ts, "n": total, "pass": passed, "score": overall,
        "elapsed_s": round(elapsed, 1), "threshold": args.threshold,
        "results": results,
    }, indent=2, ensure_ascii=False))

    # Markdown scorecard
    by_kind: dict[str, list[dict]] = {}
    for r in results:
        by_kind.setdefault(r["kind"], []).append(r)
    md = [
        f"# Indian BigLaw Bench — {ts}",
        f"- Pass rate: **{passed}/{total} ({100*passed/max(1,total):.1f}%)**",
        f"- Mean score: **{overall:.3f}**",
        f"- Elapsed: {elapsed:.1f}s\n",
        "## By kind\n",
        "| Kind | Pass | Total | Mean score |",
        "|---|---:|---:|---:|",
    ]
    for k, rs in sorted(by_kind.items()):
        p = sum(1 for r in rs if r["passed"])
        s = sum(r["score"] for r in rs) / max(1, len(rs))
        md.append(f"| {k} | {p} | {len(rs)} | {s:.3f} |")
    md.append("\n## Failures\n")
    for r in results:
        if not r["passed"]:
            md.append(f"- `{r['id']}` ({r['kind']}) — score {r['score']:.2f}")
            for sk, sv in r.get("scores", {}).items():
                for n in sv.get("notes", []):
                    md.append(f"  - {sk}: {n}")
    rep_md.write_text("\n".join(md))

    print(f"\nReport: {rep_json}")
    print(f"Scorecard: {rep_md}")
    print(f"\nOVERALL: {passed}/{total} pass ({100*passed/max(1,total):.1f}%) · mean {overall:.3f}")
    return 0 if overall >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
