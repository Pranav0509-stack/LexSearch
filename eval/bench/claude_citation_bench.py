"""
Claude Citation-Faithfulness Bench
==================================

A second bench that tests one specific question: **when the LLM is given
full-text excerpts from all six Sanhita corpora (judgments + legal_docs +
pipeline_docs + legal_qa + statutes + documents), does it cite faithfully?**

Why this matters: the BigLaw bench measures retrieval (do we find the right
cases?). This bench measures generation (does the LLM stick to what we
retrieved, or does it hallucinate citations from training data?).

Pipeline per question:
  1. Pull `--top-k` (default 8) hits from /api/cases/smart-search across
     ALL 6 corpora with `with_text=true` so we get the actual text body
     of each row, not just titles.
  2. Build a prompt with [E1]..[Ek] grounding blocks, one per corpus row.
  3. Ask the LLM (Claude if ANTHROPIC_API_KEY present, else Gemini) the
     question. The system prompt enforces "cite [E*] for every claim or
     write '(not in corpus)'".
  4. Score:
       · marker_resolution   — every [E*] in the answer is in 1..k
       · text_overlap        — at least one substantive phrase from each
                                cited row appears in the answer (proves
                                the LLM used the actual text, not invented)
       · per-corpus mix      — how many [E*] markers point at each corpus
                                type (judgments / legal_docs / pipeline_docs
                                / legal_qa / statutes / documents)
       · banned-phrase clean — no "as an AI", "I think", etc.

Run:
    python3 eval/bench/claude_citation_bench.py             # full search subset
    python3 eval/bench/claude_citation_bench.py --limit 10  # smoke
    python3 eval/bench/claude_citation_bench.py --prefer anthropic
    python3 eval/bench/claude_citation_bench.py --prefer gemini

Output:
    eval/reports/claude_citation_<timestamp>.json
    eval/reports/claude_citation_<timestamp>.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
BACKEND = os.environ.get("SANHITA_BACKEND", "http://localhost:8080")
QFILE   = Path(__file__).parent / "questions.jsonl"
REPORTS = ROOT / "eval" / "reports"

# ── Configuration ────────────────────────────────────────────────────────

DEMO_CODE = os.environ.get("SANHITA_DEMO_CODE", "SNHT-DEMO-2026")


def _ensure_session() -> httpx.Client:
    """Return an httpx.Client that's already authenticated against the demo code."""
    c = httpx.Client(base_url=BACKEND, timeout=120.0)
    r = c.post("/api/login", json={"code": DEMO_CODE})
    r.raise_for_status()
    return c


# ── Retrieval (multi-corpus, with text) ──────────────────────────────────

def retrieve_with_text(client: httpx.Client, query: str, top_k: int = 8) -> list[dict]:
    """Hit /api/cases/smart-search → fetch each hit's full text via
    /api/cases/document/{id} so we have the actual body, not just metadata."""
    r = client.post("/api/cases/smart-search",
                    json={"q": query, "mode": "hybrid", "limit": top_k * 2})
    r.raise_for_status()
    hits = (r.json().get("hits") or [])[:top_k]

    # Resolve full text for each hit. Truncate to 600 chars per row so the
    # prompt doesn't blow past 50K. Some corpora (legal_qa) put the body in
    # `summary` already; pipeline_docs / judgments need /document/{id}.
    enriched: list[dict] = []
    for h in hits:
        case_id = h.get("case_id") or h.get("doc_id") or ""
        body_text = h.get("snippet") or h.get("summary") or ""
        if len(body_text) < 200 and case_id:
            try:
                dr = client.get(f"/api/cases/document/{case_id}", timeout=15.0)
                if dr.status_code == 200:
                    d = dr.json()
                    body_text = (d.get("full_text") or d.get("text") or body_text)[:1200]
            except Exception:
                pass
        enriched.append({
            "title":        h.get("title", ""),
            "court":        h.get("court", ""),
            "year":         h.get("year"),
            "source_table": h.get("source_table", ""),
            "case_id":      case_id,
            "text":         (body_text or "")[:600],
        })
    return enriched


# ── LLM call (prefers Anthropic if key set, else Gemini via router) ──────

def call_llm(client: httpx.Client, question: str, sources: list[dict],
             prefer: str = "anthropic") -> str:
    """Build a grounded prompt and call /api/editor/ai/write-section.

    The system prompt (in doc_editor.py) already forces [E*] citation +
    "(not in corpus)" admissions, so we just need to deliver the question
    + sources as a clean instruction.
    """
    sources_block = "\n\n".join(
        f"[E{i+1}] {s['title']} — {s['source_table']} · {s['court']} {s['year']}\n   {s['text']}"
        for i, s in enumerate(sources)
    )
    instruction = (
        f"Question: {question}\n\n"
        "Answer the question using ONLY the GROUNDING SOURCES below. "
        "Append a marker like [E1] or [E3] after every factual claim. "
        "If a claim cannot be grounded in the sources, write '(not in corpus)' "
        "— do not invent citations.\n\n"
        f"GROUNDING SOURCES (from Sanhita's 83M-row corpus):\n{sources_block}"
    )
    r = client.post("/api/editor/ai/write-section",
                    json={
                        "instruction": instruction,
                        "doc_type": "legal_research",
                        "context": "",
                        "prefer": prefer,
                    }, timeout=120.0)
    r.raise_for_status()
    return r.json().get("text", "") or ""


# ── Scoring ──────────────────────────────────────────────────────────────

BANNED_RE = re.compile(
    r"\b(?:as an ai|i think|i believe|in my opinion|as an artificial intelligence|"
    r"as a language model|i'm not a lawyer|disclaimer:?\s+i)",
    re.IGNORECASE,
)
CITE_RE = re.compile(r"\[E(\d+)\]")


@dataclass
class Score:
    qid: str
    question: str
    answer: str
    sources_count: int
    markers_emitted: list[int] = field(default_factory=list)
    markers_resolved: bool = True
    markers_out_of_range: list[int] = field(default_factory=list)
    no_citation_for_claims: bool = False
    banned_phrase: str | None = None
    text_overlap_pct: float = 0.0
    per_corpus_cites: dict[str, int] = field(default_factory=dict)
    elapsed_ms: int = 0
    error: str | None = None

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        if self.banned_phrase:
            return False
        if self.markers_out_of_range:
            return False
        # If sources were provided but the LLM emitted zero markers, fail.
        if self.sources_count > 0 and not self.markers_emitted:
            return False
        # Text-overlap floor: 30% of cited sources must contribute a phrase.
        return self.text_overlap_pct >= 0.30


def _substantive_phrases(text: str) -> list[str]:
    """Pull 3+-word capitalized phrases — proper nouns + act names."""
    return re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,4}\b", text)


def score_answer(qid: str, question: str, answer: str,
                 sources: list[dict], elapsed_ms: int) -> Score:
    s = Score(qid=qid, question=question, answer=answer,
              sources_count=len(sources), elapsed_ms=elapsed_ms)

    # ── Banned phrases
    bp = BANNED_RE.search(answer)
    if bp:
        s.banned_phrase = bp.group(0)

    # ── Marker resolution
    markers = [int(m.group(1)) for m in CITE_RE.finditer(answer)]
    s.markers_emitted = sorted(set(markers))
    s.markers_out_of_range = [n for n in markers if n < 1 or n > len(sources)]
    s.markers_resolved = not s.markers_out_of_range

    # ── Per-corpus distribution
    counts: Counter[str] = Counter()
    for n in markers:
        if 1 <= n <= len(sources):
            counts[sources[n - 1].get("source_table", "unknown")] += 1
    s.per_corpus_cites = dict(counts)

    # ── Text-overlap (did the LLM actually use the source text?)
    if markers:
        used_sources = {n - 1 for n in markers if 1 <= n <= len(sources)}
        if used_sources:
            cited_phrases_present = 0
            ans_lower = answer.lower()
            for idx in used_sources:
                src_phrases = _substantive_phrases(sources[idx].get("text", ""))
                if any(p.lower() in ans_lower for p in src_phrases[:5]):
                    cited_phrases_present += 1
            s.text_overlap_pct = cited_phrases_present / len(used_sources)

    return s


# ── Main runner ──────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Truncate to N questions")
    ap.add_argument("--top-k", type=int, default=8, help="Sources retrieved per question")
    ap.add_argument("--prefer", default="anthropic",
                    choices=["anthropic", "gemini", "groq"],
                    help="LLM provider preference (falls back via router)")
    ap.add_argument("--filter", default="search",
                    help="Substring match on q.kind or q.id (default: search-mode only)")
    ap.add_argument("--delay-s", type=float, default=1.0,
                    help="Sleep N seconds between calls to dodge Gemini RPM throttle")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    questions = []
    for ln in QFILE.read_text().splitlines():
        if not ln.strip():
            continue
        q = json.loads(ln)
        if args.filter and args.filter not in q.get("kind", "") and args.filter not in q.get("id", ""):
            continue
        questions.append(q)

    if args.limit > 0:
        questions = questions[: args.limit]

    print(f"Running citation bench on {len(questions)} questions  "
          f"(top-k={args.top_k}, prefer={args.prefer})")
    print()

    client = _ensure_session()
    scores: list[Score] = []
    t_total = time.monotonic()
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question = q.get("query") or q.get("question") or q.get("body_md") or ""
        t0 = time.monotonic()
        try:
            sources = retrieve_with_text(client, question, top_k=args.top_k)
            answer = call_llm(client, question, sources, prefer=args.prefer)
            elapsed = int((time.monotonic() - t0) * 1000)
            s = score_answer(qid, question, answer, sources, elapsed)
        except Exception as e:
            s = Score(qid=qid, question=question, answer="",
                      sources_count=0, elapsed_ms=int((time.monotonic() - t0) * 1000),
                      error=str(e)[:200])
        scores.append(s)
        flag = "✓" if s.passed else "✗"
        cites = ",".join(str(m) for m in s.markers_emitted[:5]) or "—"
        corpora = ",".join(s.per_corpus_cites.keys())
        print(f"  {flag} [{i:>3}/{len(questions)}] {qid:30s} cites=[{cites}] "
              f"src={s.sources_count}  overlap={s.text_overlap_pct:.0%}  "
              f"corpora={{ {corpora} }}  ({s.elapsed_ms}ms)")
        if args.verbose and s.error:
            print(f"        ERROR: {s.error}")
        # Throttle to avoid hitting Gemini Flash free-tier RPM cap (15 RPM).
        if args.delay_s > 0 and i < len(questions):
            time.sleep(args.delay_s)

    total_elapsed = time.monotonic() - t_total
    passed = sum(1 for s in scores if s.passed)

    # Per-corpus citation aggregation
    corpus_totals: Counter[str] = Counter()
    for s in scores:
        for k, v in s.per_corpus_cites.items():
            corpus_totals[k] += v
    banned_count = sum(1 for s in scores if s.banned_phrase)
    out_of_range = sum(1 for s in scores if s.markers_out_of_range)

    # Write JSON + markdown reports
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS / f"claude_citation_{ts}.json"
    md_path = REPORTS / f"claude_citation_{ts}.md"

    json_path.write_text(json.dumps({
        "timestamp": ts,
        "prefer": args.prefer,
        "top_k": args.top_k,
        "n_questions": len(scores),
        "passed": passed,
        "pass_rate": round(passed / max(1, len(scores)), 3),
        "banned_phrase_count": banned_count,
        "markers_out_of_range_count": out_of_range,
        "corpus_citation_totals": dict(corpus_totals),
        "elapsed_s": round(total_elapsed, 1),
        "scores": [asdict(s) for s in scores],
    }, indent=2))

    md_lines = [
        f"# Claude Citation-Faithfulness Bench — {ts}",
        f"- Provider preference: **{args.prefer}** (router falls back if key absent)",
        f"- Questions: {len(scores)}",
        f"- **Pass rate: {passed}/{len(scores)} ({passed/max(1,len(scores)):.1%})**",
        f"- Banned-phrase violations: {banned_count}",
        f"- Out-of-range [E*] markers: {out_of_range}",
        f"- Elapsed: {total_elapsed:.1f}s ({total_elapsed/max(1,len(scores)):.1f}s/q)",
        "",
        "## Citation distribution across corpora",
        "",
        "| Corpus | # of [E*] markers pointing at it |",
        "|---|---:|",
    ]
    for k in ("judgments", "legal_docs", "pipeline_docs", "statutes", "legal_qa", "documents"):
        md_lines.append(f"| {k} | {corpus_totals.get(k, 0)} |")
    md_lines += [
        "",
        "## Per-question detail",
        "",
        "| Qid | passed | sources | markers | overlap | banned | elapsed |",
        "|---|:---:|---:|---|---:|---|---:|",
    ]
    for s in scores:
        flag = "✓" if s.passed else "✗"
        cites = ",".join(str(m) for m in s.markers_emitted[:6]) or "—"
        md_lines.append(
            f"| {s.qid} | {flag} | {s.sources_count} | `[{cites}]` | "
            f"{s.text_overlap_pct:.0%} | {s.banned_phrase or ''} | {s.elapsed_ms}ms |"
        )
    md_path.write_text("\n".join(md_lines))

    print()
    print(f"OVERALL  {passed}/{len(scores)} passed  ({passed/max(1,len(scores)):.1%})  "
          f"in {total_elapsed:.1f}s")
    print(f"Reports: {json_path}\n         {md_path}")
    return 0 if passed >= len(scores) * 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())
