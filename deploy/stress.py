"""Sanhita stress test — find the bottlenecks before launch.

Runs four scenarios:

  1. Concurrent smart-search (50 parallel hybrid queries)
  2. PDF proxy load (20 parallel S3 streams, first 64KB only)
  3. Compliance on a 10K-word body
  4. LLM-backed quick-edit roundtrip (real Gemini call)

Reports p50 / p95 / p99 / max / errors for each. Writes a structured
JSON to eval/reports/stress_<ts>.json and a human-readable scorecard.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics as st
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3

import httpx

BACKEND = "http://localhost:8080"
DB = "/Users/pranav/Desktop/india-judgments-corpus/india_courts.db"
REPORTS = Path("/Users/pranav/Desktop/LexSearch-main 2/eval/reports")
REPORTS.mkdir(parents=True, exist_ok=True)

# 30 diverse legal queries to exercise different FTS5 + FAISS paths
QUERIES = [
    "section 138 cheque dishonour",
    "anticipatory bail elderly accused medical",
    "writ petition article 226 alternative remedy",
    "section 482 quashing bhajan lal",
    "section 34 arbitration set aside award",
    "IBC section 7 financial creditor moratorium",
    "motor accident compensation multiplier sarla verma",
    "domestic violence act 2005 protection order",
    "section 138 NI act 30 days notice cure period",
    "regular bail section 439 triple test",
    "section 80 CPC notice government suit",
    "section 12A commercial courts mediation",
    "section 167(2) default bail chargesheet",
    "puttaswamy right to privacy",
    "kesavananda bharati basic structure",
    "section 124 indian contract act indemnity",
    "section 27 contract act restraint of trade",
    "section 73 damages indian contract act",
    "section 74 liquidated damages penalty",
    "transfer of property act section 54 sale",
    "registration act section 17 compulsory",
    "indian succession act section 276 probate",
    "POSH act 2013 internal complaints committee",
    "DPDP act 2023 data fiduciary obligations",
    "section 43A IT act sensitive personal data",
    "SARFAESI section 13 enforcement secured creditor",
    "section 12(5) arbitration unilateral perkins",
    "section 14 IBC moratorium gujarat urja",
    "trademark act section 29 deceptive similarity",
    "consumer protection act 2019 CCPA",
]


@dataclass
class Stage:
    name: str
    n: int
    latencies_ms: list[float] = field(default_factory=list)
    errors: list[str]            = field(default_factory=list)
    extra: dict                  = field(default_factory=dict)

    def summary(self) -> dict:
        if not self.latencies_ms:
            return {"name": self.name, "n": self.n, "errors": len(self.errors), "no_data": True}
        s = sorted(self.latencies_ms)
        p = lambda q: s[min(int(len(s) * q), len(s) - 1)]
        return {
            "name":     self.name,
            "n":        self.n,
            "ok":       len(self.latencies_ms),
            "errors":   len(self.errors),
            "p50_ms":   round(p(0.50), 1),
            "p95_ms":   round(p(0.95), 1),
            "p99_ms":   round(p(0.99), 1),
            "max_ms":   round(s[-1], 1),
            "mean_ms":  round(st.mean(s), 1),
            "extra":    self.extra,
            "errs_sample": self.errors[:5],
        }


# ── Stage 1: concurrent smart-search ─────────────────────────────────

async def stage_search(stage: Stage):
    """Fire all 30 queries in parallel via hybrid mode, then 20 more random
    queries to reach 50 concurrent connections (test the engine + FTS5)."""
    async with httpx.AsyncClient(timeout=60.0) as cli:
        queries = list(QUERIES) + QUERIES[:20]
        async def hit(q):
            t0 = time.time()
            try:
                r = await cli.post(f"{BACKEND}/api/cases/smart-search",
                                   json={"q": q, "mode": "hybrid", "limit": 5})
                r.raise_for_status()
                d = r.json()
                stage.extra.setdefault("hits_total", 0)
                stage.extra["hits_total"] += len(d.get("hits", []))
            except Exception as e:
                stage.errors.append(f"{q[:30]}: {str(e)[:80]}")
                return
            stage.latencies_ms.append((time.time() - t0) * 1000)
        await asyncio.gather(*(hit(q) for q in queries))


# ── Stage 2: PDF proxy load ──────────────────────────────────────────

async def stage_pdf(stage: Stage):
    conn = sqlite3.connect(DB, timeout=10.0)
    rows = conn.execute(
        "SELECT pdf_url FROM pipeline_docs WHERE source='aws_s3_hc' AND has_pdf=1 LIMIT 20"
    ).fetchall()
    conn.close()
    keys = [r[0].replace("s3://indian-high-court-judgments/", "") for r in rows]
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as cli:
        async def hit(key: str):
            t0 = time.time()
            try:
                r = await cli.get(f"{BACKEND}/pdf/{key}",
                                   headers={"Range": "bytes=0-65535"})
                r.raise_for_status()
                ct = r.headers.get("content-type", "")
                if "pdf" not in ct.lower():
                    stage.errors.append(f"bad ct: {ct} (key {key[:60]})")
                    return
            except Exception as e:
                stage.errors.append(f"{key[:60]}: {str(e)[:80]}")
                return
            stage.latencies_ms.append((time.time() - t0) * 1000)
        await asyncio.gather(*(hit(k) for k in keys))


# ── Stage 3: huge-body compliance ────────────────────────────────────

async def stage_compliance_xl(stage: Stage):
    """Run compliance plugins on a 10K-word body to ensure regex scanning
    doesn't blow up. Use 5 parallel hits."""
    boilerplate = (
        "This Agreement covers the processing of personal data of customers and "
        "is subject to DPDP Act 2023. The Service Provider acts as a Data Processor "
        "and shall implement reasonable security practices under Section 43A of the "
        "Information Technology Act, 2000. Default interest at 24% per annum shall "
        "be charged on overdue amounts. The arbitrator shall be a sole arbitrator. "
        "The Lessee shall pay stamp duty per the Maharashtra Stamp Act. "
        "The Borrower may terminate this Agreement immediately on the Customer's "
        "admission of insolvency proceedings under the IBC. "
    )
    big_body = (boilerplate * 200)  # ~10K words
    async with httpx.AsyncClient(timeout=120.0) as cli:
        async def hit(_):
            t0 = time.time()
            try:
                r = await cli.post(f"{BACKEND}/api/contract/compliance",
                                   json={"body_md": big_body, "doc_type": "msa"})
                r.raise_for_status()
                d = r.json()
                stage.extra.setdefault("findings_seen", 0)
                stage.extra["findings_seen"] += d.get("count", 0)
            except Exception as e:
                stage.errors.append(str(e)[:120])
                return
            stage.latencies_ms.append((time.time() - t0) * 1000)
        await asyncio.gather(*(hit(i) for i in range(5)))
    stage.extra["body_chars"] = len(big_body)
    stage.extra["body_words"] = len(big_body.split())


# ── Stage 4: LLM quick-edit (real Gemini roundtrip) ─────────────────

async def stage_quick_edit(stage: Stage):
    """5 parallel polish + 5 cite calls. Tests the Gemini integration on
    a real legal paragraph."""
    text = (
        "The Petitioner herein, after the receipt of the impugned order dated "
        "12 March 2025 passed by the learned Sole Arbitrator, has approached "
        "this Hon'ble Court invoking Section 34 of the Arbitration and "
        "Conciliation Act, 1996, contending that the impugned award is in "
        "manifest violation of Section 28(3) of the said Act and is patently "
        "illegal within the meaning of Section 34(2A)."
    )
    async with httpx.AsyncClient(timeout=120.0) as cli:
        async def hit(action: str):
            t0 = time.time()
            try:
                r = await cli.post(f"{BACKEND}/api/contract/quick-edit",
                                   json={"action": action, "text": text})
                r.raise_for_status()
                d = r.json()
                if d.get("unchanged"):
                    stage.errors.append(f"{action} unchanged: {d.get('reason')}")
                    return
                stage.extra.setdefault("models", set())
                stage.extra["models"].add(d.get("model", "?"))
            except Exception as e:
                stage.errors.append(f"{action}: {str(e)[:120]}")
                return
            stage.latencies_ms.append((time.time() - t0) * 1000)
        tasks = [hit("polish")  for _ in range(5)] \
              + [hit("cite")    for _ in range(3)] \
              + [hit("shorten") for _ in range(2)]
        await asyncio.gather(*tasks)
    if "models" in stage.extra:
        stage.extra["models"] = sorted(stage.extra["models"])


# ── Driver ───────────────────────────────────────────────────────────

async def main():
    print("═══ Sanhita stress test ═══")
    stages = [
        Stage("smart-search (50 parallel hybrid)", 50),
        Stage("PDF proxy (20 parallel S3 streams)", 20),
        Stage("compliance XL (5× 10K-word body)",   5),
        Stage("quick-edit Gemini (5 polish + 3 cite + 2 shorten)", 10),
    ]
    fns = [stage_search, stage_pdf, stage_compliance_xl, stage_quick_edit]
    for stage, fn in zip(stages, fns):
        print(f"\n→ {stage.name} (n={stage.n}) …")
        t0 = time.time()
        await fn(stage)
        elapsed = time.time() - t0
        s = stage.summary()
        print(json.dumps(s, indent=2, default=list))
        print(f"   wall-clock: {elapsed:.1f}s")

    # Persist
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = REPORTS / f"stress_{ts}.json"
    out.write_text(json.dumps([s.summary() for s in stages], indent=2, default=list))
    print(f"\nReport: {out}")
    # Markdown card
    md = REPORTS / f"stress_{ts}.md"
    lines = [f"# Sanhita stress test — {ts}", ""]
    for s in stages:
        d = s.summary()
        lines.append(f"## {d['name']}")
        if d.get("no_data"):
            lines.append(f"- **No data — {d['errors']} errors**")
        else:
            lines.append(f"- n={d['n']} · ok={d['ok']} · errors={d['errors']}")
            lines.append(f"- p50 = **{d['p50_ms']} ms**, p95 = **{d['p95_ms']} ms**, p99 = {d['p99_ms']} ms, max = {d['max_ms']} ms")
            lines.append(f"- mean = {d['mean_ms']} ms")
        if d.get("extra"):
            lines.append(f"- extra: `{d['extra']}`")
        if d.get("errs_sample"):
            lines.append("- error samples:")
            for e in d["errs_sample"]:
                lines.append(f"  - `{e}`")
        lines.append("")
    md.write_text("\n".join(lines))
    print(f"Markdown: {md}")


if __name__ == "__main__":
    asyncio.run(main())
