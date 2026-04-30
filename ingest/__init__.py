"""Daily ingestion pipeline for LexSearch.

Scripts here fetch fresh Indian court judgments (Supreme Court + top High
Courts) and write parquet partitions compatible with the existing
`indian-supreme-court-judgments` / `indian-high-court-judgments` layout that
server.py already reads.

Run order (via cron / GitHub Actions):
  1. sc_daily.py        — 21:00 IST every day
  2. hc_daily.py        — 22:00 IST every day
  3. rebuild_bm25.py    — 23:30 IST every day (consumes both)
"""
