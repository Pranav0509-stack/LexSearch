# LexSearch — Indian High Court Judgments

A personal legal research tool to search, read, and download Indian High Court judgments from the public [indian-high-court-judgments](https://github.com/vanga/indian-high-court-judgments) dataset (16.7M+ cases, CC-BY-4.0).

## Features

- **8 Search Filters** — Court, Bench, Year, Case Title/Party, CNR Number, Judge Name, Case Type, Disposal Status
- **In-Browser PDF Reading** — Read judgments directly in the browser with zoom, page navigation, and keyboard shortcuts
- **Download** — Save relevant judgments as PDF with one click
- **25 High Courts** — All Indian High Courts with their benches
- **No Setup Required** — Pulls data live from public AWS S3, no database or API keys needed

## Tech Stack

- **Backend:** Python, FastAPI, Pandas, PyArrow, s3fs
- **Frontend:** Vanilla HTML/CSS/JS, PDF.js
- **Data:** AWS S3 public bucket (`indian-high-court-judgments`)

## Run Locally

```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 8080
```

Open [http://localhost:8080](http://localhost:8080)

## Deploy on Render (Free)

1. Fork this repo
2. Go to [render.com](https://render.com) → New → Blueprint
3. Connect the repo — `render.yaml` handles the rest

## Dataset Limitations

- Covers **High Courts only** — no Supreme Court, District Courts, or Tribunals
- Updated **quarterly** from eCourts portal (not real-time)
- Some older PDFs are scanned images — text search may not work on them

## Data Source

[vanga/indian-high-court-judgments](https://github.com/vanga/indian-high-court-judgments) · [openjustice-in](https://github.com/openjustice-in)

---

*Made for Shreya Didi*
