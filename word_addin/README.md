# Sanhita Word Add-in

A Word task-pane that brings Sanhita's Indian-law stack directly into Microsoft Word — where lawyers actually work.

## Four tabs

| Tab | What it does |
|---|---|
| **Draft** | Pick from 26 Sanhita templates (5 verbatim Government forms + statutory + court-aligned + practice-standard). Fill required slots. Inserts the rendered document into the open Word file with proper headings + bold runs. |
| **Search** | Hybrid BM25 + semantic search over 70M+ Indian court records. Click "Insert citation" on any hit to drop a parenthetical case-name reference at the cursor. |
| **Compliance** | Reads the current document body and runs 8 plug-ins (DPDP · RBI/FEMA · SEBI · IBC §14 · GST · IT Act §43A · POSH · Stamp Duty). Surfaces severity-tagged findings with rule-id + remediation. |
| **Quick edit** | Select text. Choose Polish / Shorten / Cite. AI rewrites in place — **preserves statute citations verbatim**, never invents references. Cite mode attaches real Indian-law case citations from Sanhita's corpus. |

## Files

| File | Purpose |
|---|---|
| `manifest.xml` | Office add-in manifest. Side-loaded into Word. |
| `taskpane.html` | The task-pane UI (Sanhita warm-parchment palette). |
| `taskpane.js` | Office.js wiring + Sanhita API calls. |
| `taskpane.css` | (inline in HTML) |

## Local setup

1. Start a static server for the add-in HTML on `:3001`:

```bash
cd "/Users/pranav/Desktop/LexSearch-main 2/word_addin"
python3 -m http.server 3001
```

2. Ensure the Sanhita backend is up at `:8080`:

```bash
curl http://localhost:8080/api/contract/health
```

3. Side-load `manifest.xml` into Word:

**macOS:** copy `manifest.xml` to `~/Library/Containers/com.microsoft.Word/Data/Documents/wef/`.

**Windows / Office 365 admin:** Upload via the Office Add-ins dialog or deploy via the Microsoft 365 admin centre.

Then in Word: **Insert → My Add-ins → Sanhita Drafter → Open**.

## Production deploy

1. Host the three files on a TLS endpoint (e.g. `https://addin.sanhita.ai/`).
2. Update `manifest.xml` URLs from `http://localhost:3001/...` to your production host.
3. Submit to Microsoft AppSource for one-click install across Indian firms.

## Why this matters

Indian law firms run on Word. Every plaint, vakalatnama, MSA, and Sec.34 petition starts and ends as a `.docx`. Pulling Sanhita's drafter, statute search, and compliance plug-ins into the task pane means a lawyer never has to leave the document — and our 70M-row corpus is one keystroke away from any active draft.
