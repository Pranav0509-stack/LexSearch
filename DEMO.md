# Sanhita demo — research → draft → automation agent

A 4-minute walkthrough that shows Sanhita doing the three things a lawyer
actually pays for, in one continuous thread:

1. **Research** the law and pull cited cases
2. **Draft** a legal document grounded in what was just researched
3. **Automate** a multi-step matter — research + memo + save to Drive +
   Gmail draft + matter-sheet row — in a single agent turn

The story: an Indian client (Acme Trading Pvt Ltd) holds a dishonoured
₹14.75 lakh cheque from M/s Bharat Industries. We need to advise, send a
§138 NI Act notice, and brief the partner.

---

## Two ways to run it

### A. CLI demo (record this for a screencast)

```bash
# Backend must be running (.claude/launch.json → "FastAPI Backend")
bash demo.sh
```

Acts run sequentially in a single thread. Each act prints answer +
citations + provider/grounding/latency to stdout — easy to screen-record.

Run a single act: `bash demo.sh research | draft | agent`.

### B. UI demo (record this for the marketing video)

Open http://localhost:3001, log in, then drive the UI as below. Every
step is something a lawyer would do; the implementation under the hood is
already wired.

---

## ACT 1 — Research (≈30 s)

**Mode:** Assistant pane, no toggles (default = research).

**Type:**
> What are the leading Supreme Court rulings on Section 138 of the
> Negotiable Instruments Act regarding the burden of proof for dishonour
> of cheque, especially the presumption under §139 and its rebuttal?
> List the cases by name.

**What to point to on screen:**

| What you see                       | Why it matters                                               |
| ---------------------------------- | ------------------------------------------------------------ |
| Streaming "via gemini" chip        | Gemini 2.5 Flash is the brain (Part A of the rollout)        |
| `[1] [2] [3]` inline citations     | Every claim is grounded — refused if grounding < 60%         |
| Citations rail on the right        | Real cases from BM25 + Indian Kanoon, not the seed corpus    |
| Grounding pill (e.g. `82% grounded`) | The 6-gate validator passed                                |
| Latency badge                      | p95 ≈ 6 s on Gemini Flash                                    |

**Talking point:** *"This is the same UX as Harvey, but every citation
maps back to a real Indian Kanoon URL — open one and verify."*

---

## ACT 2 — Drafting (≈45 s)

**Mode:** Same thread, click the **Canvas** toggle in the composer.

**Type:**
> On the basis of the cases cited above, draft a §138 statutory notice
> from Acme Trading Pvt Ltd (Mumbai) to the drawer, M/s Bharat
> Industries (Pune), demanding ₹14,75,000 within 15 days. Use formal
> Indian legal-notice register, include the cause-of-action paragraph,
> and cite the §139 presumption.

**What to point to on screen:**

| What you see                              | Why it matters                                                |
| ----------------------------------------- | ------------------------------------------------------------- |
| Long-form notice draft, no `[n]` blocks   | Drafting mode skips retrieval — only G3 (banned phrases) gates|
| Statute references like "§139 NI Act"     | Citation policy lets statutes through, not bare case names    |
| Citation rail still shows Act 1's cases   | Citation carry-across — the model treated them as facts       |
| `refused: false`                          | The 6-gate validator does NOT block in draft mode             |

**Talking point:** *"In research mode we refuse to fabricate. In draft
mode we let the lawyer cook. The model knew the §138 cases from Act 1
— it didn't re-fetch them, it built on them."*

---

## ACT 3 — Automation agent (≈90 s)

**Pre-req:** click **Settings → Connect Google** once (one-time OAuth).
If skipped, the agent still runs — Google tool calls return "not
connected" and the agent finishes the prose and tells the user to
connect.

**Mode:** Same thread. **No toggle needed** — the auto-agent detector
sees "find...and draft...save to Drive...email" and switches to agent
mode by itself. (Plan B's auto-detection.)

**Type:**
> Find the most recent Bombay High Court rulings on §138 NI Act
> dishonour where the drawer pleaded financial hardship as a defence
> (last 3 years). Then draft a 1-page client memo for our partner
> explaining (a) the prevailing position, (b) whether Acme should accept
> a settlement at ₹10L vs litigate, and (c) recommended next steps. Save
> the memo to my Google Drive as "Acme v. Bharat — §138 memo", queue a
> Gmail draft to client@acmetrading.example with the memo body, and
> append a row to the Sanhita matter tracker for this case.

**What to point to on screen** (agent trace expands as it runs):

| Tool call                                | What's happening                                              |
| ---------------------------------------- | ------------------------------------------------------------- |
| `research_legal(query="§138 hardship…")` | Hits BM25 + Indian Kanoon                                     |
| `summarise_for_brief(...)`               | Cleans the cases into memo-ready bullet points                |
| `create_google_doc(title, content_md)`   | New Google Doc in user's Drive, returns docs.google.com URL   |
| `create_gmail_draft(to, subject, body)`  | **Draft only** in user's Gmail — never auto-sends             |
| `append_sheet_row(matter, …)`            | New row in "Sanhita — Matter Tracker" sheet                   |
| Final answer                              | Markdown summary with the Doc URL + Gmail draft URL inline    |

**Talking point sequence:**

1. *"Notice I never picked an agent toggle. Sanhita saw the multi-step
   shape of the request and routed it itself."*
2. *"The Gmail integration uses the `gmail.compose` scope — Sanhita
   physically cannot send mail. It can only stage a draft for the
   lawyer to review and click Send."*
3. *Open the new Google Doc in another tab.* *"Same memo body the model
   gave us, formatted as a real Doc. The matter tracker row is one
   click away in Sheets."*
4. *"Citations from Acts 1 and 2 carry through — the rail at right shows
   the original Supreme Court cases plus the new Bombay HC matches the
   agent pulled."*

---

## What this proves

- **Brain swap is live:** every answer says `via gemini` (Part A)
- **BM25 corpus is live:** citations resolve to real case URLs (Part B1)
- **Modes work cleanly:** research grounds, draft frees, web cites
  snippets, agent chains tools (Part B2 + auto-detection)
- **Google Workspace is plug-and-play:** one OAuth click in Settings,
  agent gets four new tools (Plan C)
- **Citation continuity:** what you cite in turn 1 is still on the rail
  in turn 5 — no re-grounding tax (Plan B3)

## What's deliberately NOT in the demo

- **Streaming token-by-token** — the full-response UX is fine for now
- **Auto-send Gmail** — never. Drafts only, by design
- **Voice or WhatsApp** — that's the NyayaSathi track
- **Full 16M-doc BM25** — running on 200K sample; nightly cron rebuilds

---

## If the trace says `provider: groq` instead of `gemini`

That means the backend is running an instance started *before* Plan A's
chain reorder + `GEMINI_API_KEY` landed in `.claude/launch.json`. Stop
the FastAPI Backend and restart it — env changes do not hot-reload.
Same for "BM25 ready: 0 docs": the index builds in a background thread
on first start, so a fresh restart kicks it off.

## Smoke checklist before recording

```bash
# 1. Backend up?
curl -sf http://localhost:8080/api/me | jq .

# 2. Gemini reachable?
curl -s -X POST http://localhost:8080/api/brief/chat \
  -H 'Content-Type: application/json' -b /tmp/sanhita-demo.cookies \
  -d '{"thread_id":1,"question":"hi","jurisdiction":"IN"}' \
  | jq '.llm.provider'   # → "gemini"

# 3. BM25 ready? (check FastAPI logs for "BM25 ready: NN docs")

# 4. Google connected? (Settings → Connect Google → Connected as …)

# 5. Run the script:
bash demo.sh
```

Re-recordable: every act stores its raw response under
`/tmp/sanhita-demo.act{1,2,3}.json` so you can diff runs or pull a
quote out of the trace.
