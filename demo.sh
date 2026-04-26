#!/usr/bin/env bash
# Sanhita end-to-end demo — three acts in one thread.
#
#   Act 1  Research mode   →  cite Supreme Court rulings on §138 NI Act
#   Act 2  Drafting mode   →  draft a §138 statutory notice (citations carry across)
#   Act 3  Agent mode      →  multi-step: research + Google Doc + Gmail draft + Sheet row
#
# Pre-req:
#   • FastAPI Backend running on :8080  (`/.claude/launch.json` "FastAPI Backend")
#   • LEXSEARCH_ADMIN_TOKEN=testadmin   (already set in launch.json)
#   • For Act 3's Google tools, click "Connect Google" in Settings first.
#     The agent is graceful — if Google isn't connected, the tools return
#     a "not connected" error and the agent finishes the prose answer.
#
# Usage:
#   bash demo.sh           # run the whole show
#   bash demo.sh research  # only Act 1
#   bash demo.sh draft     # only Act 2
#   bash demo.sh agent     # only Act 3
#
set -euo pipefail

BACKEND="${BACKEND:-http://localhost:8080}"
ADMIN_TOKEN="${LEXSEARCH_ADMIN_TOKEN:-testadmin}"
COOKIE_JAR="${COOKIE_JAR:-/tmp/sanhita-demo.cookies}"
THREAD_FILE="/tmp/sanhita-demo.thread"

# Cosmetics — make the transcript readable on a slide.
B="$(printf '\033[1m')"; R="$(printf '\033[0m')"
DIM="$(printf '\033[2m')"; CYAN="$(printf '\033[36m')"; GREEN="$(printf '\033[32m')"

hr() { printf '%s\n' "────────────────────────────────────────────────────────────────────"; }
banner() {
  echo
  hr
  printf "${B}${CYAN}%s${R}\n" "$1"
  [ $# -ge 2 ] && printf "${DIM}%s${R}\n" "$2"
  hr
}

require() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing dep: $1. brew install $1" >&2
    exit 1
  }
}
require curl
require jq

# ── Step 0 — make sure the backend is up ──────────────────────────────────
if ! curl -sf "$BACKEND/api/me" >/dev/null; then
  echo "Backend not reachable at $BACKEND. Start it from .claude/launch.json." >&2
  exit 1
fi

# ── Step 1 — log in (or mint a demo code) ─────────────────────────────────
login_or_mint_code() {
  # Already have a session cookie? Reuse.
  if [ -f "$COOKIE_JAR" ] && curl -sf -b "$COOKIE_JAR" "$BACKEND/api/me" \
      | jq -e '.authenticated == true' >/dev/null; then
    echo "${GREEN}✓ session reused${R} (${COOKIE_JAR})"
    return
  fi

  banner "Logging in" "Requesting an access code, admin-approving, and exchanging it for a session."

  # 1a. file an access request
  local rid
  rid=$(curl -s -X POST "$BACKEND/api/access-request" \
        -H 'Content-Type: application/json' \
        -d '{"name":"Demo User","email":"demo+sanhita@example.com","role":"litigation_partner","firm":"Demo & Co.","bar_no":"DEM/0001","note":"end-to-end demo"}' \
        | jq -r '.request_id')
  echo "  request_id=$rid"

  # 1b. admin-approve to mint a plaintext access code
  local code
  code=$(curl -s -X POST "$BACKEND/api/admin/requests/$rid/approve" \
         -H "Authorization: Bearer $ADMIN_TOKEN" | jq -r '.access_code')
  if [ "$code" = "null" ] || [ -z "$code" ]; then
    echo "  (request already approved or admin token wrong; falling back to ${SANHITA_CODE:-})"
    code="${SANHITA_CODE:-}"
  fi
  [ -z "$code" ] && { echo "No access code; export SANHITA_CODE=… and retry." >&2; exit 1; }
  echo "  access_code=${code:0:4}…${code: -2}"

  # 1c. exchange for a session cookie
  curl -s -c "$COOKIE_JAR" -X POST "$BACKEND/api/login" \
       -H 'Content-Type: application/json' \
       -d "{\"code\":\"$code\"}" >/dev/null
  echo "${GREEN}✓ session minted${R}"
}

# ── Step 2 — pick or open a thread ────────────────────────────────────────
ensure_thread() {
  if [ -f "$THREAD_FILE" ]; then
    THREAD_ID=$(cat "$THREAD_FILE")
    return
  fi
  THREAD_ID=$(curl -s -b "$COOKIE_JAR" -X POST "$BACKEND/api/brief/threads" \
              -H 'Content-Type: application/json' \
              -d '{"title":"§138 NI Act demo — Acme v. Drawer"}' \
              | jq -r '.thread.id // .id')
  echo "$THREAD_ID" > "$THREAD_FILE"
  echo "${GREEN}✓ new thread${R} id=$THREAD_ID"
}

pretty_answer() {
  jq -r '
    "── ANSWER ──\n" +
    (.answer_markdown // .answer // .text // "(no answer)") + "\n\n" +
    "── CITATIONS ──\n" +
    (
      (.citations // []) | to_entries | map(
        "[" + ((.key + 1) | tostring) + "] " +
        (.value.title // .value.case // .value.url // "?") +
        (if .value.url then "\n    " + .value.url else "" end) +
        (if .value.source then "\n    source=" + .value.source else "" end)
      ) | join("\n")
    ) + "\n\n" +
    "── META ──\n" +
    "provider:   " + ((.llm.provider // "?") | tostring) + "\n" +
    "model:      " + ((.llm.model // "?") | tostring) + "\n" +
    "grounding:  " + (((.validation.confidence // 0) * 100 | floor) | tostring) + "%\n" +
    "latency_ms: " + ((.llm.latency_ms // 0) | tostring) + "\n" +
    "refused:    " + ((.refused // false) | tostring) +
    (if (.validation.reasons // []) | length > 0
     then "\nguard:      " + ((.validation.reasons // []) | join(", "))
     else "" end)
  '
}

# ── ACT 1 — research mode ─────────────────────────────────────────────────
act_research() {
  banner "ACT 1 — Research" \
    "Question: §138 NI Act — burden of proof on dishonour. Hits BM25 + Indian Kanoon → Gemini."

  local q="What are the leading Supreme Court rulings on Section 138 of the Negotiable Instruments Act regarding the burden of proof for dishonour of cheque, especially the presumption under §139 and its rebuttal? List the cases by name."

  curl -s -b "$COOKIE_JAR" -X POST "$BACKEND/api/brief/chat" \
       -H 'Content-Type: application/json' \
       -d "$(jq -nc --argjson tid "$THREAD_ID" --arg q "$q" \
            '{thread_id:$tid, question:$q, jurisdiction:"IN"}')" \
       | tee /tmp/sanhita-demo.act1.json | pretty_answer
}

# ── ACT 2 — drafting mode (citation carry-across) ────────────────────────
act_draft() {
  banner "ACT 2 — Drafting" \
    "Same thread. No retrieval. Citations from Act 1 ride along as factual scaffolding."

  local q="On the basis of the cases cited above, draft a §138 statutory notice from Acme Trading Pvt Ltd (Mumbai) to the drawer, M/s Bharat Industries (Pune), demanding ₹14,75,000 within 15 days. Use formal Indian legal-notice register, include cause of action paragraph, and cite the §139 presumption."

  curl -s -b "$COOKIE_JAR" -X POST "$BACKEND/api/brief/draft" \
       -H 'Content-Type: application/json' \
       -d "$(jq -nc --argjson tid "$THREAD_ID" --arg q "$q" \
            '{thread_id:$tid, question:$q, jurisdiction:"IN"}')" \
       | tee /tmp/sanhita-demo.act2.json | pretty_answer
}

# ── ACT 3 — agent mode (multi-step + Google tools) ───────────────────────
act_agent() {
  banner "ACT 3 — Automation agent" \
    "Multi-step: research recent rulings + draft a memo + save to Drive + queue Gmail draft + log a Sheet row."

  local q="Find the most recent Bombay High Court rulings on §138 NI Act dishonour where the drawer pleaded financial hardship as a defence (last 3 years). Then draft a 1-page client memo for our partner explaining (a) the prevailing position, (b) whether Acme should accept a settlement at ₹10L vs litigate, and (c) recommended next steps. Save the memo to my Google Drive as 'Acme v. Bharat — §138 memo', queue a Gmail draft to client@acmetrading.example with the memo body, and append a row to the Sanhita matter tracker for this case."

  curl -s -b "$COOKIE_JAR" -X POST "$BACKEND/api/brief/agent" \
       -H 'Content-Type: application/json' \
       --max-time 180 \
       -d "$(jq -nc --argjson tid "$THREAD_ID" --arg q "$q" \
            '{thread_id:$tid, question:$q, jurisdiction:"IN"}')" \
       | tee /tmp/sanhita-demo.act3.json \
       | jq -r '
           "── ANSWER ──\n" + (.answer_markdown // .answer // "(no answer)") + "\n\n" +
           "── TOOL CALLS ──\n" +
           ((.tool_trace // .agent_trace // []) | map(
             "• " + (.name // .tool // "?") +
             "  args=" + ((.args // {}) | tostring | .[0:120]) +
             (if .result and (.result | tostring | length) > 0
              then "\n    → " + ((.result | tostring) | .[0:160])
              else "" end)
           ) | join("\n")) + "\n\n" +
           "── CITATIONS (carried + new) ──\n" +
           ((.citations // []) | to_entries | map(
             "[" + ((.key + 1) | tostring) + "] " +
             (.value.title // .value.case // .value.url // "?")
           ) | join("\n")) + "\n\n" +
           "── META ──\n" +
           "turns:      " + ((.agent_turns // .turns // 0) | tostring) + "\n" +
           "provider:   " + ((.llm.provider // "?") | tostring)
         '
}

# ── orchestration ─────────────────────────────────────────────────────────
mode="${1:-all}"

login_or_mint_code
ensure_thread

case "$mode" in
  research) act_research ;;
  draft)    act_draft    ;;
  agent)    act_agent    ;;
  all)
    act_research
    act_draft
    act_agent
    ;;
  *)
    echo "Unknown mode '$mode'. Use: research | draft | agent | all" >&2
    exit 2
    ;;
esac

banner "Done" \
  "Open the React app at http://localhost:3001 → History → '§138 NI Act demo' to see this thread."
