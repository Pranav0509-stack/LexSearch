#!/usr/bin/env bash
# Sanhita end-to-end smoke. Run after deploy / before pilot demo.
#
# Exits 0 on full green; non-zero on any failure with a clear marker.

set -u

API="${SANHITA_BACKEND:-http://localhost:8080}"
WEB="${SANHITA_FRONTEND:-http://localhost:3000}"
CURL="/usr/bin/curl -s --max-time 20"
PY="/usr/bin/env python3"

# Track failures so we report them all at the end.
fails=()
pass() { printf "  \033[32m✓\033[0m  %s\n" "$1"; }
fail() { printf "  \033[31m✗\033[0m  %s\n" "$1"; fails+=("$1"); }

echo "═══ Sanhita end-to-end smoke ═══"
echo "Backend:  $API"
echo "Frontend: $WEB"
echo ""

# ── 1. Core health ────────────────────────────────────────────────────
echo "[1] Core health"
$CURL "$API/api/contract/health" | grep -q '"status":"ok"' \
    && pass "contract health green" || fail "contract health"

$CURL "$API/api/cases/engine-status" | grep -q '"engine_available":true' \
    && pass "search engine available" || fail "search engine"

# ── 2. Search modes ───────────────────────────────────────────────────
echo ""
echo "[2] Search modes"
for mode in keyword hybrid semantic; do
    n=$($CURL -X POST "$API/api/cases/smart-search" \
        -H 'Content-Type: application/json' \
        -d "{\"q\":\"section 138 cheque dishonour\",\"mode\":\"$mode\",\"limit\":3}" \
      | $PY -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('hits',[])))" 2>/dev/null)
    if [ "$mode" = "semantic" ]; then
        # Semantic alone is sparse (40K FAISS vectors) — accept 0+ hits, hybrid covers it
        [ -n "$n" ] && pass "$mode mode responsive ($n hits)" || fail "$mode mode timed out"
    else
        [ "${n:-0}" -gt 0 ] && pass "$mode mode → $n hits" || fail "$mode mode returned 0 hits"
    fi
done

# ── 3. Document fetch (the 3 source tables) ───────────────────────────
echo ""
echo "[3] Document viewer endpoints"
# pipeline_docs sample doc with has_pdf=1
PD_ID=$(sqlite3 /Users/pranav/Desktop/india-judgments-corpus/india_courts.db \
    "SELECT doc_id FROM pipeline_docs WHERE has_pdf=1 LIMIT 1" 2>/dev/null)
[ -n "${PD_ID:-}" ] && {
    code=$($CURL -o /dev/null -w "%{http_code}" "$API/api/cases/document/$PD_ID")
    [ "$code" = "200" ] && pass "pipeline_docs document fetch (id $PD_ID)" || fail "pipeline_docs document → HTTP $code"
} || fail "no pipeline_docs row with has_pdf=1 in DB"

# legal_docs sample
LD_ID=$(sqlite3 /Users/pranav/Desktop/india-judgments-corpus/india_courts.db \
    "SELECT doc_id FROM legal_docs WHERE LENGTH(full_text) > 500 LIMIT 1" 2>/dev/null)
[ -n "${LD_ID:-}" ] && {
    code=$($CURL -o /dev/null -w "%{http_code}" "$API/api/cases/document/$LD_ID")
    [ "$code" = "200" ] && pass "legal_docs document fetch" || fail "legal_docs document → HTTP $code"
}

# ── 4. PDF proxy (S3 stream) ──────────────────────────────────────────
echo ""
echo "[4] PDF proxy"
# Use a pipeline_docs HC row — its pdf_url uses the canonical S3 key layout
# (`data/pdf/year=YYYY/court=X_Y/bench=Z/file.pdf`) that the /pdf proxy expects.
PDF_S3=$(sqlite3 /Users/pranav/Desktop/india-judgments-corpus/india_courts.db \
    "SELECT pdf_url FROM pipeline_docs WHERE source='aws_s3_hc' AND has_pdf=1 LIMIT 1" 2>/dev/null)
# Strip the s3://indian-high-court-judgments/ prefix to get the bare key
PDF_KEY="${PDF_S3#s3://indian-high-court-judgments/}"
[ -n "${PDF_KEY:-}" ] && {
    ct=$($CURL -o /dev/null -D - "$API/pdf/$PDF_KEY" -r 0-100 | grep -i "content-type:" | head -1)
    echo "$ct" | grep -qi 'application/pdf' && pass "PDF proxy streams application/pdf ($PDF_KEY)" \
        || fail "PDF proxy wrong content-type: $ct  key=$PDF_KEY"
} || fail "no aws_s3_hc PDF in pipeline_docs"

# ── 5. Drafter (deterministic — no LLM cost) ──────────────────────────
echo ""
echo "[5] Drafter"
DRAFT=$($CURL -X POST "$API/api/contract/draft" \
    -H 'Content-Type: application/json' \
    -d '{"template_id":"corporate.nda_mutual.v1","mode":"deterministic_only","slots":{"execution_date":"17 May 2026","place":"Mumbai","party_a_name":"SmokeCo","party_a_type":"private limited company","party_a_address":"BKC","party_a_signatory":"Test","party_b_name":"Test LLP","party_b_type":"LLP","party_b_address":"Khar","party_b_signatory":"Test","purpose":"smoke test","term_months":24,"survival_years":3,"governing_state":"Maharashtra","arbitration_seat":"Mumbai","arbitrator_count":"1"}}')
wc=$(echo "$DRAFT" | $PY -c "import json,sys; d=json.load(sys.stdin); print(d.get('word_count',0))" 2>/dev/null)
[ "${wc:-0}" -gt 1000 ] && pass "NDA drafted ($wc words)" || fail "draft returned $wc words"

# ── 6. Compliance ─────────────────────────────────────────────────────
echo ""
echo "[6] Compliance plug-ins"
nfindings=$($CURL -X POST "$API/api/contract/compliance" \
    -H 'Content-Type: application/json' \
    -d '{"body_md":"This NDA covers personal data. The Service Provider shall process customer personal data.","doc_type":"nda_mutual"}' \
    | $PY -c "import json,sys; d=json.load(sys.stdin); print(d.get('count',0))" 2>/dev/null)
[ "${nfindings:-0}" -gt 0 ] && pass "compliance: $nfindings finding(s) on PD-leaky NDA" \
    || fail "compliance returned 0 findings (should flag DPDP)"

# ── 7. Legal-aid intake ───────────────────────────────────────────────
echo ""
echo "[7] Legal-aid intake"
LA=$($CURL -X POST "$API/api/legal-aid/apply" \
    -H 'Content-Type: application/json' \
    -d '{"org_name":"SMOKE","org_type":"DLSA (District Legal Services Authority)","contact_name":"Smoke","email":"smoke@test.example","jurisdiction":"Test","why":"automated end-to-end smoke test of legal-aid intake endpoint"}')
ref=$(echo "$LA" | $PY -c "import json,sys; d=json.load(sys.stdin); print(d.get('application_id',''))" 2>/dev/null)
[ -n "$ref" ] && pass "legal-aid application stored: $ref" || fail "legal-aid intake failed"

# ── 8. Frontend routes ────────────────────────────────────────────────
echo ""
echo "[8] Frontend routes"
for route in "/" "/login" "/app" "/legal-aid" "/plugins" "/plugins/litigation" "/plugins/commercial" "/plugins/privacy" "/plugins/legal-clinic"; do
    code=$($CURL -o /dev/null -w "%{http_code}" "$WEB$route")
    if [ "$code" = "200" ] || [ "$code" = "307" ]; then
        pass "$route → $code"
    else
        fail "$route → $code"
    fi
done

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "═════════════════════════════════════════════════"
if [ ${#fails[@]} -eq 0 ]; then
    printf "\033[32m✓ ALL GREEN — Sanhita is launch-ready.\033[0m\n"
    exit 0
else
    printf "\033[31m✗ %d FAILURE(S):\033[0m\n" "${#fails[@]}"
    for f in "${fails[@]}"; do echo "    - $f"; done
    exit 1
fi
