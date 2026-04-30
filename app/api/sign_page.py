"""
Mobile web eSignature page (plan §3.3).

Flow:
  GET  /s/{token}          → HTML page with PDF preview + name field + OTP input
  POST /s/{token}/request  → sends SMS OTP
  POST /s/{token}/verify   → verifies OTP, stamps PDF, returns signed PDF URL
"""

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter()

_SIGN_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NyayaSathi — Sign Document</title>
<style>
 body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 640px;
         margin: 0 auto; padding: 24px; }}
 .hint {{ color: #555; font-size: 14px; }}
 input {{ display: block; width: 100%; padding: 12px; margin: 8px 0;
           font-size: 16px; border: 1px solid #ccc; border-radius: 6px; }}
 button {{ padding: 12px 20px; font-size: 16px; border: 0; border-radius: 6px;
           background: #0b5; color: #fff; }}
 iframe {{ width: 100%; height: 60vh; border: 1px solid #ddd; border-radius: 6px; }}
</style></head>
<body>
  <h1>Sign your document</h1>
  <p class="hint">Not legal advice — legal information + drafting. See footer of PDF.</p>
  <iframe src="/s/{token}/preview"></iframe>
  <form id="f">
    <input name="signer_name" placeholder="Type your full legal name" required>
    <button type="button" onclick="req()">Send OTP</button>
    <input name="otp" placeholder="6-digit OTP" pattern="\\d{{6}}">
    <button type="submit">Sign</button>
  </form>
<script>
 async function req() {{
   await fetch('/s/{token}/request', {{ method: 'POST' }});
   alert('OTP sent to your registered mobile.');
 }}
 document.getElementById('f').onsubmit = async (e) => {{
   e.preventDefault();
   const fd = new FormData(e.target);
   const r = await fetch('/s/{token}/verify', {{ method: 'POST', body: fd }});
   const j = await r.json();
   if (j.ok) location.href = j.signed_url;
   else alert(j.error || 'Verification failed');
 }};
</script>
</body></html>"""


@router.get("/{token}", response_class=HTMLResponse)
async def sign_page(token: str) -> HTMLResponse:
    # TODO: validate token → document id; 404 if unknown.
    return HTMLResponse(_SIGN_HTML.replace("{token}", token))


@router.post("/{token}/request")
async def request_otp(token: str) -> dict:
    # TODO: generate OTP, SMS via Plivo, store in Redis with 10-min TTL.
    return {"ok": True}


@router.post("/{token}/verify")
async def verify_otp(
    token: str,
    signer_name: str = Form(...),
    otp: str = Form(...),
) -> JSONResponse:
    # TODO: verify Redis OTP; stamp_pdf_with_signature; upload signed; email.
    if len(otp) != 6 or not otp.isdigit():
        raise HTTPException(status_code=400, detail="invalid OTP format")
    return JSONResponse({"ok": True, "signed_url": "/s/placeholder"})


@router.get("/{token}/preview")
async def preview_pdf(token: str) -> dict:
    # TODO: stream the unsigned PDF from S3.
    return {"ok": True, "token": token}
