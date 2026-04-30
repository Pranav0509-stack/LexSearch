# LexSearch — Scraping Policy

This document governs how LexSearch's `ingest/` scripts fetch public court
data. It exists so contributors, reviewers, and the courts themselves can
see exactly what we do.

## What we fetch

1. **Supreme Court of India** — daily judgments published on
   `main.sci.gov.in`. Public, openly linked, no login required.
2. **Top 6 High Courts** — daily orders / reportable judgments published
   on each court's official eCourts-linked portal:
   - Delhi (`dhcmisc.nic.in`)
   - Bombay (`bombayhighcourt.nic.in`)
   - Madras (`mhc.tn.gov.in`)
   - Karnataka (`karnatakajudiciary.kar.nic.in`)
   - Calcutta (`calcuttahighcourt.gov.in`)
   - Allahabad (`allahabadhighcourt.in`)
3. **Quarterly backfill** from the `vanga/indian-high-court-judgments`
   CC-BY-4.0 dataset on GitHub (derived from eCourts). No scraping here;
   just S3 copy.

## What we do *not* fetch

- Sealed / in-camera proceedings.
- Cause lists or case status pages behind CAPTCHA or login.
- Personal data of parties beyond what the judgment itself publishes.
- PDFs flagged "Not for publication" on the court's site.

## How we behave on the wire

- **Identification.** All requests send a `User-Agent` of the form
  `NyayaSathiIngest/1.0 (+https://nyaysathi-website.vercel.app; contact=pranavpandey.pr@gmail.com)`.
  Court IT teams can see who we are and how to reach us.
- **Rate limiting.** One request every 2 seconds per court adapter
  (`HC_DAILY_RATE_LIMIT_S`). SC pulls max once per day per date window.
- **Retries.** `tenacity` backoff, max 3 attempts, exponential 2-30s.
- **No JS execution.** We parse HTML with regex/BeautifulSoup only — we
  never drive a headless browser, never log in, never submit forms with
  fabricated session state.
- **robots.txt.** Our schedule respects the published `robots.txt` for
  each court site. If a court adds a `Disallow` for the paths we use, the
  adapter is removed in the next release.
- **No redistribution of raw PDFs.** We store extracted text + metadata;
  end users viewing a judgment on NyayaSathi are redirected to the
  court's own PDF URL.

## Data retention

Extracted text is retained indefinitely as part of the BM25 index so
answers remain reproducible. If a court issues a redaction or take-down,
open an issue or email the contact above and we will drop the row within
24 hours.

## Opt-out

Any court registry may request exclusion. We will comply immediately and
in writing. Contact: `pranavpandey.pr@gmail.com`.

## Why we do this

NyayaSathi is a free voice-first legal helpline. Callers — often rural,
first-generation litigants — describe a problem and get the correct
Section + a citation to an actual reported judgment. That grounding only
works if the underlying corpus is current. The alternative (no daily
updates) means the helpline gives outdated law. This pipeline is the
minimum required to meet that standard.
