"use strict";

/**
 * citation-guard.js — Anti-hallucination citation grounding for NyayaSathi
 *
 * Extracts legal citations from LLM responses and verifies them against
 * the RAG chunks that were actually retrieved. Computes a grounding score
 * and sanitizes low-confidence responses.
 */

// ── Known Acts that exist in Indian law (used to whitelist Act-level citations) ──
const KNOWN_ACTS = [
  "BNS", "BNSS", "IPC", "CrPC", "CPC", "BSA",
  "RERA", "POCSO", "POSH",
  "DV Act", "Domestic Violence Act",
  "RTI Act", "Right to Information",
  "Consumer Protection Act",
  "CGST", "GST",
  "Income Tax Act",
  "Trade Marks Act", "Trademark",
  "Copyright Act",
  "Patents Act",
  "NMC", "Medical Council",
  "RTE Act", "Right to Education",
  "RPWD Act", "Rights of Persons with Disabilities",
  "Forest Rights Act", "FRA",
  "PESA",
  "Maintenance Act", "Senior Citizen",
  "Maternity Benefit Act",
  "Insurance Act", "IRDAI",
  "Prevention of Corruption", "PC Act",
  "Whistleblower", "WBP Act",
  "Lokpal",
  "FSS Act", "FSSAI", "Food Safety",
  "NGT Act", "National Green Tribunal",
  "UGC",
  "Labour",
  "Transfer of Property Act",
  "Registration Act",
  "Limitation Act",
  "Companies Act",
  "SEBI",
  "Negotiable Instruments Act",
  "Motor Vehicles Act",
  "Arbitration",
  // v10 additions
  "Constitution", "Hindu Marriage Act", "Muslim Women",
  "Indian Contract Act", "Code on Wages",
  "Payment of Gratuity", "RFCTLARR", "Land Acquisition",
  "Partnership Act", "Bonded Labour",
  "Industrial Disputes Act", "Transgender",
  "Digital Personal Data Protection", "IT Act",
  "Environmental Protection Act", "Forest Conservation Act",
  "Legal Services Authorities Act", "NALSA",
  "Street Vendors Act",
];

// ── Regex patterns ──
// English: "Section 138", "Sec. 138", "S.138"
const RE_SECTION = /\b(?:Section|Sec\.?|S\.)\s*(\d{1,4}[A-Z]?(?:\([a-z0-9]+\))?)/gi;
// Hindi Devanagari: "सेक्शन 138", "धारा 138"
const RE_SECTION_HI = /(?:सेक्शन|धारा)\s*(\d{1,4}[A-Z]?)/g;
// Hindi Article: "अनुच्छेद 21", "आर्टिकल 32"
const RE_ARTICLE_HI = /(?:अनुच्छेद|आर्टिकल)\s*(\d{1,3}[A-Z]?)/g;
// Hindi Act: "अधिनियम", "एक्ट", "कानून"
const RE_ACT_HI = /(?:अधिनियम|एक्ट)\s+(\d{4}|\S+)/g;
const RE_ACT = new RegExp(
  "\\b(" + KNOWN_ACTS.map(a => a.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")\\b",
  "gi"
);
const RE_DEADLINE = /within\s+(\d+)\s+(days?|hours?|months?|years?)/gi;
const RE_DEADLINE_HI = /(\d+)\s*(?:दिन|महीने?|साल|घंटे?)\s*(?:के\s+)?(?:अंदर|भीतर|में)/g;
const RE_PROCEDURE = /(?:file|submit|lodge)\s+(?:a\s+)?(?:complaint|petition|FIR|application|appeal|writ)\s+(?:at|before|with|in)\s+([^.,;\n]{5,40})/gi;
const RE_ARTICLE = /\b(?:Article)\s+(\d{1,3}[A-Z]?)/gi;

/**
 * extractCitations(responseText)
 * Returns an array of citation objects found in the LLM response.
 */
function extractCitations(responseText) {
  if (!responseText) return [];
  const citations = [];
  const seen = new Set();

  const addCitation = (type, raw, searchTerms) => {
    const key = `${type}:${raw.toLowerCase().trim()}`;
    if (!seen.has(key)) {
      seen.add(key);
      citations.push({ type, raw: raw.trim(), searchTerms });
    }
  };

  // Section numbers (English)
  let m;
  RE_SECTION.lastIndex = 0;
  while ((m = RE_SECTION.exec(responseText)) !== null) {
    const secNum = m[1];
    addCitation("section", m[0], [
      `section ${secNum}`.toLowerCase(),
      `sec ${secNum}`.toLowerCase(),
      `s.${secNum}`.toLowerCase(),
      secNum.toLowerCase(),
    ]);
  }

  // Section numbers (Hindi Devanagari: सेक्शन/धारा)
  RE_SECTION_HI.lastIndex = 0;
  while ((m = RE_SECTION_HI.exec(responseText)) !== null) {
    const secNum = m[1];
    addCitation("section", m[0], [
      `section ${secNum}`.toLowerCase(),
      `sec ${secNum}`.toLowerCase(),
      secNum.toLowerCase(),
    ]);
  }

  // Article numbers (English)
  RE_ARTICLE.lastIndex = 0;
  while ((m = RE_ARTICLE.exec(responseText)) !== null) {
    const artNum = m[1];
    addCitation("article", m[0], [`article ${artNum}`.toLowerCase(), artNum]);
  }

  // Article numbers (Hindi: अनुच्छेद/आर्टिकल)
  RE_ARTICLE_HI.lastIndex = 0;
  while ((m = RE_ARTICLE_HI.exec(responseText)) !== null) {
    const artNum = m[1];
    addCitation("article", m[0], [`article ${artNum}`.toLowerCase(), `अनुच्छेद ${artNum}`, artNum]);
  }

  // Act names
  RE_ACT.lastIndex = 0;
  while ((m = RE_ACT.exec(responseText)) !== null) {
    const raw = m[0];
    addCitation("act", raw, [raw.toLowerCase()]);
  }

  // Deadlines (English)
  RE_DEADLINE.lastIndex = 0;
  while ((m = RE_DEADLINE.exec(responseText)) !== null) {
    addCitation("deadline", m[0], [m[0].toLowerCase()]);
  }

  // Deadlines (Hindi: "30 दिन के अंदर")
  RE_DEADLINE_HI.lastIndex = 0;
  while ((m = RE_DEADLINE_HI.exec(responseText)) !== null) {
    addCitation("deadline", m[0], [m[0].toLowerCase(), `within ${m[1]}`]);
  }

  // Procedures: "file a complaint at Consumer Forum"
  RE_PROCEDURE.lastIndex = 0;
  while ((m = RE_PROCEDURE.exec(responseText)) !== null) {
    addCitation("procedure", m[0], [m[0].toLowerCase().slice(0, 40)]);
  }

  return citations;
}

/**
 * verifyCitation(citation, ragChunks)
 * Checks whether a citation's searchTerms appear in any of the retrieved RAG chunks.
 * Returns { verified: bool, matchedChunk: string|null }
 */
function verifyCitation(citation, ragChunks) {
  if (!ragChunks || ragChunks.length === 0) {
    return { verified: false, matchedChunk: null };
  }

  // Build cross-lingual search terms: धारा 138 ↔ Section 138
  const allTerms = [...citation.searchTerms];
  for (const term of citation.searchTerms) {
    // If Hindi section, add English equivalent
    const secMatch = term.match(/^(?:section|sec|s\.)\s*(\d+)/i);
    if (secMatch) {
      allTerms.push(`धारा ${secMatch[1]}`, `सेक्शन ${secMatch[1]}`);
    }
    // If just a number, search all forms
    if (/^\d{1,4}[a-z]?$/i.test(term)) {
      allTerms.push(`section ${term}`, `sec ${term}`);
    }
  }

  for (const chunk of ragChunks) {
    const chunkLower = chunk.toLowerCase();
    for (const term of allTerms) {
      if (chunkLower.includes(term.toLowerCase())) {
        return { verified: true, matchedChunk: chunk };
      }
    }
  }
  return { verified: false, matchedChunk: null };
}

/**
 * computeGroundingScore(responseText, ragChunks)
 * Returns {
 *   score: number (0–1),
 *   total: number,
 *   verified: number,
 *   unverified: Citation[]
 * }
 *
 * Score logic:
 *   - 0 citations found → 0.7 (moderate trust; no verifiable claims, no false ones)
 *   - Only act/deadline/procedure citations (no section numbers) → 0.75
 *   - Has section citations → verified_sections / total_sections
 *   - Combined score: weighted average (sections count most)
 */
function computeGroundingScore(responseText, ragChunks) {
  const citations = extractCitations(responseText);

  if (citations.length === 0) {
    return { score: 0.7, total: 0, verified: 0, unverified: [] };
  }

  const verifiedList = [];
  const unverifiedList = [];

  for (const cite of citations) {
    const result = verifyCitation(cite, ragChunks);
    if (result.verified) {
      verifiedList.push(cite);
    } else {
      unverifiedList.push(cite);
    }
  }

  // Section-level citations are highest risk — weight them more heavily
  const sectionCites = citations.filter(c => c.type === "section" || c.type === "article");
  const nonSectionCites = citations.filter(c => c.type !== "section" && c.type !== "article");

  let score;

  if (sectionCites.length === 0) {
    // No section numbers claimed — moderate trust
    const verifiedNonSection = nonSectionCites.filter(c =>
      verifyCitation(c, ragChunks).verified
    ).length;
    score = nonSectionCites.length > 0
      ? 0.6 + 0.35 * (verifiedNonSection / nonSectionCites.length)
      : 0.7;
  } else {
    const verifiedSections = sectionCites.filter(c =>
      verifyCitation(c, ragChunks).verified
    ).length;
    const sectionRatio = verifiedSections / sectionCites.length;

    const verifiedNonSection = nonSectionCites.length > 0
      ? nonSectionCites.filter(c => verifyCitation(c, ragChunks).verified).length / nonSectionCites.length
      : 1.0;

    // Sections: 70% weight, non-sections: 30% weight
    score = 0.7 * sectionRatio + 0.3 * verifiedNonSection;
  }

  return {
    score: Math.min(1.0, Math.max(0.0, score)),
    total: citations.length,
    verified: verifiedList.length,
    unverified: unverifiedList,
  };
}

/**
 * sanitizeResponse(responseText, ragChunks, lang, fallbackText)
 *
 * If grounding score >= 0.7 → return responseText unchanged.
 * If grounding score < 0.7 → try to salvage grounded sentences, else use fallbackText.
 *
 * Returns sanitized string.
 */
function sanitizeResponse(responseText, ragChunks, lang, fallbackText) {
  const { score, unverified } = computeGroundingScore(responseText, ragChunks);

  if (score >= 0.7) return responseText;

  // Try to extract sentences that don't contain unverified section numbers
  const unverifiedRaws = unverified
    .filter(c => c.type === "section" || c.type === "article")
    .map(c => c.raw.toLowerCase());

  if (unverifiedRaws.length === 0) {
    // Score was low for non-section reasons — trust it anyway (borderline case)
    return responseText;
  }

  // Split into sentences and keep those that don't cite unverified sections
  const sentences = responseText
    .split(/(?<=[.!?।])\s+/)
    .filter(s => s.trim().length > 5);

  const groundedSentences = sentences.filter(sentence => {
    const sLower = sentence.toLowerCase();
    return !unverifiedRaws.some(raw => sLower.includes(raw));
  });

  const groundedText = groundedSentences.join(" ").trim();
  const wordCount = groundedText.split(/\s+/).filter(Boolean).length;

  if (wordCount >= 20) {
    // Enough grounded content — use it with a helpline note
    const helplineNote = lang === "hi-IN"
      ? " Kisi bhi doubt ke liye, NALSA 15100 pe free legal sahayata lein."
      : " For specific legal advice, please call NALSA at 15100 for free assistance.";
    return groundedText + helplineNote;
  }

  // Not enough grounded content — use fallback
  return fallbackText ||
    (lang === "hi-IN"
      ? "Aapke sawaal ke liye sahi jaankari ke liye NALSA 15100 pe call karein. Yeh ek free legal aid helpline hai."
      : "For accurate legal information on this topic, please call NALSA at 15100. This is a free legal aid helpline.");
}

module.exports = {
  extractCitations,
  verifyCitation,
  computeGroundingScore,
  sanitizeResponse,
};
