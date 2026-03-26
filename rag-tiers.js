"use strict";

/**
 * rag-tiers.js — NyayaSathi Three-Tier RAG System
 *
 * Three separate corpora for different court tiers:
 *   - Supreme Court (SC): Constitutional law, landmark judgments, PILs, fundamental rights
 *   - High Court (HC): Writs (Art 226), bail, appeals, quashing, state-level matters
 *   - District Court (DC): Filing procedures, forms, timelines, consumer, family, labour
 *
 * Query classification routes queries to relevant tiers.
 * Results from all tiers are merged by score and top-k returned.
 */

// ═══════════════════════════════════════════════════════════
//  SUPREME COURT CORPUS — Constitutional law & landmark judgments
// ═══════════════════════════════════════════════════════════

const SC_CORPUS = [
    // ── Constitutional Articles ──
    {
        id: "sc-constitution-1", tags: ["fundamental rights", "constitution", "article 14", "article 19", "article 21", "equality", "freedom", "right to life"],
        text: `Fundamental Rights (Part III, Constitution of India): Article 14 — Right to Equality before law. Article 15 — Prohibition of discrimination on grounds of religion, race, caste, sex, or place of birth. Article 19 — Six freedoms: speech, assembly, association, movement, residence, and profession. Article 21 — Right to life and personal liberty — 'No person shall be deprived of his life or personal liberty except according to procedure established by law.' Article 22 — Protection against arrest and detention. These rights are enforceable through Supreme Court (Article 32) or High Court (Article 226).`
    },
    {
        id: "sc-constitution-2", tags: ["writ", "article 32", "habeas corpus", "mandamus", "supreme court", "fundamental rights enforcement"],
        text: `Supreme Court Writs under Article 32: (1) Habeas Corpus — 'produce the body'. Used when someone is illegally detained. File directly in SC or HC. (2) Mandamus — command to government/public authority to perform duty. (3) Certiorari — quash order of lower court/tribunal. (4) Prohibition — stop lower court from exceeding jurisdiction. (5) Quo Warranto — challenge someone holding public office without authority. Article 32 is itself a fundamental right — SC called it 'the very soul of the Constitution'.`
    },
    {
        id: "sc-constitution-3", tags: ["directive principles", "article 39", "article 41", "article 46", "legal aid", "free lawyer"],
        text: `Directive Principles of State Policy (Part IV): Article 39(a) — Right to adequate livelihood. Article 39A — Free legal aid for poor. This led to NALSA (Legal Services Authorities Act 1987). Article 41 — Right to work, education, public assistance. Article 43 — Living wage for workers. Article 46 — Promote educational/economic interests of SC/ST/weaker sections. These are not directly enforceable but courts read them with Part III to expand fundamental rights.`
    },

    // ── PIL (Public Interest Litigation) ──
    {
        id: "sc-pil-1", tags: ["pil", "public interest", "article 32", "letter petition", "locus standi"],
        text: `Public Interest Litigation (PIL) in Supreme Court: (1) Any citizen can file PIL under Article 32 for violation of fundamental rights of a group. (2) No need to be personally affected (relaxed locus standi). (3) Even a postcard/letter to Chief Justice is accepted. (4) Filing: Draft petition stating violation, parties affected, relief sought. (5) No court fee in most cases. (6) Famous PILs: Bandhua Mukti Morcha (bonded labour), MC Mehta (pollution), Vishaka (workplace harassment). (7) Cannot file PIL for personal gain — court imposes costs on frivolous PILs.`
    },

    // ── Landmark Judgments ──
    {
        id: "sc-maneka-gandhi", tags: ["article 21", "right to life", "due process", "fair trial", "maneka gandhi", "passport"],
        text: `Maneka Gandhi v. Union of India (1978): Expanded Article 21 — 'right to life' includes right to live with dignity, right to travel abroad, right to privacy, right to livelihood. The procedure depriving life/liberty must be 'fair, just, and reasonable' (not merely any law). This transformed Article 21 from a narrow protection into the most powerful fundamental right. Used in: right to clean environment, right to education, right to health, right to shelter.`
    },
    {
        id: "sc-kesavananda", tags: ["basic structure", "constitution", "kesavananda bharati", "amendment", "parliament power"],
        text: `Kesavananda Bharati v. State of Kerala (1973): Established the 'Basic Structure Doctrine' — Parliament can amend the Constitution but cannot alter its basic structure. Basic structure includes: (1) Supremacy of Constitution, (2) Republican and democratic form of government, (3) Secular character, (4) Separation of powers, (5) Federal character, (6) Judicial review, (7) Fundamental rights. This is the most important constitutional case — prevents any government from destroying democracy through constitutional amendments.`
    },
    {
        id: "sc-vishaka", tags: ["sexual harassment", "workplace", "vishaka", "posh", "icc", "women safety"],
        text: `Vishaka v. State of Rajasthan (1997): SC laid down binding guidelines against sexual harassment at workplace. Led to POSH Act 2013. Key rules: (1) Every employer must set up Internal Complaints Committee (ICC). (2) ICC must have external women's rights member. (3) Complaint within 3 months (extendable by 3 months). (4) ICC must complete inquiry within 90 days. (5) Employer who fails to constitute ICC: fine up to ₹50,000. (6) Complaint to District Officer if no ICC or if employer is the accused.`
    },
    {
        id: "sc-dk-basu", tags: ["custodial torture", "police brutality", "dk basu", "arrest guidelines", "lockup death"],
        text: `D.K. Basu v. State of West Bengal (1997): SC's 11-point guidelines for arrest: (1) Police must carry accurate identification. (2) Arrest memo must be signed by witness (family/respectable person). (3) Arrestee has right to inform family/friend. (4) Arrestee must be medically examined every 48 hours. (5) Copies of documents to be sent to local Magistrate. (6) Lawyer access from time of arrest. (7) No torture/third-degree — violation is contempt of SC. (8) If custodial death — mandatory judicial inquiry + FIR against officers. Compensation for custodial torture ordered by SC.`
    },
    {
        id: "sc-puttaswamy", tags: ["privacy", "right to privacy", "puttaswamy", "aadhaar", "data protection"],
        text: `K.S. Puttaswamy v. Union of India (2017): SC held Right to Privacy is a fundamental right under Article 21. (1) Privacy includes bodily autonomy, informational privacy, personal choices. (2) Any restriction must pass: (a) legality — prescribed by law, (b) legitimate aim, (c) proportionality. (3) Led to Digital Personal Data Protection Act 2023. (4) Aadhaar partially upheld but struck down Section 57 (private entities cannot mandate Aadhaar). Used to challenge: mass surveillance, DNA profiling, phone tapping without judicial warrant.`
    },
    {
        id: "sc-navtej", tags: ["section 377", "lgbtq", "homosexuality", "navtej johar", "decriminalization"],
        text: `Navtej Singh Johar v. Union of India (2018): SC decriminalized homosexuality — struck down Section 377 IPC (now inapplicable under BNS). (1) Consensual sexual acts between adults in private are no longer criminal. (2) LGBTQ+ community has equal protection under Articles 14, 15, 19, 21. (3) Constitution recognizes right to sexual orientation as part of right to life and dignity. (4) 'History owes an apology to the LGBTQ+ community.' Section 377 still applies to: non-consensual acts, acts with minors, bestiality.`
    },
    {
        id: "sc-lalita-kumari", tags: ["fir", "mandatory registration", "lalita kumari", "cognizable offence", "police duty"],
        text: `Lalita Kumari v. Government of UP (2013): SC made FIR registration MANDATORY for cognizable offences. Key holdings: (1) Police MUST register FIR on receiving information of cognizable offence — no preliminary inquiry allowed. (2) If police refuse: complaint to SP/DIG, or file private complaint under CrPC 156(3)/BNSS before Magistrate. (3) Preliminary inquiry allowed ONLY for matrimonial/family disputes, commercial offences, medical negligence, corruption cases. (4) FIR registration is a legal duty — refusal is punishable.`
    },
    {
        id: "sc-arnesh-kumar", tags: ["arrest", "498a", "cruelty", "husband", "dowry", "arnesh kumar", "guidelines arrest"],
        text: `Arnesh Kumar v. State of Bihar (2014): SC restricted automatic arrest in Section 498A IPC (now BNS 85/86) dowry harassment cases. Guidelines: (1) Police must not automatically arrest — must examine if arrest is necessary. (2) Magistrate must examine necessity before authorizing detention. (3) Checklist to assess: is accused likely to flee, tamper evidence, threaten witnesses? (4) Applies to ALL offences with punishment up to 7 years. (5) If guidelines violated: departmental action + contempt of court. This prevents misuse of 498A for harassment.`
    },
    {
        id: "sc-transgender", tags: ["transgender", "third gender", "nalsa", "hijra", "gender identity"],
        text: `NALSA v. Union of India (2014): SC recognized transgender persons as 'Third Gender' with full constitutional rights. (1) Right to self-identified gender — no surgery required. (2) Entitled to reservation as socially/educationally backward class. (3) Government must provide education, health, employment to transgender persons. Led to Transgender Persons (Protection of Rights) Act 2019. Rights: (a) No discrimination in education, employment, healthcare. (b) Right to reside in household. (c) Pension and social security.`
    },
    {
        id: "sc-bommai", tags: ["president rule", "article 356", "governor", "state government", "sr bommai", "floor test"],
        text: `S.R. Bommai v. Union of India (1994): SC restricted President's Rule under Article 356. Holdings: (1) President's proclamation is judicially reviewable. (2) State government cannot be dismissed on subjective 'Governor's satisfaction'. (3) Floor test is the only way to determine majority — not Governor's opinion. (4) Secularism is basic structure — violation can justify Article 356. (5) State assembly dissolution before parliamentary approval is unconstitutional. This case protects federalism and prevents misuse of Article 356.`
    },
    {
        id: "sc-shayara-bano", tags: ["triple talaq", "muslim divorce", "shayara bano", "instant talaq", "muslim women"],
        text: `Shayara Bano v. Union of India (2017): SC declared instant triple talaq (talaq-e-biddat) unconstitutional. (1) Saying 'talaq' three times at once is void. (2) Muslim Women (Protection of Rights on Marriage) Act 2019 makes it criminal — 3 years imprisonment. (3) Wife entitled to subsistence allowance + custody. (4) Only valid forms: Talaq-e-Ahsan (single pronouncement + 3-month iddat) and Talaq-e-Hasan (3 pronouncements over 3 months). (5) Woman can seek maintenance under DV Act 2005 Section 12 or CrPC 125/BNSS.`
    },
    {
        id: "sc-mc-mehta", tags: ["pollution", "absolute liability", "mc mehta", "hazardous industry", "factory pollution"],
        text: `M.C. Mehta v. Union of India (1987): SC established 'Absolute Liability' doctrine for hazardous industries. (1) Enterprise engaged in hazardous activity is absolutely liable for harm — no exceptions (stricter than Rylands v Fletcher). (2) Compensation must be proportional to company's financial capacity. (3) Right to clean environment is part of Article 21. Used in: Oleum gas leak case, Ganga pollution cleanup, vehicular pollution in Delhi. Led to: Environment Protection Act enforcement, National Green Tribunal creation.`
    },
    {
        id: "sc-olga-tellis", tags: ["right to livelihood", "street vendor", "hawker", "eviction", "olga tellis", "pavement dweller"],
        text: `Olga Tellis v. Bombay Municipal Corporation (1985): SC held right to livelihood is part of Article 21. (1) Pavement dwellers cannot be evicted without due process. (2) Right to life includes right to livelihood — deprivation of livelihood is deprivation of life. (3) Eviction must follow natural justice — notice, hearing, alternative arrangement. Led to Street Vendors (Protection of Livelihood) Act 2014. Hawkers/vendors must get: (a) Certificate of vending, (b) Designated vending zones, (c) Protection from harassment/eviction.`
    },
    {
        id: "sc-pudr", tags: ["bonded labour", "minimum wage", "forced labour", "pudr", "child labour", "article 23"],
        text: `PUDR v. Union of India (1982): SC expanded Article 23 — any labour paid below minimum wage is 'forced labour'. (1) Even if worker 'consented', sub-minimum wage = forced labour. (2) Government must identify, release, and rehabilitate bonded labourers. (3) Bonded Labour System (Abolition) Act 1976 — bonded labour is criminal offence. (4) Child Labour (Prohibition) Act 1986 — no child below 14 in hazardous employment. (5) Rehabilitation: ₹20,000 under Central scheme. (6) Complaint: District Magistrate or Labour Department.`
    },
    // ── BNS↔IPC Mapping ──
    {
        id: "sc-bns-map-1", tags: ["bns", "ipc", "conversion", "old law new law", "bharatiya nyaya sanhita", "section mapping"],
        text: `BNS 2023 ↔ IPC Section Mapping (Part 1 — most used): IPC 302 (Murder) → BNS 101. IPC 304 (Culpable homicide) → BNS 105. IPC 304A (Death by negligence) → BNS 106. IPC 306 (Abetment of suicide) → BNS 108. IPC 354 (Outraging modesty) → BNS 74. IPC 376 (Rape) → BNS 63. IPC 379 (Theft) → BNS 303. IPC 380 (Theft in dwelling) → BNS 305. IPC 406 (Criminal breach of trust) → BNS 316. IPC 420 (Cheating) → BNS 318. IPC 498A (Cruelty by husband) → BNS 85. IPC 499/500 (Defamation) → BNS 356. IPC 506 (Criminal intimidation) → BNS 351. IPC 307 (Attempt to murder) → BNS 109. IPC 323 (Voluntarily causing hurt) → BNS 115. IPC 326 (Grievous hurt) → BNS 117.`
    },
    {
        id: "sc-bns-map-2", tags: ["bns", "ipc", "forgery", "criminal intimidation", "cheating", "theft"],
        text: `BNS 2023 ↔ IPC Section Mapping (Part 2): IPC 363 (Kidnapping) → BNS 137. IPC 366 (Kidnapping woman) → BNS 139. IPC 376D (Gang rape) → BNS 70. IPC 377 (Unnatural offences) → Partially repealed. IPC 395 (Dacoity) → BNS 310. IPC 399 (Preparation for dacoity) → BNS 311. IPC 411 (Receiving stolen property) → BNS 317. IPC 463 (Forgery) → BNS 336. IPC 467 (Forgery of valuable security) → BNS 338. IPC 468 (Forgery for cheating) → BNS 340. IPC 471 (Using forged document) → BNS 341. IPC 120B (Criminal conspiracy) → BNS 61. IPC 34 (Common intention) → BNS 3(5). IPC 149 (Unlawful assembly) → BNS 190. Note: BNS adds new offences not in IPC — mob lynching (BNS 103), organized crime (BNS 111), terrorism (BNS 113).`
    },

    // ── Case Studies — SC ──
    {
        id: "sc-case-pil-environment", tags: ["pil", "case study", "outcome", "environment", "pollution", "mc mehta"],
        text: `Case Study — PIL for Environment (MC Mehta v. Union of India): MC Mehta, a lawyer, filed PIL about Ganga pollution and Delhi air pollution. SC ordered closure of polluting tanneries in Kanpur, relocation of hazardous industries from Delhi, introduction of CNG buses in Delhi. Outcome: 168 factories closed/relocated, CNG mandate saved thousands of lives. Timeline: Filed 1985, orders spanning 1987-2004. Key lesson: One person's PIL can change national policy. No court fee, no personal injury needed — just public interest.`
    },
    {
        id: "sc-case-right-to-education", tags: ["case study", "outcome", "right to education", "article 21a", "rte", "fundamental right"],
        text: `Case Study — Right to Education (Unnikrishnan v. State of AP, 1993): Students challenged capitation fees in private colleges. SC held: Right to education up to age 14 is a fundamental right under Article 21. Led to 86th Constitutional Amendment (2002) adding Article 21A. Then Right to Education Act 2009 — free and compulsory education for ages 6-14. Outcome: 25% seats in private schools reserved for EWS children. Over 1 crore children benefited. Timeline: Judgment 1993, Amendment 2002, RTE Act 2009. Key: SC judgment → Constitutional amendment → Parliament legislation.`
    },
    {
        id: "sc-case-privacy-puttaswamy", tags: ["case study", "outcome", "privacy", "aadhaar", "article 21", "puttaswamy"],
        text: `Case Study — Right to Privacy (KS Puttaswamy v. Union of India, 2017): Retired HC judge challenged Aadhaar mandatory linking with bank accounts and phones. SC 9-judge bench unanimously held: Right to Privacy is a fundamental right under Article 21. Government cannot force citizens to share biometric data without safeguards. Outcome: Aadhaar made optional for bank accounts, mobile phones. Mandatory only for government subsidies and income tax. Timeline: Filed 2012, landmark judgment 2017, Aadhaar Act upheld with modifications 2018. Key: Even after government policy, citizens' privacy is protected — challenge in SC under Article 32.`
    },
    {
        id: "sc-case-triple-talaq", tags: ["case study", "outcome", "triple talaq", "muslim women", "article 14", "shayara bano"],
        text: `Case Study — Triple Talaq (Shayara Bano v. Union of India, 2017): Shayara Bano's husband gave instant triple talaq via WhatsApp. She challenged this practice in SC. SC held 3:2 — instant triple talaq is unconstitutional, violates Article 14 (equality). Outcome: Muslim Women (Protection of Rights on Marriage) Act 2019 passed — triple talaq is now a criminal offence, husband can get up to 3 years jail. Wife entitled to maintenance and custody. Timeline: Petition 2016, SC judgment August 2017, Act passed July 2019. Key: Even personal law practices can be challenged if they violate fundamental rights.`
    },
];

// ═══════════════════════════════════════════════════════════
//  HIGH COURT CORPUS — Writs, bail, appeals, state-level
// ═══════════════════════════════════════════════════════════

const HC_CORPUS = [
    // ── Article 226 Writs ──
    {
        id: "hc-writ-1", tags: ["writ", "article 226", "high court", "mandamus", "certiorari", "habeas corpus"],
        text: `High Court Writs under Article 226: Broader than SC — can issue writs for ANY legal right (not just fundamental rights). (1) Habeas Corpus — challenge illegal detention/arrest. (2) Mandamus — compel government to act (e.g., issue passport, give pension). (3) Certiorari — quash illegal orders of tribunals/lower courts. (4) Prohibition — stop tribunal from exceeding jurisdiction. (5) Quo Warranto — challenge public office holder. Filing: Engage advocate, draft petition, pay court fee (~₹500-2000), file in HC registry. Time: Usually 2-4 weeks for admission.`
    },
    {
        id: "hc-writ-2", tags: ["article 226", "government order", "arbitrary action", "natural justice", "quash"],
        text: `When to file HC Writ Petition: (1) Government refuses to act on your application (mandamus). (2) Arbitrary transfer/suspension of government employee. (3) University/board refuses results or admission unfairly. (4) Municipal body demolishes without notice. (5) Environmental pollution not addressed by authorities. (6) Challenge unconstitutional state law. (7) Revenue department action without hearing. Requirements: Violation of legal/fundamental right, no alternative remedy (or alternative remedy is too slow/ineffective). Court fee: ₹500-2000. Need advocate-on-record (or PIL can be filed in person).`
    },

    // ── Bail in High Court ──
    {
        id: "hc-bail-1", tags: ["bail", "regular bail", "high court", "sessions court", "non-bailable"],
        text: `Bail from High Court — When Sessions Court refuses: (1) Regular Bail (BNSS Sec 480): Apply after arrest. If Sessions Court rejects, appeal to HC. (2) Grounds for bail: not flight risk, cooperating with investigation, no tampering, personal bonds, illness, women/children. (3) HC can grant bail even for serious offences if trial is delayed beyond reasonable time. (4) Surety amount decided by court. (5) Bail conditions: surrender passport, mark attendance, don't leave jurisdiction. (6) Violation of bail conditions → bail cancellation. Apply through advocate.`
    },
    {
        id: "hc-bail-2", tags: ["anticipatory bail", "pre-arrest bail", "section 482", "bnss 482"],
        text: `Anticipatory Bail (BNSS Sec 482, formerly CrPC 438): Apply BEFORE arrest when you reasonably apprehend arrest. (1) File in Sessions Court first; if rejected, appeal to HC. (2) Grounds: false case, malicious prosecution, no evidence, non-cognizable offence wrongly shown as cognizable. (3) HC can impose conditions: cooperate with investigation, don't leave India, join investigation when called. (4) Duration: usually 'till charge sheet' or specific period. (5) Cannot get anticipatory bail for: POCSO, SC/ST Atrocities Act (unless special circumstances), rape.`
    },
    {
        id: "hc-bail-3", tags: ["default bail", "statutory bail", "charge sheet delay", "bnss 479"],
        text: `Default/Statutory Bail (BNSS Sec 479, formerly CrPC 167(2)): Automatic right to bail if police don't file charge sheet within: (1) 60 days — for offences up to 3 years imprisonment. (2) 90 days — for offences above 3 years. (3) 180 days — for death penalty/life imprisonment cases. You MUST apply for default bail before charge sheet is filed. Once charge sheet filed, right expires. Application: File before Magistrate/Sessions Court. If refused, approach HC. This is a fundamental right — cannot be denied if conditions are met.`
    },

    // ── Criminal Revision / Quashing ──
    {
        id: "hc-quashing-1", tags: ["quashing", "section 482", "bnss 528", "fir quash", "high court", "inherent powers"],
        text: `Quashing of FIR/Criminal Case (BNSS Sec 528, formerly CrPC 482): HC can quash FIR or criminal proceedings using inherent powers. Grounds: (1) Offence is not made out from FIR itself. (2) Case is frivolous/vexatious. (3) Parties have settled (for compoundable offences). (4) Continuation of proceedings would be abuse of process. (5) Matrimonial disputes settled by parties (SC in Gian Singh). Requirements: File petition u/s 528 BNSS before HC. Engage senior advocate. Court fee ~₹1000-2000. Cannot quash: non-compoundable offences (murder, rape) generally.`
    },
    {
        id: "hc-revision-1", tags: ["criminal revision", "revision petition", "bnss", "sessions court order"],
        text: `Criminal Revision in High Court (BNSS Sec 442): Challenge orders of Sessions/Magistrate Court that cannot be appealed. (1) Revision lies when: lower court acted illegally, with material irregularity, or failed to exercise jurisdiction. (2) Time limit: Usually 90 days from order. (3) Cannot re-appreciate evidence in revision. (4) Examples: challenge bail rejection, challenge charge framing, challenge maintenance order. (5) Filing: Petition + certified copy of impugned order + court fee. Need advocate.`
    },

    // ── Appeals ──
    {
        id: "hc-appeal-1", tags: ["appeal", "criminal appeal", "conviction appeal", "sentence appeal", "high court"],
        text: `Criminal Appeals in High Court: (1) Appeal against conviction by Sessions Court — file within 30 days (BNSS). (2) Appeal against acquittal — only by State/prosecution or victim (not accused). (3) Appeal on sentence — if sentence is excessive or inadequate. (4) Suspension of sentence pending appeal — apply for bail during appeal. (5) Court fee varies (₹500-5000). (6) Death sentence — automatic appeal/confirmation by HC. (7) Need certified copy of judgment + grounds of appeal. Engage HC advocate.`
    },
    {
        id: "hc-appeal-civil", tags: ["civil appeal", "first appeal", "second appeal", "regular first appeal"],
        text: `Civil Appeals from District Court to HC: (1) Regular First Appeal (RFA) — against decree of civil judge. File within 30 days. (2) Second Appeal (CPC Sec 100) — only on 'substantial question of law'. Very limited. (3) Execution appeal — against order in execution proceedings. (4) Matrimonial appeal — within 30 days of family court order. (5) Court fee: Ad valorem (based on suit value) or fixed. (6) Stay of decree: Apply separately, court may impose condition (deposit amount). (7) HC can increase/decrease damages awarded by lower court.`
    },

    // ── State-level matters ──
    {
        id: "hc-dlsa-1", tags: ["dlsa", "district legal services", "free lawyer", "legal aid", "eligibility", "state helpline"],
        text: `State DLSA (District Legal Services Authority) — Free Legal Aid: Eligible: (1) SC/ST, (2) Women, (3) Children, (4) Persons with disability, (5) Industrial workers, (6) Victims of mass disaster/trafficking, (7) Income below ₹3 lakh/year (₹5 lakh in some states), (8) Persons in custody. How: Visit nearest DLSA office (in every District Court complex) or call state helpline. Major state helplines: Delhi — 1516. UP/MP/Bihar/Rajasthan — 15100 (NALSA). Karnataka — 1800-200-7878. Tamil Nadu — 1800-599-0019. Always free — no fees for legal aid cases.`
    },
    {
        id: "hc-lokadalat-1", tags: ["lok adalat", "settlement", "compromise", "adr", "legal services", "no appeal"],
        text: `Lok Adalat (People's Court) — Fast & Free Settlement: (1) Organized by DLSA/SLSA under Legal Services Authorities Act 1987. (2) Handles: motor accident claims, matrimonial disputes, labour disputes, utility bill disputes, bank recovery cases. (3) No court fee — even if pending court case, fee is refunded. (4) Award is deemed decree of civil court — enforceable but NO appeal. (5) National Lok Adalat held quarterly — massive clearance of cases. (6) Permanent Lok Adalat — for public utility services (electricity, water, telecom). (7) Both parties must agree to settlement.`
    },
    {
        id: "hc-cyber-jurisdiction", tags: ["cyber crime", "jurisdiction", "it act", "where to file", "online crime"],
        text: `Cyber Crime Jurisdiction & Filing: (1) FIR can be filed at ANY police station (Zero FIR) — no jurisdiction issue. (2) Complaint on cybercrime.gov.in — auto-routed to correct state. (3) IT Act Sec 46 — adjudicating officer is Secretary, IT Department of state. (4) For compensation >₹5 crore — approach Cyber Appellate Tribunal. (5) Appeal against cyber adjudicating officer — to High Court within 45 days. (6) State cyber crime cells: call 1930 helpline, they connect to state cell. (7) Preserve evidence: screenshots, URLs, transaction IDs, chat logs — before filing.`
    },
    {
        id: "hc-service-1", tags: ["government job", "service matter", "cat", "transfer", "suspension", "promotion"],
        text: `Government Service Matters — CAT & HC: (1) Central Administrative Tribunal (CAT) handles central government employee disputes. (2) State Administrative Tribunals for state employees. (3) Appeal from CAT/SAT directly to HC (Division Bench). (4) Matters: transfer, promotion, suspension, disciplinary action, pension. (5) Limitation: Usually 1 year from order. (6) Filing: OA (Original Application) in tribunal. Court fee: ₹50-500. (7) No advocate needed in tribunal (can argue in person). (8) Interim relief: Stay of transfer/suspension pending hearing.`
    },

    // ── Case Studies — HC ──
    {
        id: "hc-case-bail-granted", tags: ["bail", "case study", "outcome", "anticipatory bail", "section 438"],
        text: `Case Study — Anticipatory Bail Granted (HC): Ravi, a small shopkeeper, was falsely accused of cheating (IPC 420/BNS 318) by a business rival. Before arrest, his lawyer filed anticipatory bail under BNSS Section 482 (old CrPC 438) in High Court. HC noted: (1) No prior criminal record, (2) Dispute was civil in nature, (3) No flight risk. Outcome: Anticipatory bail granted with conditions — surrender passport, appear before IO weekly. Case later settled in Lok Adalat. Timeline: Bail application to order — 12 days. Cost: Advocate fee ₹15,000-25,000. Key: File anticipatory bail BEFORE arrest if you fear false case.`
    },
    {
        id: "hc-case-writ-pension", tags: ["writ", "case study", "outcome", "pension", "mandamus", "government"],
        text: `Case Study — Writ for Pension (HC Mandamus): Retired teacher Kamla Devi's pension was stopped after 3 years without reason. She filed writ petition under Article 226 in HC seeking mandamus. HC held: Pension is a right, not a bounty — cannot be stopped without show cause notice and hearing (natural justice). Outcome: HC directed department to restore pension within 4 weeks with 9% interest on arrears. Total arrears recovered: ₹4.2 lakh. Timeline: Filing to judgment — 6 weeks. Cost: Court fee ₹500, advocate ₹10,000. Key: Government cannot stop pension/salary without giving you a hearing first.`
    },
    {
        id: "hc-case-quashing-498a", tags: ["quashing", "case study", "outcome", "498a", "matrimonial", "section 482"],
        text: `Case Study — Quashing of 498A Case (HC): After mutual divorce, wife's family had filed IPC 498A (cruelty/BNS 85) case against husband's family including elderly parents. Husband filed quashing petition under BNSS Section 528 (old CrPC 482) in HC. HC noted: (1) Mutual divorce already granted, (2) Settlement deed signed, (3) Continuing case against elderly parents serves no purpose. Outcome: FIR quashed for all accused. Timeline: 3 months from filing to quashing. Cost: ₹30,000-50,000 total. Key: After mutual settlement, 498A cases can be quashed by HC — especially for distant relatives and elderly family members.`
    },
    {
        id: "hc-case-rera-builder", tags: ["rera", "case study", "outcome", "builder", "delay", "flat", "refund", "possession"],
        text: `Case Study — RERA Builder Delay (HC Writ): Buyer Meena paid ₹52 lakh for a flat in 2018 with promised possession by 2020. Builder delayed 3 years, kept demanding additional charges. Steps: (1) Filed complaint on RERA portal (free), (2) RERA authority ordered refund with 10.5% interest, (3) Builder didn't comply, (4) Filed writ in HC under Article 226. HC upheld RERA order, imposed ₹2 lakh cost on builder. Outcome: Full refund ₹52 lakh + ₹16.4 lakh interest + ₹2 lakh compensation = ₹70.4 lakh total. Timeline: RERA order 6 months, HC writ 4 months. Key: RERA complaints are free, fast, and enforceable. If builder doesn't comply, HC enforces with costs.`
    },
    {
        id: "hc-case-anticipatory-bail", tags: ["anticipatory bail", "case study", "outcome", "cheque bounce", "section 438", "ni act"],
        text: `Case Study — Anticipatory Bail for Cheque Bounce: Businessman Ravi issued cheque of ₹8 lakh which bounced. Payee filed case under NI Act Section 138 (criminal). Ravi feared arrest, applied for anticipatory bail under BNSS Section 482 (old CrPC 438) in Sessions Court — rejected. Filed in HC. HC granted anticipatory bail with conditions: (1) Deposit 20% of cheque amount (₹1.6 lakh) in court, (2) Appear before trial court on all dates, (3) Not tamper with evidence. Outcome: Ravi got bail, later settled case by paying ₹6 lakh (negotiated down). Timeline: HC bail in 2 weeks. Cost: Advocate ₹15,000-25,000. Key: Cheque bounce is compoundable — can be settled by paying. Anticipatory bail protects from arrest during trial.`
    },
];

// ═══════════════════════════════════════════════════════════
//  DISTRICT COURT CORPUS — Filing procedures, forms, timelines
// ═══════════════════════════════════════════════════════════

const DC_CORPUS = [
    // ── FIR & Police ──
    {
        id: "dc-fir-1", tags: ["fir", "police", "complaint", "thana", "daroga"],
        text: `FIR (First Information Report) filing at Police Station: Under BNSS 2023 Section 173 (formerly CrPC 154), any person can give information about a cognizable offence to police in writing or orally. Police MUST register FIR — they cannot refuse. If refused: (1) Send written complaint to SP by registered post. (2) File private complaint under BNSS Section 175 before Magistrate. (3) File Zero FIR at any police station regardless of jurisdiction — must be transferred to correct station within 24 hours. National helpline: 112. NALSA legal aid: 15100.`
    },
    {
        id: "dc-fir-2", tags: ["zero fir", "transfer", "jurisdiction", "online fir", "e-fir"],
        text: `Zero FIR & Online FIR: (1) Zero FIR: File at ANY police station regardless of where crime occurred. Station must accept and transfer within 24 hours. (2) E-FIR/Online FIR: Most states allow filing FIR online for theft, vehicle theft, lost documents. Websites: Delhi — delhipolice.nic.in, UP — uppolice.gov.in, Maharashtra — citizen.mahapolice.gov.in. (3) After online FIR: Visit police station within 3 days with original documents. (4) FIR copy: Police must give you FREE copy of FIR (your right under law). (5) FIR tracking: Use case number on state police portal.`
    },

    // ── Bail at District Level ──
    {
        id: "dc-bail-1", tags: ["bail", "arrest", "bailable", "police station bail"],
        text: `Bail at Police Station & District Court: (1) Bailable offences — bail as of RIGHT at police station. Police cannot refuse. No need to go to court. (2) Non-bailable — apply to District/Sessions Court. (3) Documents needed: ID proof, address proof, surety bond, FIR copy. (4) Surety: Usually 1-2 persons with fixed assets. Court decides surety amount. (5) Bail hearing: Usually same day or next day. (6) If Sessions Court refuses: appeal to High Court. (7) Interim bail: Can be granted same day in urgent cases. Free lawyer through DLSA if you cannot afford one.`
    },

    // ── Consumer Forum (District Level) ──
    {
        id: "dc-consumer-1", tags: ["consumer", "refund", "defective", "product", "warranty", "forum", "commission", "edaakhil"],
        text: `Consumer Forum — District Consumer Disputes Redressal Commission: (1) Jurisdiction: Claims up to ₹1 crore. (2) Filing: Online on edaakhil.nic.in or in person at District Forum. (3) Court fee: ₹100-₹5000 based on claim value. (4) Documents: Bill/receipt, warranty card, complaint copy to company, company response (or non-response). (5) Time limit: Within 2 years of cause of action. (6) No lawyer needed — can argue in person. (7) Hearing: Usually monthly. Decision: 3-5 months. (8) Relief: Refund, replacement, compensation for harassment + litigation cost. (9) Covers: products, services, e-commerce, insurance, banking, telecom.`
    },
    {
        id: "dc-consumer-2", tags: ["consumer complaint", "online shopping", "ecommerce", "amazon", "flipkart", "refund"],
        text: `Consumer Complaint for Online Shopping: (1) First: Complain to seller/platform (Amazon, Flipkart). (2) If no response in 30 days or unsatisfactory: File on consumerhelpline.gov.in or call 1800-11-4000. (3) If still unresolved: File complaint on edaakhil.nic.in (District Consumer Forum). (4) Attach: Order screenshot, payment proof, complaint emails, seller response. (5) Can claim: Refund + interest, compensation for mental agony, litigation costs. (6) E-commerce companies are liable even if seller is third-party (Consumer Protection E-Commerce Rules 2020).`
    },

    // ── Family Court ──
    {
        id: "dc-divorce-1", tags: ["divorce", "mutual consent", "contested", "family court", "maintenance"],
        text: `Divorce in Family/District Court: (1) Mutual Consent (Hindu Marriage Act Sec 13B): Both agree. File joint petition. 6-month cooling period (can be waived). (2) Contested Divorce: File on grounds — cruelty, desertion (2 years), adultery, mental disorder, conversion. (3) Maintenance (BNSS 144, formerly CrPC 125): Wife/children/parents can claim. Amount: 25-30% of husband's income typically. (4) Court fee: ₹500-2000. (5) Where to file: Family Court where wife resides or where marriage was performed. (6) Duration: Mutual — 6-18 months. Contested — 2-5 years. NALSA 15100 for free lawyer.`
    },
    {
        id: "dc-custody-1", tags: ["child custody", "guardianship", "visitation", "minor", "family court"],
        text: `Child Custody in Family Court: (1) Welfare of child is paramount — not parent's rights. (2) Generally: Mother preferred for children below 5 years. (3) Father gets custody if: mother unfit, child's wish (if above 9), better environment. (4) Shared custody: Courts increasingly recognizing. (5) Visitation rights: Non-custodial parent usually gets weekends, holidays. (6) File: Guardian & Wards Act petition in Family/District Court. (7) International custody: Hague Convention cases handled by HC. (8) Modification: Either parent can seek modification if circumstances change.`
    },
    {
        id: "dc-dv-1", tags: ["domestic violence", "dv act", "protection order", "wife", "safety", "498a"],
        text: `Domestic Violence — Filing at District Level: (1) Protection of Women from DV Act 2005: File complaint with Protection Officer or Magistrate. (2) Who can file: Wife, live-in partner, mother, sister, any female household member. (3) Relief: Protection order (no contact), residence order (cannot evict), monetary relief (maintenance), custody order. (4) Emergency: Call 181 Women Helpline or visit One Stop Centre (OSC). (5) FIR under BNS 85 (498A IPC) at police station for cruelty by husband. (6) Magistrate must hear within 3 days + decide within 60 days. (7) Free lawyer: DLSA provides. No court fee.`
    },

    // ── Labour Court ──
    {
        id: "dc-labour-1", tags: ["salary", "wages", "labour court", "unpaid salary", "termination", "pf", "esi"],
        text: `Labour Court — District Level Filing: (1) Unpaid salary: Written complaint to Labour Commissioner (free, no lawyer needed). (2) Wrongful termination: File before Industrial Tribunal/Labour Court. If >240 days employment: 15 days salary/year compensation. (3) PF complaint: epfindia.gov.in or toll-free 1800-118-005. (4) ESI complaint: esic.nic.in. (5) Gratuity: File application before Controlling Authority (Labour Dept). Due within 30 days of retirement/resignation. (6) Minimum wage violation: Complaint to Labour Inspector, area office. (7) Documents: Salary slips, appointment letter, bank statements, attendance records.`
    },

    // ── MACT (Motor Accident Claims Tribunal) ──
    {
        id: "dc-mact-1", tags: ["accident", "mact", "motor vehicle", "compensation", "insurance claim", "hit and run"],
        text: `MACT (Motor Accident Claims Tribunal) — District Court: (1) File claim within 6 months of accident (extendable). (2) Who can file: Victim or legal heirs if death. (3) Documents: FIR copy, MLC (medical report), hospital bills, salary proof, vehicle insurance copy. (4) Compensation formula: Multiplier method (age + income × multiplier). (5) No court fee. (6) Hit & Run: Claim from government fund (₹50,000 injury, ₹2 lakh death — Section 161 MV Act). (7) Insurance company is party — they pay. (8) Appeal: To HC within 90 days. (9) Free lawyer from DLSA for accident victims.`
    },

    // ── Property/Land (District Level) ──
    {
        id: "dc-property-1", tags: ["property", "suit", "civil suit", "title", "possession", "declaration"],
        text: `Property Civil Suit at District Court: (1) Title suit: Establish ownership. File under CPC Order VII. (2) Possession suit: Recover possession from illegal occupant. (3) Injunction: Prevent encroachment or illegal construction. (4) Partition suit: Divide joint/ancestral property among co-owners. (5) Court fee: Ad valorem — percentage of property value. (6) Documents: Sale deed, khata/patta, tax receipts, encumbrance certificate, mutation records. (7) Duration: 3-10 years typically. (8) Limitation: 12 years for possession, 3 years for declaration. (9) Mediation available — faster resolution.`
    },
    {
        id: "dc-mutation-1", tags: ["mutation", "daakhil kharij", "khata", "patta", "land record", "patwari", "tehsildar"],
        text: `Land Mutation/Dakhil Kharij — Tehsil Level: (1) After property purchase/inheritance: Apply for mutation at Tehsil office. (2) Documents: Sale deed/succession certificate, previous owner's records, ID proof, application form. (3) Fee: ₹50-500 (varies by state). (4) Process: Patwari verifies → Tehsildar approves → Name changed in revenue records. (5) If rejected: Appeal to Revenue Court (SDM → ADM → Board of Revenue). (6) Online portals: Bihar — biharbhumi.bihar.gov.in, UP — upbhulekh.gov.in, MP — mpbhulekh.gov.in. (7) Mutation is NOT proof of ownership — it's just revenue record update. Title comes from registered sale deed.`
    },

    // ── Rent Disputes ──
    {
        id: "dc-rent-1", tags: ["rent", "landlord", "tenant", "eviction", "rent control", "deposit"],
        text: `Rent Disputes at District Level: (1) Eviction by landlord: Must go through Rent Controller/civil court. No self-help eviction. (2) Grounds for eviction: Non-payment of rent, subletting without consent, damage to property, personal need (bonafide). (3) Deposit refund: Landlord must return within 30-45 days. If not: file in civil court or consumer forum. (4) Rent agreement: Register if >11 months. E-registration available in most states. (5) Rent increase: Limited by state Rent Control Acts (usually 10% max). (6) Tenant improvements: Cannot evict if tenant made permanent improvements with consent.`
    },

    // ── Criminal Complaint before Magistrate ──
    {
        id: "dc-magistrate-1", tags: ["private complaint", "magistrate", "164 statement", "bnss 175", "complaint case"],
        text: `Private Complaint before Magistrate (BNSS Sec 175, formerly CrPC 200): When police refuse to file FIR or investigate. (1) Draft written complaint with facts, evidence, and sections of law violated. (2) File before Metropolitan/Judicial Magistrate. (3) Magistrate examines complainant on oath. (4) If prima facie case: Orders investigation or issues process (summons/warrant). (5) No lawyer needed — can file in person. (6) Court fee: ₹10-100. (7) Time: 15-30 days for initial order. (8) Useful for: cheating cases, fraud, defamation, criminal breach of trust, neighbour disputes. (9) Also: Section 164 BNSS — record statement before Magistrate as evidence.`
    },

    // ── Succession Certificate ──
    {
        id: "dc-succession-1", tags: ["succession", "will", "death", "inheritance", "probate", "nominee", "legal heir"],
        text: `Succession Certificate at District Court: Required to claim bank deposits, shares, insurance of deceased. (1) If WILL exists: Apply for Probate in District Court. (2) If NO will: Apply for Succession Certificate under Indian Succession Act Sec 372. (3) Documents: Death certificate, family tree, relationship proof, list of assets, no-objection from other legal heirs. (4) Court fee: 2-3% of asset value (varies by state). (5) Process: Court issues public notice → 45 days for objections → Certificate issued. (6) Duration: 3-6 months. (7) Bank/insurance may accept: Affidavit + indemnity bond for amounts <₹1 lakh (RBI guideline). (8) Nominee ≠ Owner — legal heirs have superior claim.`
    },

    // ── RTI at District Level ──
    {
        id: "dc-rti-1", tags: ["rti", "right to information", "pio", "first appeal", "information commission"],
        text: `RTI Filing at District Level: (1) Application to PIO (Public Information Officer) of concerned department. (2) Fee: ₹10 (postal order/cash/online). BPL applicants exempt. (3) Response due within 30 days (48 hours if life/liberty issue). (4) If no response/unsatisfactory: First Appeal to officer senior to PIO within 30 days. (5) If first appeal fails: Second Appeal to State/Central Information Commission within 90 days. (6) Online: rtionline.gov.in (central), state portals vary. (7) Can ask for: government records, inspection of documents, certified copies. (8) PIO must give reasons for refusal. Penalty: ₹250/day up to ₹25,000 on PIO for delay.`
    },

    // ── Court Fee & Limitation ──
    {
        id: "dc-limitation-1", tags: ["limitation", "time limit", "court fee", "filing deadline", "statute of limitations"],
        text: `Limitation Periods (Limitation Act 1963) — Key Deadlines: (1) Recovery of money/debt: 3 years. (2) Recovery of possession of property: 12 years. (3) Suit for damages: 1 year (accident), 3 years (breach of contract). (4) Appeal from District Court: 30 days (criminal), 90 days (civil). (5) Consumer complaint: 2 years. (6) Cheque bounce (NI Act 138): 30 days from legal notice expiry. (7) Labour complaint: 3 years. (8) Family court matters: No strict limitation. (9) Condonation of delay: Court may condone if 'sufficient cause' shown. (10) IMPORTANT: Missing limitation period = case barred forever. Consult lawyer early.`
    },

    // ── Mediation & ADR ──
    {
        id: "dc-mediation-1", tags: ["mediation", "conciliation", "settlement", "court annexed", "arbitration"],
        text: `Mediation & Alternative Dispute Resolution (District Level): (1) Court-Annexed Mediation: Judge refers case to mediation centre. Free. Settlement = decree. (2) Private Mediation: Under Mediation Act 2023. Choose mediator. (3) Arbitration: For commercial disputes. Arbitration Act 1996. Award enforceable as decree. (4) Lok Adalat: Free, no appeal. For pending cases + pre-litigation. (5) Conciliation: Labour disputes before Conciliation Officer. (6) Consumer Mediation: Consumer Commission can refer to mediation. (7) Benefits: Faster (weeks vs years), cheaper, confidential, preserves relationships. (8) Mediation success rate: ~65% in court-annexed centres.`
    },

    // ── Cheque Bounce ──
    {
        id: "dc-cheque-1", tags: ["cheque bounce", "138", "negotiable instruments", "legal notice", "dishonour"],
        text: `Cheque Bounce — NI Act 138 at District Court: (1) Cheque must be presented within 3 months of date. (2) If bounced: Send legal notice (registered post/speed post) within 30 days of bank memo. (3) If no payment within 15 days of notice: File criminal complaint in Magistrate Court within 30 days. (4) Punishment: Up to 2 years imprisonment + fine up to 2× cheque amount. (5) Also: File civil suit for money recovery. (6) STRICT deadlines — missing any = case barred. (7) E-filing available: ecourts.gov.in. (8) Evidence needed: Bounced cheque, bank memo, legal notice, postal receipt, reply (if any). Engage lawyer.`
    },

    // ── Cyber Crime at District Level ──
    {
        id: "dc-cyber-1", tags: ["cyber crime", "online fraud", "upi fraud", "otp scam", "digital arrest", "1930"],
        text: `Cyber Crime — Report at District Level: (1) Call 1930 National Cyber Crime Helpline immediately. (2) File complaint online: cybercrime.gov.in. (3) Also file FIR at nearest police station — mention IT Act sections. (4) For UPI/bank fraud: Call bank within 30 minutes to freeze account. RBI mandate: Bank must reverse unauthorized transactions reported within 3 working days. (5) Preserve: Screenshots of messages, transaction IDs, bank statements, phone records. (6) Digital Arrest scam: NO government agency arrests via video call — hang up immediately. (7) State cyber cells have dedicated investigation teams. (8) Compensation: File civil suit or approach adjudicating officer under IT Act.`
    },

    // ── Senior Citizen Maintenance ──
    {
        id: "dc-senior-1", tags: ["senior citizen", "elderly", "parents", "maintenance", "tribunal"],
        text: `Senior Citizen Maintenance — District Level: Maintenance & Welfare of Parents and Senior Citizens Act 2007. (1) Senior citizen (60+) can claim maintenance from children/heirs. (2) Tribunal: SDM or designated officer at district level. (3) Application: Simple form, no court fee, no lawyer needed. (4) Maximum maintenance: ₹10,000/month (may vary by state amendment). (5) Decision within 90 days. (6) Non-compliance: Up to 1 month imprisonment. (7) Also: Can revoke property transfer made under condition of maintenance. (8) Old age homes: State government must establish in every district. (9) Helpline: 14567 (Elder Line — national).`
    },

    // ── Insurance Claim ──
    {
        id: "dc-insurance-1", tags: ["insurance", "claim rejection", "life insurance", "health insurance", "irdai", "ombudsman"],
        text: `Insurance Claim Rejection — District Level Remedies: (1) First: File complaint with company's Grievance Officer. (2) If no response in 15 days: IRDAI IGMS portal (igms.irda.gov.in). (3) Insurance Ombudsman: Free, for claims up to ₹30 lakh (life) / ₹20 lakh (general). Decision within 90 days. 17 centres across India. (4) Consumer Forum: If Ombudsman doesn't help, file on edaakhil.nic.in (District Forum for claims up to ₹1 crore). (5) Documents: Policy copy, claim form, rejection letter, medical records/death certificate. (6) Key: Insurance company must give reason for rejection. Arbitrary rejection = deficiency in service.`
    },

    // ── Case Studies — DC ──
    {
        id: "dc-case-cheque-bounce", tags: ["cheque bounce", "case study", "outcome", "138", "ni act", "negotiable instrument"],
        text: `Case Study — Cheque Bounce (NI Act 138): Ram lent ₹5 lakh to Shyam who gave a post-dated cheque. Cheque bounced. Ram's steps: (1) Sent legal notice within 30 days of bounce memo, (2) Waited 15 days for payment, (3) Filed complaint under NI Act Section 138 in Magistrate court within 30 days after notice period. Outcome: Magistrate convicted Shyam — ordered ₹10 lakh compensation (2x cheque amount) + 9% annual interest. Timeline: 8 months to judgment. Cost: Court fee ₹200, advocate ₹15,000. Key: Missing the 30-day notice deadline bars the case permanently. Always send notice by registered post with AD.`
    },
    {
        id: "dc-case-consumer-refund", tags: ["consumer", "case study", "outcome", "refund", "deficiency", "forum", "edaakhil"],
        text: `Case Study — Consumer Forum Refund: Priya bought a ₹45,000 washing machine that stopped working after 2 months. Company refused repair under warranty. She filed complaint on edaakhil.nic.in (District Consumer Forum). Documents filed: (1) Purchase bill, (2) Warranty card, (3) Written complaint to company + their response, (4) Photos of defect. Outcome: Forum ordered company to replace machine + pay ₹10,000 compensation for mental agony + ₹5,000 litigation costs. Timeline: 4 months filing to order. Cost: Court fee ₹200 (for claims up to ₹5 lakh). Key: Keep all bills and written complaints — oral complaints have no proof value.`
    },
    {
        id: "dc-case-dv-protection", tags: ["domestic violence", "case study", "outcome", "protection order", "dv act", "maintenance"],
        text: `Case Study — Domestic Violence Protection Order: Sunita faced physical abuse from husband and in-laws. Steps: (1) Called Women Helpline 181, (2) Protection Officer visited home and filed DIR (Domestic Incident Report), (3) Filed application under DV Act 2005 in Magistrate court. Outcome: Court passed protection order within 3 days (ex-parte) — husband prohibited from entering shared household. Residence order granted. Maintenance of ₹15,000/month ordered. Timeline: Emergency order in 3 days, final order in 60 days. Cost: Free (legal aid through DLSA). Key: DV Act covers physical, emotional, verbal, sexual, and economic abuse — even threats count.`
    },
    {
        id: "dc-case-labour-salary", tags: ["labour", "case study", "outcome", "salary", "wages", "labour commissioner"],
        text: `Case Study — Unpaid Salary Recovery: Amit worked at a private company for 6 months, was fired without notice and denied 3 months' salary (₹1.2 lakh). Steps: (1) Sent legal notice to company, (2) Filed complaint with Labour Commissioner (free, no lawyer needed), (3) Labour Inspector summoned company. Outcome: Labour Commissioner ordered company to pay ₹1.2 lakh salary + ₹36,000 notice period compensation + 12% interest. Company paid within 30 days to avoid criminal prosecution. Timeline: 45 days. Cost: Zero (Labour Commissioner proceedings are free). Key: File within 1 year of last working day. Keep salary slips, offer letter, and attendance records.`
    },
    {
        id: "dc-case-mact-accident", tags: ["mact", "case study", "outcome", "accident", "compensation", "motor vehicle", "hit and run"],
        text: `Case Study — MACT Accident Compensation: Rajesh, a daily wage worker (₹500/day), was hit by a truck while crossing the road. Suffered leg fracture, hospitalized 3 months. Steps: (1) FIR filed at police station, (2) Filed MACT claim in District Court through DLSA (free legal aid). Compensation calculated: (1) Medical expenses: ₹2.8 lakh (actual bills), (2) Loss of income: ₹500 × 90 days = ₹45,000, (3) Future disability (15%): ₹500 × 365 × 15 years × 15% = ₹4.1 lakh (multiplier method), (4) Pain and suffering: ₹1 lakh. Total: ₹8.35 lakh + 7.5% interest from date of accident. Timeline: 14 months. Cost: Free through DLSA. Key: Even if driver unknown (hit-and-run), claim against Motor Accident Claims Tribunal — paid from government fund.`
    },
    {
        id: "dc-case-consumer-ecommerce", tags: ["consumer", "case study", "outcome", "online shopping", "e-commerce", "refund", "defective product"],
        text: `Case Study — Consumer Complaint for Online Shopping Fraud: Priya ordered a mobile phone worth ₹25,000 from an e-commerce site. Received a fake/refurbished phone. Company refused refund. Steps: (1) Filed complaint on National Consumer Helpline (1800-11-4000), (2) Sent legal notice via email, (3) Filed consumer complaint in District Consumer Commission (fee ₹200). Documents: Order screenshot, payment proof, product photos, delivery receipt. Outcome: Commission ordered company to refund ₹25,000 + ₹10,000 compensation for mental harassment + ₹5,000 litigation cost = ₹40,000 total. Timeline: 3 months. Cost: Filing fee ₹200 only. Key: Consumer complaints up to ₹50 lakh go to District Commission. No lawyer needed — you can argue yourself.`
    },
    {
        id: "dc-case-domestic-violence-dv", tags: ["domestic violence", "dv act", "case study", "outcome", "protection order", "maintenance", "women"],
        text: `Case Study — Domestic Violence Protection Order: Sunita was beaten by her husband and in-laws, thrown out of matrimonial home with her 2 children. Steps: (1) Called Women Helpline 181, (2) Protection Officer filed DIR (Domestic Incident Report), (3) Filed application under DV Act Section 12 in Magistrate Court through DLSA. Magistrate granted: (1) Protection order — husband restrained from violence (Section 18), (2) Residence order — right to live in shared household (Section 19), (3) Maintenance — ₹15,000/month for wife + ₹8,000/month per child (Section 20), (4) Custody of children to mother (Section 21). Timeline: Ex-parte interim order in 3 days, final order in 60 days. Cost: Free through DLSA. Key: DV Act covers physical, emotional, verbal, economic, and sexual abuse. Live-in partners also protected.`
    },
    {
        id: "dc-case-rent-eviction", tags: ["rent", "case study", "outcome", "eviction", "tenant", "landlord", "deposit", "agreement"],
        text: `Case Study — Tenant Eviction & Deposit Recovery: Tenant Raju lived in a rented house for 3 years, paid ₹1 lakh security deposit. Landlord forcibly locked the house and refused to return deposit. Steps: (1) Filed police complaint for house-trespass (BNS 329), (2) Sent legal notice demanding deposit return within 15 days, (3) Filed suit in Rent Controller Court for deposit recovery + damages. Outcome: Rent Controller ordered landlord to return ₹1 lakh deposit + ₹20,000 compensation + 12% interest from eviction date. Also ordered landlord to allow tenant to collect belongings. Timeline: Interim order 2 weeks, final order 4 months. Cost: Court fee ₹500, advocate ₹10,000. Key: Landlord cannot forcibly evict — must follow legal process through Rent Controller. Self-help eviction is illegal.`
    },
    {
        id: "dc-case-cyber-fraud-upi", tags: ["cyber crime", "case study", "outcome", "upi fraud", "online fraud", "digital arrest", "bank freeze"],
        text: `Case Study — UPI/Online Fraud Recovery: Mohan received a call claiming to be from his bank, shared OTP, lost ₹2.5 lakh from his account via UPI. Steps: (1) Immediately called bank helpline — froze account, (2) Filed complaint on cybercrime.gov.in within 24 hours (Golden Hour), (3) Filed FIR at Cyber Crime police station, (4) Bank traced the beneficiary account, froze it. Outcome: ₹1.8 lakh recovered (72%) as beneficiary account was frozen quickly. Remaining ₹70,000 — bank filed insurance claim. Timeline: Recovery within 45 days. Cost: Zero. Key: Report within first hour — 'Golden Hour' for cyber fraud. Call 1930 (National Cyber Crime Helpline) immediately. Never share OTP, PIN, or CVV with anyone — banks never ask for these on call.`
    },
    // ── Rural India & Government Schemes Corpus ──
    {
        id: "dc-land-boundary-dispute", tags: ["land", "boundary", "encroachment", "neighbor", "demarcation", "patwari", "revenue"],
        text: `Land Boundary Dispute — Encroachment by Neighbor: If your neighbor has encroached on your land or moved the boundary, steps: (1) Get certified copy of Khatauni/Khasra from Tehsil office showing your land area and boundaries, (2) File written complaint to Tehsildar/SDM requesting demarcation survey by Lekhpal/Patwari, (3) If Patwari refuses or delays, file complaint to District Collector/DM, (4) If encroachment continues, file civil suit for permanent injunction + possession in District Court under Section 6 of Specific Relief Act. Cost: Tehsil complaint free, Court suit ₹500-2000 fee. Timeline: Demarcation survey 15-30 days, Court injunction 2-6 months. Key: Always get land measured by government surveyor before filing suit — private measurement has no legal value. Keep all revenue records (Khatauni, sale deed, mutation order) safe.`
    },
    {
        id: "dc-land-record-correction", tags: ["land", "record", "khatauni", "khasra", "patwari", "mutation", "correction", "revenue"],
        text: `Land Record Correction — Khatauni/Khasra Errors: If your name is wrong, area is incorrect, or land type is misclassified in revenue records: (1) Apply to Tehsildar with correct documents (sale deed, inheritance certificate, court order), (2) Tehsildar issues notice to all parties, holds hearing, (3) If correction approved, Lekhpal updates records. If Tehsildar refuses: Appeal to SDM within 30 days, then to Commissioner. For mutation after death of owner: Apply with death certificate + legal heir certificate + all heirs' consent. Timeline: Mutation 30-90 days, correction 60-120 days. Cost: Application ₹50-200. Key: Never pay bribe to Patwari — file RTI if records not updated. Online portals: UP (Bhulekh), MP (Bhuabhilekh), Bihar (Bhumi Jankari), Rajasthan (Apna Khata).`
    },
    {
        id: "dc-panchayat-justice", tags: ["panchayat", "gram sabha", "sarpanch", "village", "corruption", "pradhan", "rural"],
        text: `Panchayat Justice — Gram Sabha Disputes & Sarpanch Corruption: Panchayati Raj Act gives villages self-governance. If Sarpanch/Pradhan is corrupt or misusing funds: (1) Raise issue in Gram Sabha meeting (must be held quarterly), (2) File written complaint to Block Development Officer (BDO) with evidence, (3) File complaint to District Panchayat Raj Officer, (4) For financial fraud — file complaint to Lokayukta or Anti-Corruption Bureau. To remove Sarpanch: Need no-confidence motion signed by 2/3rd of Gram Panchayat members, submitted to SDM. For NREGA work disputes: Complaint to Programme Officer, then District Programme Coordinator. Key: Gram Sabha is the most powerful body — attend every meeting, demand reading of accounts. RTI can be filed for all panchayat expenditure records.`
    },
    {
        id: "dc-crop-insurance-pmfby", tags: ["crop", "insurance", "pmfby", "farmer", "kisan", "fasal", "claim", "agriculture"],
        text: `Crop Insurance — PMFBY (Pradhan Mantri Fasal Bima Yojana) Claims: If crops damaged by flood, drought, hailstorm, or pest attack: (1) Inform insurance company or bank within 72 hours of crop damage through Crop Insurance App, CSC center, or bank branch, (2) Take photos/videos of damaged crop with GPS location, (3) Insurance company sends surveyor within 10 days, (4) Claim settled within 45 days of harvest. If claim rejected: (1) Appeal to District Level Monitoring Committee (DLMC), (2) File complaint on PMFBY portal, (3) Escalate to State Level Monitoring Committee. Premium: Kharif 2%, Rabi 1.5%, Horticulture 5% of sum insured. Key: Sowing certificate from Patwari is essential. Non-loanee farmers can also enroll. Deadline: 1 week before sowing season ends.`
    },
    {
        id: "dc-govt-schemes-eligibility", tags: ["scheme", "pm kisan", "nrega", "ayushman", "government", "subsidy", "benefit"],
        text: `Government Schemes for Rural India — Eligibility & Application: (1) PM-KISAN: ₹6,000/year for all farmer families (except tax-payers). Register at pmkisan.gov.in with Aadhaar + land records. Installment not received? Check status online, complain to District Agriculture Officer. (2) NREGA/MGNREGA: 100 days guaranteed employment at minimum wage. Apply at Gram Panchayat. Job card mandatory. If work not given within 15 days — entitled to unemployment allowance. Complaint: Programme Officer → District Coordinator → State helpline. (3) Ayushman Bharat (PM-JAY): Free treatment up to ₹5 lakh/year for BPL families at empaneled hospitals. Check eligibility at mera.pmjay.gov.in or call 14555. Get e-card at CSC center with Aadhaar + ration card. (4) PM Ujjwala: Free LPG connection for BPL women. Apply at nearest LPG distributor.`
    },
    {
        id: "dc-aadhaar-correction", tags: ["aadhaar", "aadhar", "correction", "name", "dob", "address", "linking", "uidai"],
        text: `Aadhaar Correction & Issues: Name/DOB/Address/Gender correction: (1) Online: Login at myaadhaar.uidai.gov.in → Update Demographics → Upload supporting document → Pay ₹50, (2) Offline: Visit nearest Aadhaar Enrollment Center with original documents. Biometric update: Must visit center (fingerprint/iris cannot be updated online). Aadhaar-Bank linking: Visit bank branch with Aadhaar or use bank's app/net banking. Aadhaar-Mobile linking: Visit telecom store with Aadhaar for biometric verification. Lost Aadhaar: Download e-Aadhaar from uidai.gov.in (free). If someone misuses your Aadhaar: (1) Lock biometrics on mAadhaar app, (2) File complaint at uidai.gov.in/complaint, (3) File FIR for identity theft. Key: Aadhaar is proof of identity, NOT proof of citizenship or address.`
    },
    {
        id: "dc-ration-card-pds", tags: ["ration", "card", "pds", "food", "bpl", "apl", "cancelled", "shop"],
        text: `Ration Card — New Card, Corrections & Complaints: New ration card: Apply online on state food department portal or at Tehsil/Block office with family photo, Aadhaar of all members, address proof, income certificate. Types: AAY (poorest, 35kg grain), PHH (priority, 5kg/person), APL. If ration card wrongly cancelled: (1) Apply for restoration at Food & Civil Supplies office, (2) File appeal to District Supply Officer within 30 days. If PDS shop dealer gives less ration or asks for extra money: (1) Complain to Block Supply Officer, (2) Call state food helpline, (3) File complaint on food department portal, (4) Complain to District Collector if no action. Key: Ration is your legal right under National Food Security Act 2013. Dealer cannot refuse grain if your name is on the list.`
    },
    {
        id: "dc-stalking-bns-351a", tags: ["stalking", "harassment", "online", "physical", "complaint", "bns", "women safety"],
        text: `Stalking (BNS Section 78, earlier IPC 354D): Physical stalking (following, watching, contacting repeatedly) and cyber stalking (repeated messages, fake profiles, tracking online activity) are criminal offenses. Punishment: First offense — up to 3 years imprisonment + fine. Repeat offense — up to 5 years + fine. How to file complaint: (1) Save all evidence — screenshots of messages, call logs, photos of stalker near your location, (2) File FIR at nearest police station — stalking is cognizable (police must register FIR), (3) If police refuses FIR — complain to SP/DCP or file complaint to Magistrate under BNSS Section 223, (4) For cyber stalking — also file on cybercrime.gov.in. Can also get protection order from Magistrate. Key: Tell someone you trust immediately. Call Women Helpline 181 or Police 112.`
    },
    {
        id: "dc-digital-arrest-scam", tags: ["digital arrest", "scam", "fake police", "video call", "fraud", "cyber crime"],
        text: `Digital Arrest Scam — Fake Police/CBI/Customs Video Calls: Scammers video-call pretending to be police, CBI, customs, or RBI officers. They claim your Aadhaar/phone is linked to a crime, drug parcel, or money laundering. They demand you stay on video call ('digital arrest') and transfer money to 'verify' your innocence. THIS IS 100% FAKE. No real police or government officer will: (1) Arrest you on video call, (2) Ask for money to 'clear' your name, (3) Threaten you to stay online, (4) Show fake arrest warrants on screen. What to do: (1) Disconnect immediately — this is NOT a real arrest, (2) Call local police (112) to verify, (3) Report on cybercrime.gov.in or call 1930, (4) Block the number. If already paid: File FIR immediately, contact your bank to reverse the transaction. Key: No arrest happens on video call. Real police comes to your door with a written warrant.`
    },
    {
        id: "dc-whatsapp-fraud", tags: ["whatsapp", "fraud", "bank", "impersonation", "otp", "message", "cyber crime"],
        text: `WhatsApp/SMS Fraud — Bank Impersonation & Phishing: Common scams: (1) Fake bank messages asking to update KYC/PAN via link, (2) 'You won lottery' messages, (3) Fake job offers asking for registration fee, (4) Someone pretending to be your relative asking for urgent money. How to protect yourself: Never click links in messages claiming to be from banks — banks never send links via WhatsApp. Never share OTP, PIN, CVV, or UPI PIN with anyone. If you shared OTP/clicked a link: (1) Immediately call your bank's helpline to block card/account, (2) Change all passwords and UPI PIN, (3) File complaint on cybercrime.gov.in within 24 hours (Golden Hour — increases recovery chances), (4) File FIR at police station. Key: Banks, RBI, SBI, or any government body will NEVER call/message asking for OTP or account details. Call 1930 immediately.`
    },
    {
        id: "dc-sim-swap-attack", tags: ["sim swap", "sim", "phone", "stolen", "bank", "mobile", "fraud"],
        text: `SIM Swap Attack — Phone Stolen or SIM Duplicated: If your SIM suddenly stops working or you get 'no network': Scammer may have got a duplicate SIM issued using fake documents to access your bank OTPs. Immediate steps: (1) Call your telecom operator immediately to block the SIM, (2) Call your bank to freeze all accounts linked to that mobile number, (3) File FIR at police station mentioning 'SIM swap fraud', (4) File complaint on cybercrime.gov.in. If phone is stolen: (1) Block phone using IMEI number — dial *#06# on any phone to find IMEI (note it now), (2) Report on ceir.gov.in (Central Equipment Identity Register), (3) File FIR with IMEI number, (4) Inform bank to delink the number. Key: Keep IMEI number saved separately. Enable SIM lock PIN. Do not keep banking apps without app lock.`
    },
    {
        id: "dc-caste-discrimination", tags: ["caste", "discrimination", "sc", "st", "dalit", "atrocity", "school", "workplace"],
        text: `Caste Discrimination — SC/ST (Prevention of Atrocities) Act, 1989: If you face caste-based discrimination, untouchability, abuse, or violence: (1) File FIR under SC/ST Act at any police station — this is a special Act with stronger punishments, (2) Police MUST register FIR — refusal is itself a punishable offense, (3) Investigation must be done by DSP-rank officer, (4) Trial in Special Court with special Public Prosecutor. Covers: Caste-based abuse/slurs, denial of entry to public places, forced manual scavenging, social boycott, denial of water/land/services, assault. Punishment: 6 months to 5 years imprisonment + fine. Victim gets travel allowance for court, maintenance during trial, and compensation from government. If police refuses FIR: Complain to SP/DIG, or file complaint directly to Special Court/Magistrate. Helpline: Call National SC/ST Helpline 14566.`
    },
    {
        id: "dc-reservation-denial", tags: ["reservation", "quota", "college", "admission", "promotion", "obc", "sc", "st", "ews"],
        text: `Reservation Denial — College/Job/Promotion: If your reserved category seat is denied in college admission, government job, or promotion: (1) Get written rejection/merit list from the institution, (2) Verify your caste/category certificate is valid and issued by competent authority (Tehsildar/SDM), (3) File complaint to institution's grievance cell with documents, (4) If no response — file complaint to National Commission for SC (ncsc.nic.in) or National Commission for ST (ncst.nic.in) or National Commission for BC. For EWS: Income certificate from Tehsildar (below ₹8 lakh/year). For college admission disputes: File writ petition in High Court under Article 226 — courts frequently intervene in reservation violations. Key: Reservation is a constitutional right (Articles 15(4), 16(4), 46). Denial is punishable.`
    },
    {
        id: "dc-pension-widow-oldage", tags: ["pension", "widow", "old age", "disability", "vidhwa", "vridha", "application", "appeal"],
        text: `Pension — Widow, Old Age & Disability: (1) Widow Pension (Indira Gandhi National Widow Pension): For BPL widows aged 40-79 — ₹300-500/month (varies by state). Apply at Block/Tehsil office with: death certificate of husband, age proof, BPL card, Aadhaar, bank passbook. (2) Old Age Pension (IGNOAPS): For BPL persons aged 60+ — ₹200-500/month central + state top-up. Apply at Gram Panchayat/Block office. (3) Disability Pension (IGNDPS): For BPL persons with 80%+ disability — ₹300/month. Need disability certificate from government hospital. If pension stopped or not received: (1) Complain to Block Development Officer (BDO), (2) File RTI asking reason for stoppage, (3) Complain to District Social Welfare Officer, (4) File complaint on state CM helpline/portal. Key: Pension is direct benefit transfer to bank/post office account. Keep Aadhaar linked to account.`
    },
    {
        id: "dc-child-forced-marriage", tags: ["child marriage", "forced marriage", "girl", "minor", "prohibition", "complaint"],
        text: `Forced / Child Marriage — Girl's Rights & Complaint: Legal marriage age: 18 for women, 21 for men. Child marriage is a criminal offense under Prohibition of Child Marriage Act 2006. Punishment: 2 years imprisonment + ₹1 lakh fine for anyone who arranges, performs, or promotes child marriage. Girl's rights: (1) Can refuse marriage at any age — consent is mandatory, (2) Child marriage is voidable at girl's option until 2 years after reaching 18, (3) Girl is entitled to maintenance until remarriage. How to stop a child marriage: (1) Call Childline 1098 (24/7), (2) Inform District Child Marriage Prohibition Officer, (3) Call Women Helpline 181, (4) File FIR at police station — this is a cognizable offense, (5) Any person can file complaint to Magistrate for injunction to stop the marriage. Key: Parents, priest, groom — all are punishable. Girl is never punished. After Prohibition of Child Marriage (Amendment) Act 2024, minimum age being revised.`
    },
];

// ═══════════════════════════════════════════════════════════
//  CORPUS PREPROCESSING (BM25)
// ═══════════════════════════════════════════════════════════

function preprocessCorpus(corpus) {
    const lower = corpus.map(doc => ({
        ...doc,
        textLower: doc.text.toLowerCase(),
        tags: doc.tags || [],
        tagsSet: new Set((doc.tags || []).map(t => t.toLowerCase())),
    }));
    const totalLen = lower.reduce((s, d) => s + d.textLower.split(/\s+/).length, 0);
    const avgLen = totalLen / lower.length;
    return { lower, avgLen };
}

const { lower: SC_LOWER, avgLen: SC_AVG } = preprocessCorpus(SC_CORPUS);
const { lower: HC_LOWER, avgLen: HC_AVG } = preprocessCorpus(HC_CORPUS);
const { lower: DC_LOWER, avgLen: DC_AVG } = preprocessCorpus(DC_CORPUS);

// ═══════════════════════════════════════════════════════════
//  QUERY CLASSIFIER — route to relevant tiers
// ═══════════════════════════════════════════════════════════

const SC_KEYWORDS = new Set([
    "supreme court", "article", "constitution", "fundamental", "pil", "basic structure",
    "landmark", "judgment", "constitution bench", "writ article 32", "bns ipc mapping",
    "maneka gandhi", "kesavananda", "vishaka", "dk basu", "puttaswamy", "navtej",
    "lalita kumari", "arnesh kumar", "transgender", "bommai", "shayara bano",
    "mc mehta", "olga tellis", "bonded labour",
    // Hindi
    "सुप्रीम कोर्ट", "उच्चतम न्यायालय", "अनुच्छेद", "संविधान", "मौलिक अधिकार",
]);

const HC_KEYWORDS = new Set([
    "high court", "writ petition", "article 226", "anticipatory bail", "quashing",
    "section 482", "criminal revision", "appeal", "lok adalat", "dlsa", "cat",
    "state tribunal", "service matter", "transfer petition",
    // Hindi
    "उच्च न्यायालय", "हाई कोर्ट", "रिट", "अग्रिम ज़मानत", "अपील",
]);

const DC_KEYWORDS = new Set([
    "fir", "police station", "thana", "consumer forum", "edaakhil", "district court",
    "family court", "labour court", "mact", "accident claim", "rent", "landlord",
    "mutation", "tehsil", "magistrate", "cheque bounce", "succession", "rti",
    "divorce", "maintenance", "custody", "domestic violence", "cyber crime", "1930",
    "senior citizen", "insurance claim", "mediation",
    "land", "boundary", "panchayat", "ration", "kisan", "farmer", "aadhar", "aadhaar",
    "scheme", "pension", "widow", "stalking", "caste", "reservation", "college", "admission",
    "digital arrest", "whatsapp", "sim swap", "crop insurance", "pmfby", "nrega", "pm kisan",
    "ayushman", "child marriage", "forced marriage", "encroachment", "patwari", "khatauni",
    // Hindi
    "थाना", "एफ आई आर", "पुलिस", "उपभोक्ता", "तलाक", "किराया", "म्यूटेशन",
    "दहेज", "ज़मीन", "पंचायत", "राशन", "किसान", "पेंशन", "जाति", "आरक्षण",
]);

function classifyQuery(query) {
    const lower = query.toLowerCase();
    const tiers = { sc: 0, hc: 0, dc: 0 };

    for (const kw of SC_KEYWORDS) {
        if (lower.includes(kw)) tiers.sc += 2;
    }
    for (const kw of HC_KEYWORDS) {
        if (lower.includes(kw)) tiers.hc += 2;
    }
    for (const kw of DC_KEYWORDS) {
        if (lower.includes(kw)) tiers.dc += 2;
    }

    // Default: search all tiers with equal weight
    // If no keywords matched, return all tiers
    return tiers;
}

// ═══════════════════════════════════════════════════════════
//  BM25 RETRIEVAL (shared logic)
// ═══════════════════════════════════════════════════════════

const K1 = 1.5, B = 0.75;

// Hindi→English keyword map for cross-lingual retrieval
const HINDI_MAP = {
    "kanoon": "law", "kanooni": "legal", "thana": "police", "daroga": "police", "vakeel": "lawyer",
    "kiraya": "rent", "makaan": "house", "ghar": "house", "zameen": "land", "jamin": "land",
    "naukri": "job", "salary": "salary", "karz": "loan", "udhaar": "loan",
    "shaadi": "marriage", "talaq": "divorce", "baccha": "child", "paisa": "money",
    "fareb": "fraud", "dhoka": "fraud", "chori": "theft", "maar": "assault",
    "dhamki": "threat", "gaali": "abuse", "adalat": "court", "nyayalay": "court",
    "madad": "help", "haq": "right", "adhikar": "right", "samasya": "problem",
    "fir": "fir", "bail": "bail", "court": "court", "police": "police", "lawyer": "lawyer",
    "property": "property", "accident": "accident", "fraud": "fraud", "divorce": "divorce",
    "wakeel": "lawyer", "advocate": "lawyer", "rishwat": "bribe", "vivad": "dispute",
    "khoon": "murder", "hatya": "murder", "aurat": "woman", "mahila": "woman",
    "buzurg": "senior", "kisan": "farmer", "mazdoor": "labour", "bima": "insurance",
    "insaaf": "justice", "nyay": "justice", "saza": "punishment",
    "samvidhan": "constitution", "panchayat": "panchayat",
    "dahej": "dowry", "sasural": "in-laws", "gaon": "village",
    "mukhiya": "village head", "gram pradhan": "village head", "sarpanch": "village head",
    "fasal": "crop", "beej": "seed", "khet": "farm", "mandi": "market",
    "waris": "heir", "wasiyat": "will", "virasat": "inheritance", "batwara": "partition",
    "peshi": "court hearing", "tareekh": "court date", "ghotala": "scam",
    "dabang": "intimidation", "goonda": "criminal", "bhrashtachar": "corruption",
    "aadhar": "aadhaar", "ration": "ration", "bijli": "electricity",
    "pension": "pension", "vidhwa": "widow", "budha": "elderly",
    "jati": "caste", "aarakshan": "reservation", "dalit": "dalit",
};

function tokenize(text) {
    return text.toLowerCase().replace(/[^\w\s\u0900-\u097F]/g, " ").split(/\s+/).filter(w => w.length > 1);
}

function expandHindi(tokens) {
    const expanded = new Set(tokens);
    for (const t of tokens) {
        if (HINDI_MAP[t]) expanded.add(HINDI_MAP[t]);
    }
    return [...expanded];
}

function phraseScore(docText, query) {
    const qLower = query.toLowerCase();
    const dLower = docText.toLowerCase();
    const tokens = qLower.replace(/[^\w\s]/g, " ").split(/\s+/).filter(w => w.length > 2);
    let bonus = 0;
    for (let i = 0; i < tokens.length - 1; i++) {
        if (dLower.includes(tokens[i] + " " + tokens[i + 1])) bonus += 2.0;
    }
    for (let i = 0; i < tokens.length - 2; i++) {
        if (dLower.includes(tokens[i] + " " + tokens[i + 1] + " " + tokens[i + 2])) bonus += 4.0;
    }
    return bonus;
}

function retrieveFromCorpus(query, corpus, corpusLower, avgLen, k = 2) {
    const rawTokens = tokenize(query);
    const tokens = expandHindi(rawTokens);

    const scores = corpusLower.map(doc => {
        const words = doc.textLower.split(/\s+/);
        const N = words.length;
        let score = 0;

        // BM25
        for (const t of tokens) {
            const tf = words.filter(w => w === t).length;
            if (tf === 0) continue;
            const idf = Math.log((corpus.length + 1) / (corpus.filter(c => c.text.toLowerCase().split(/\s+/).includes(t)).length + 0.5));
            const bm = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * N / avgLen));
            score += idf * bm;
        }

        // Tag bonus
        for (const t of tokens) {
            if (doc.tagsSet.has(t)) score += 3.0;
        }
        for (const t of tokens) {
            for (const tag of doc.tags) {
                if (tag.includes(t) || t.includes(tag)) score += 1.0;
            }
        }

        // Phrase bonus
        score += phraseScore(doc.text, query);

        return { id: doc.id, text: doc.text, score };
    });

    return scores
        .filter(s => s.score > 1.0)
        .sort((a, b) => b.score - a.score)
        .slice(0, k);
}

// ═══════════════════════════════════════════════════════════
//  MULTI-TIER RETRIEVAL — main entry point
// ═══════════════════════════════════════════════════════════

// Cache for tier retrieval
const tierCache = new Map();
const TIER_CACHE_MAX = 100;

/**
 * Retrieve from all three tiers, merge by score, return top-k.
 * Uses query classification to weight tiers.
 */
function retrieveMultiTier(query, k = 3) {
    const cacheKey = query.toLowerCase().trim();
    if (tierCache.has(cacheKey)) return tierCache.get(cacheKey);

    const tiers = classifyQuery(query);
    const perTier = Math.max(2, Math.ceil(k / 2));

    // Retrieve from each tier
    const sc = tiers.sc > 0 || (tiers.hc === 0 && tiers.dc === 0)
        ? retrieveFromCorpus(query, SC_CORPUS, SC_LOWER, SC_AVG, perTier) : [];
    const hc = tiers.hc > 0 || (tiers.sc === 0 && tiers.dc === 0)
        ? retrieveFromCorpus(query, HC_CORPUS, HC_LOWER, HC_AVG, perTier) : [];
    const dc = tiers.dc > 0 || (tiers.sc === 0 && tiers.hc === 0)
        ? retrieveFromCorpus(query, DC_CORPUS, DC_LOWER, DC_AVG, perTier) : [];

    // If no keywords matched any tier, search all
    const allEmpty = sc.length === 0 && hc.length === 0 && dc.length === 0;
    let combined;
    if (allEmpty) {
        const scAll = retrieveFromCorpus(query, SC_CORPUS, SC_LOWER, SC_AVG, 2);
        const hcAll = retrieveFromCorpus(query, HC_CORPUS, HC_LOWER, HC_AVG, 2);
        const dcAll = retrieveFromCorpus(query, DC_CORPUS, DC_LOWER, DC_AVG, 2);
        combined = [...scAll, ...hcAll, ...dcAll];
    } else {
        combined = [...sc, ...hc, ...dc];
    }

    // Deduplicate by id, sort by score, take top-k
    const seen = new Set();
    const result = combined
        .filter(r => { if (seen.has(r.id)) return false; seen.add(r.id); return true; })
        .sort((a, b) => b.score - a.score)
        .slice(0, k);

    // Anti-hallucination: require score > 1.0 and at least 2 qualifying chunks
    let finalResult;
    if (result.length >= 2) finalResult = result;
    else if (result.length === 1 && result[0].score > 5.0) finalResult = result;
    else finalResult = [];

    // Cache
    if (tierCache.size >= TIER_CACHE_MAX) tierCache.delete(tierCache.keys().next().value);
    tierCache.set(cacheKey, finalResult);

    return finalResult;
}

/**
 * Build grounded context using multi-tier retrieval.
 * Drop-in replacement for rag.js buildContext().
 */
function buildContextMultiTier(query) {
    const hits = retrieveMultiTier(query, 3);
    if (!hits.length) return { contextString: "", chunks: [] };
    const contextString = hits.map((h, i) => `--- Reference ${i + 1} ---\n${h.text}`).join("\n\n");
    const chunks = hits.map(h => h.text);
    return { contextString, chunks };
}

module.exports = {
    SC_CORPUS,
    HC_CORPUS,
    DC_CORPUS,
    retrieveMultiTier,
    buildContextMultiTier,
    classifyQuery,
};
