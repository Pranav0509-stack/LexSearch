/**
 * NyayaSathi RAG — Indian Legal Knowledge Base
 * Vector-free retrieval using TF-IDF + BM25-style scoring
 * Covers IPC→BNS 2023, CrPC→BNSS 2023, evidence→BSA 2023 + 40+ Acts
 */

// ═══════════════════════════════════════════════════════════════════════
//  LEGAL CORPUS — grounded chunks with exact Act + Section references
// ═══════════════════════════════════════════════════════════════════════
const CORPUS = [

    // ───── FIR & POLICE ─────
    {
        id: "fir-1", tags: ["fir", "police", "complaint", "thana", "daroga"],
        text: `FIR (First Information Report) filing: Under BNSS 2023 Section 173 (formerly CrPC 154), any person can give information about a cognizable offence to police in writing or orally. Police MUST register FIR — they cannot refuse. If refused, you can: (1) Send written complaint to SP by registered post. (2) File private complaint under BNSS Section 175 before Magistrate. (3) File Zero FIR at any police station regardless of jurisdiction — it must be transferred to correct station. National helpline: 112. NALSA for legal aid: 15100.`
    },

    {
        id: "fir-2", tags: ["zero fir", "transfer", "jurisdiction"],
        text: `Zero FIR: You can file FIR at ANY police station, regardless of where the crime happened. Station must accept it as 'Zero FIR' and transfer to jurisdictional station within 24 hours. Supreme Court in Lalita Kumari v. UP (2013) held registration of FIR mandatory for cognizable offences. Police cannot ask you to wait for 'enquiry' before registering FIR. If they do, it is a punishable offence under BNSS 2023.`
    },

    {
        id: "fir-3", tags: ["anticipatory bail", "bail", "arrest", "bailable"],
        text: `Bail under BNSS 2023: (1) Bailable offences — bail as of right at police station itself. (2) Non-bailable — apply to Sessions Court or High Court. (3) Anticipatory Bail (BNSS Sec 482, formerly CrPC 438) — apply before arrest if you apprehend arrest. (4) Default bail under BNSS Sec 479 — if charge sheet not filed within 60 days (90 for serious cases), you get bail automatically. Bail application needs a lawyer. NALSA 15100 for free lawyer.`
    },

    // ───── CYBER CRIME & ONLINE FRAUD ─────
    {
        id: "cyber-1", tags: ["cyber", "online fraud", "upi", "internet", "phishing", "otp", "scam", "digital arrest"],
        text: `Cyber Crime / Online Fraud: (1) Report immediately on cybercrime.gov.in or call 1930 (National Cyber Crime Helpline). (2) Also file FIR at nearest police station — mention IT Act 2000 Sections 66 (computer-related offences), 66C (identity theft), 66D (cheating by impersonation), 420 IPC/317 BNS (cheating). (3) For UPI/bank fraud: call bank helpline within 30 minutes, freeze account. RBI mandate: bank must reverse unauthorized transactions reported within 3 working days. (4) Digital Arrest scam: No government agency arrests via video call — hang up immediately. Report to 1930.`
    },

    {
        id: "cyber-2", tags: ["cheque bounce", "138", "negotiable", "dishonour"],
        text: `Cheque Bounce — NI Act Section 138: (1) Cheque must be presented within 3 months of date. (2) If bounced, send legal notice (registered post) within 30 days of receiving bank memo. (3) If no payment within 15 days of notice, file complaint in Magistrate Court within 30 days. Punishment: 2 years imprisonment + fine up to 2x cheque amount. (4) You can also file civil suit for recovery. (5) IMPORTANT: These deadlines are strict — missing them bars the case. Consult a lawyer within 30 days of bounce.`
    },

    // ───── PROPERTY & LAND ─────
    {
        id: "prop-1", tags: ["property", "land", "registry", "mutation", "encroachment", "zameen", "jamin", "khet", "title", "deed"],
        text: `Property Disputes: (1) Property title dispute — file civil suit under CPC Order VII. Collect sale deed, khata/patta, tax receipts as evidence. (2) Encroachment — file complaint at Tehsil/Collector office + police complaint (BNS 329 — criminal trespass). (3) Mutation (Daakhil Kharij) — apply at Tehsil/Gram Panchayat after purchase or inheritance. If rejected, appeal to Revenue Court. (4) Property fraud — file FIR under BNS 317 (cheating) + 336 (forgery). Also consumer forum if builder involved.`
    },

    {
        id: "rera-1", tags: ["rera", "builder", "flat", "possession", "delay", "developer", "real estate", "plot"],
        text: `RERA (Real Estate Regulatory Authority Act 2016): (1) Builder must register project with state RERA before selling. (2) If possession delayed beyond promised date — file complaint on state RERA portal. Get interest @SBI PLR+2% for every month of delay. (3) If builder defrauds — complaint to RERA + FIR under BNS 317+318. (4) Homebuyers can demand refund with interest for defects within 5 years. (5) RERA complaint costs ₹1000 and can be filed online. State RERA websites: maharera.mahaonline.gov.in, up-rera.in, haryanarera.gov.in.`
    },

    // ───── RENT & TENANCY ─────
    {
        id: "rent-1", tags: ["rent", "kiraya", "landlord", "tenant", "deposit", "eviction", "lease", "makaan", "ghar"],
        text: `Tenant Rights: (1) Security deposit — landlord must return within 30-45 days of vacating (varies by state). File complaint at Consumer Forum or civil court if withheld. (2) Eviction — landlord cannot forcibly evict. Must give legal notice and obtain court order. (3) Rent increase — most states limit to 10% per year. (4) Rent agreement — register it (even informal ones are valid after 11 months if witnessed). (5) No written agreement — you still have tenant rights. Pay rent by bank transfer for proof. Delhi: Rent Control Act 1958. Mumbai: Maharashtra Rent Control Act 1999. Bangalore: Karnataka Rent Act 2001.`
    },

    // ───── LABOUR & EMPLOYMENT ─────
    {
        id: "labour-1", tags: ["salary", "wages", "naukri", "job", "employer", "fired", "termination", "labour", "pf", "esi", "gratuity"],
        text: `Labour Rights — Code on Wages 2019 + Industrial Disputes Act 1947: (1) Unpaid salary — file complaint with Labour Commissioner (free). (2) Wrongful termination (>240 days employment) — file before Industrial Tribunal; get 15 days salary per year as retrenchment compensation. (3) PF (EPFO) — employer must deposit 12% of basic salary. Complaint: epfindia.gov.in or 1800-118-005 toll free. (4) Gratuity — 15 days per year after 5 years of service. Payment of Gratuity Act 1972. (5) ESIC — medical benefit. Complaint: esic.nic.in. (6) Notice period violation — employer must pay in lieu of notice. Sue in civil court or labour court.`
    },

    {
        id: "labour-2", tags: ["sexual harassment", "workplace", "posh", "committee", "icc"],
        text: `Sexual Harassment at Workplace — POSH Act 2013 (Prevention of Sexual Harassment): (1) Every company with 10+ employees must have Internal Complaints Committee (ICC). (2) File written complaint to ICC within 3 months of incident. (3) No ICC or employer refuses — file complaint to Local Complaints Committee (LCC) at District level. (4) Inquiry must complete in 60 days. (5) If complaint found genuine — employer must take action within 60 days. (6) You can simultaneously file FIR under BNS 74 (sexual harassment). Women helpline: 181. National Commission for Women: 7827170170.`
    },

    // ───── CONSUMER PROTECTION ─────
    {
        id: "consumer-1", tags: ["consumer", "refund", "defective", "product", "warranty", "forum", "commission", "service", "misleading", "amazon", "flipkart", "ecommerce"],
        text: `Consumer Protection Act 2019: (1) District Consumer Commission — claims up to ₹1 crore. File complaint within 2 years. Fee: ₹100-200. (2) State Commission — ₹1-10 crore. (3) National Commission (NCDRC) — above ₹10 crore. (4) Online complaint: consumerhelpline.gov.in or call 1915 (toll free). (5) For e-commerce fraud (Amazon/Flipkart) — file on the platform first, then consumer forum. (6) You can claim: refund + compensation for mental agony + litigation cost. (7) Complaint can be filed by consumer or any consumer association. No lawyer needed.`
    },

    // ───── MOTOR ACCIDENT ─────
    {
        id: "accident-1", tags: ["accident", "insurance", "claim", "motor", "vehicle", "mact", "hit run", "compensation", "challan", "driving", "licence"],
        text: `Motor Accident Claims — Motor Vehicles Act 2019: (1) File FIR immediately (even for minor accidents). (2) Third-party insurance is mandatory. Even without FIR, file claim with insurer within 30 days. (3) MACT (Motor Accident Claims Tribunal) — file within 3 years. No court fee. Can file without lawyer. (4) Hit and Run: Solatium Fund gives ₹2 lakh for death, ₹50,000 for injury. Apply to Claims Enquiry Officer. (5) Insurance rejection — appeal to Insurance Ombudsman (irdai.gov.in/ombudsman). (6) Drunk driving — challan + criminal prosecution under BNS 125A.`
    },

    // ───── FAMILY LAW ─────
    {
        id: "divorce-1", tags: ["divorce", "talaq", "maintenance", "alimony", "custody", "child", "marriage", "shaadi", "wife", "husband", "domestic violence", "498a"],
        text: `Divorce & Family Law: (1) Hindu Marriage Act 1955 Sec 13 — divorce on grounds of cruelty, desertion (2 years), adultery, conversion, mental illness, leprosy, venereal disease, renunciation, presumed death. (2) Mutual consent divorce — Hindu Sec 13B, takes 6-18 months. (3) Maintenance: Wife can claim under CrPC Sec 125 / BNSS Sec 144 or DV Act — expedited. Court determines amount. (4) Domestic Violence — DV Act 2005: file complaint before Magistrate, get Protection Order within 3 days. Call 181 (24x7). (5) Dowry harassment — IPC 498A / BNS 85 (cruelty by husband/relatives) — file FIR. (6) Child custody — paramount consideration is child's welfare. Contact family court.`
    },

    {
        id: "divorce-2", tags: ["muslim", "talaq", "nikah", "iddat", "mehr", "triple talaq", "waqf"],
        text: `Muslim Personal Law: (1) Triple Talaq now criminal offence under Muslim Women (Protection of Rights on Marriage) Act 2019 — 3 years imprisonment. (2) Mehr — wife's right, must be paid. (3) Maintenance under CrPC 125 / BNSS 144 applies to ALL women regardless of religion. (4) Muslim Women (Protection on Divorce) Act 1986 — reasonable provision during Iddat. (5) Khula — wife can seek divorce. (6) Divorce deed must be in writing and witnessed.`
    },

    // ───── RTI ─────
    {
        id: "rti-1", tags: ["rti", "right to information", "government", "transparency", "public authority"],
        text: `RTI Act 2005 (Right to Information): (1) Any citizen can ask information from any public authority — central or state government, PSU, court. (2) File RTI application in writing (₹10 fee for central — by IPO/DD). State fees vary ₹10-50. (3) Reply mandatory within 30 days (48 hours for life/liberty). (4) If no reply or unsatisfactory — appeal to First Appellate Authority within 30 days. (5) Second appeal to Central/State Information Commission within 90 days. (6) Online: rtionline.gov.in (Central). No lawyer needed. BPL applicants — no fee.`
    },

    // ───── POCSO & CHILD ─────
    {
        id: "pocso-1", tags: ["pocso", "child", "minor", "abuse", "sexual", "juvenile", "trafficking", "adoption", "guardian"],
        text: `POCSO Act 2012 (Protection of Children from Sexual Offences): (1) Applies to children under 18. All penetrative assault, sexual assault, harassment — covered. (2) MANDATORY reporting — every person knowing of POCSO offence MUST report to Special Juvenile Police Unit or local police. Not reporting is a crime. (3) Victim name/identity cannot be disclosed. (4) Special Courts conduct in-camera proceedings. (5) Childline: 1098 (24x7 free). (6) Child trafficking: BNS 143, ITPA 1956. (7) Juvenile Justice: JJ Act 2015 — contact CWC (Child Welfare Committee).`
    },

    // ───── SC/ST ATROCITIES ─────
    {
        id: "scst-1", tags: ["sc", "st", "scheduled caste", "scheduled tribe", "atrocity", "caste", "discrimination", "dalit", "adivasi", "reservation"],
        text: `SC/ST (Prevention of Atrocities) Act 1989 + Amendment 2015: (1) Non-bailable offences — police MUST register FIR without preliminary enquiry (Supreme Court in Prathvi Raj Chauhan 2020). (2) Designated Special Courts for speedy trial. (3) Mandatory compensation to victims — state government must pay. (4) Public servant who refuses to register FIR can be prosecuted. (5) Anticipatory bail not available to accused except High Court. (6) National SC Commission: 011-23382740. National ST Commission: 011-26182215. NALSA 15100 for free legal aid.`
    },

    // ───── WILLS & SUCCESSION ─────
    {
        id: "succession-1", tags: ["will", "succession", "inheritance", "nominee", "death", "probate", "intestate", "heir", "ancestral", "partition"],
        text: `Succession & Inheritance: (1) Hindu Succession Act 1956 (amended 2005) — daughters have equal rights in ancestral property as sons. (2) Dying without will (Intestate): Class I heirs first — spouse, children, mother. Then Class II. (3) Muslim succession: governed by Muslim Personal Law — daughters get half of sons' share. (4) Will: Write will on plain paper, two witnesses, sign every page. Register at Sub-Registrar office (not mandatory but advisable). (5) Probate: needed for wills in Bombay, Calcutta, Madras. Apply to civil court. (6) Nominee in bank/insurance gets custody, not ownership — legal heirs still have claim.`
    },

    // ───── COMPANIES & BUSINESS ─────
    {
        id: "company-1", tags: ["company", "business", "partner", "firm", "gst", "tax", "mca", "fraud", "embezzlement", "contract", "breach"],
        text: `Business & Commercial Law: (1) Partnership dispute — Partnership Act 1932, file civil suit. (2) Company fraud — Companies Act 2013, file complaint with MCA (mca.gov.in) or SFIO. (3) Contract breach — Indian Contract Act 1872, file civil suit for damages within 3 years. (4) GST fraud — report to GST helpline 1800-103-4786. (5) Cheque bounce — NI Act 138 (commercial). (6) Debt recovery — file before DRT (Debt Recovery Tribunal) for amounts >₹20 lakhs involving banks. NCLT (National Company Law Tribunal) for insolvency.`
    },

    // ───── DEFAMATION & PRIVACY ─────
    {
        id: "defamation-1", tags: ["defamation", "slander", "libel", "reputation", "social media", "fake news", "privacy"],
        text: `Defamation Law: (1) Criminal defamation — BNS Section 356: false statement published damaging reputation. Complaint before Magistrate. (2) Civil defamation — file suit in civil court for damages (no limit). (3) Social media defamation — include in complaint, platform can be asked to remove. (4) IT Act 2000 Sec 66A (unconstitutional) but Sec 67 (obscene material) still applies. (5) Privacy violation — Personal Data Protection framework. File complaint under IT Act. (6) False FIR/malicious prosecution — file counter FIR + civil suit for malicious prosecution.`
    },

    // ───── ENVIRONMENT & NOISE ─────
    {
        id: "environment-1", tags: ["environment", "pollution", "noise", "nuisance", "factory", "NGT", "environment court"],
        text: `Environment & Nuisance: (1) National Green Tribunal (NGT) — file application for environmental violations. No court fee for individuals. (2) Noise pollution — Noise Pollution Rules 2000: residential areas 45dB day, 35dB night. File FIR under BNS 270 (public nuisance) + complaint to Pollution Control Board. (3) Industrial pollution — file before NGT or State Pollution Control Board. (4) Neighbors nuisance — civil suit for nuisance + injunction + damages. (5) NGT helpline: 1800-11-0035.`
    },

    // ───── IMMIGRATION & PASSPORT ─────
    {
        id: "passport-1", tags: ["passport", "visa", "immigration", "nri", "foreign", "pcc", "police clearance"],
        text: `Passport & Visa: (1) Passport application: passportindia.gov.in. Tatkal service for urgent cases. (2) Police Verification delay — file RTI or writ petition in High Court. (3) Passport impounded — get court stay order. (4) PCC (Police Clearance Certificate) — apply at Passport Seva Kendra. (5) OCI card — overseas citizen of India, apply at Indian High Commission. (6) FRRO registration — foreign nationals staying >180 days must register. (7) Visa overstay — approach FRRO immediately for regularization.`
    },

    // ───── BANKING & LOANS ─────
    {
        id: "banking-1", tags: ["bank", "loan", "emi", "npa", "recovery", "sarfaesi", "drt", "ombudsman", "insurance", "rbi"],
        text: `Banking Rights: (1) Loan recovery harassment — RBI guidelines prohibit harassment. File complaint with Banking Ombudsman (bankingombudsman.rbi.org.in) — free, must decide in 30 days. (2) SARFAESI action — bank must give 60-day notice before taking property. (3) Unauthorized transactions — report to bank within 3 days; RBI mandate for reversal. (4) Insurance mis-selling — IRDAI helpline 155255 or Ombudsman. (5) Cheque bounce from bank error — bank liable for damages. (6) Bank account freeze — approach bank first; then Banking Ombudsman; then High Court writ.`
    },

    // ───── DOMESTIC VIOLENCE ─────
    {
        id: "dv-1", tags: ["domestic violence", "498a", "cruelty", "dv act", "protection order", "wife", "safety"],
        text: `Domestic Violence Act 2005 (DV Act) — Protection: (1) File application before Magistrate — Protection Officer (PO) at District Collectorate/SDM office helps file for free. (2) EMERGENCY Protection Order — court can grant within 3 DAYS. (3) Right to reside in shared household — cannot be thrown out even if not owner. (4) Monetary relief, custody orders also available. (5) Simultaneously file FIR under BNS 85 (cruelty) = formerly IPC 498A. (6) Women helpline: 181 (24x7). iCall: 9152987821. (7) One-Stop Centres (Sakhi) at district hospitals — medical + legal + shelter free.`
    },

    // ───── NALSA & FREE LEGAL AID ─────
    {
        id: "nalsa-1", tags: ["nalsa", "dlsa", "legal aid", "free lawyer", "free", "vakeel", "poor", "help"],
        text: `Free Legal Aid — NALSA (National Legal Services Authority): (1) NALSA 15100 — toll free, 24x7. Free lawyer for eligible persons. (2) Eligible: income below ₹3 lakh/year, women, SC/ST, disabled, children, industrial workers, victims of trafficking/natural disaster, persons in custody. (3) District Legal Services Authority (DLSA) — free lawyer at your district court. Apply at DLSA office. (4) Lok Adalat — quick resolution of disputes. No court fee. Award is final (like decree). (5) Tele-Law 1516 — free video consultation with panel lawyers for rural areas.`
    },

    // ───── ARBITRATION & ADR ─────
    {
        id: "arbitration-1", tags: ["arbitration", "mediation", "conciliation", "settlement", "lok adalat", "adr"],
        text: `Alternative Dispute Resolution (ADR): (1) Arbitration Act 1996 — if contract has arbitration clause, dispute goes to arbitrator, not court. Award enforceable like court decree. (2) Mediation — voluntary, neutral mediator. Supreme Court Mediation Centres free. (3) Lok Adalat — free, consensual, award is final. Covers motor accidents, matrimonial (not divorce), labour, public utility services. (4) Conciliation — parties agree to conciliator's recommendation. (5) Online Dispute Resolution (ODR) — faster for digital/e-commerce disputes. SEBI ODR platform.`
    },

    // ───── LAND ACQUISITION ─────
    {
        id: "land-1", tags: ["land acquisition", "compensation", "nhac", "highway", "government", "collector"],
        text: `Land Acquisition — RFCTLARR Act 2013 (Right to Fair Compensation): (1) Government must give 4x market value for rural land, 2x for urban. (2) Social Impact Assessment mandatory for large acquisitions. (3) 80% affected families must consent for private projects. (4) Dispute on compensation — file before Land Acquisition Collector, then Reference Court, then High Court. (5) Temporary possession — enhanced compensation + 12% interest annually. (6) Urgency clause — compensation in advance. (7) Resettlement and rehabilitation mandatory.`
    },

    // ───── WRIT PETITIONS ─────
    {
        id: "writ-1", tags: ["writ", "pil", "high court", "supreme court", "fundamental rights", "article", "constitution"],
        text: `Constitutional Remedies — Writs: (1) Habeas Corpus — illegal detention. File in High Court or Supreme Court. Court must produce person within 24 hours. (2) Mandamus — force government/authority to perform duty. (3) Certiorari/Prohibition — quash illegal orders. (4) Quo Warranto — challenge illegal holding of public office. (5) PIL (Public Interest Litigation) — any citizen can file in High Court/Supreme Court for public interest. (6) Article 32 — direct to Supreme Court. Article 226 — High Court. (7) No filing fee for PIL in many High Courts. No lawyer needed technically, but advisable.`
    },

    // ───── INCOME TAX DISPUTES ─────
    {
        id: "tax-1", tags: ["income tax", "itr", "demand notice", "cit appeal", "tax dispute", "refund", "assessment", "tds", "tax notice"],
        text: `Income Tax Disputes — Income Tax Act 1961: (1) Demand notice under Section 156 — pay within 30 days or file rectification under Section 154. (2) Appeal against assessment order — file before CIT(Appeals) under Section 246A within 30 days of demand notice. Fee: 250 to 10,000 rupees depending on income. (3) If appeal fails — ITAT (Income Tax Appellate Tribunal) under Section 253. No fee for ITAT. (4) ITR non-filing notice under Section 148 — respond within 30 days. Consult CA or tax advocate. (5) TDS refund delayed — file grievance on incometax.gov.in or call 1800-103-0025 (toll free). (6) Income Tax Ombudsman for service complaints.`
    },

    // ───── INTELLECTUAL PROPERTY ─────
    {
        id: "ip-1", tags: ["trademark", "copyright", "patent", "intellectual property", "ip", "brand", "logo", "infringement", "piracy", "design"],
        text: `Intellectual Property Rights: (1) Trademark — register at ipindia.gov.in under Trade Marks Act 1999. Application fee: 4500 rupees for individuals/startups. Registration gives 10-year protection, renewable. Infringement — file civil suit + FIR under BNS Section 318 (cheating by impersonation). (2) Copyright — automatic on creation, registration optional under Copyright Act 1957. File civil suit for infringement; criminal complaint for piracy under Section 63 (imprisonment up to 3 years). (3) Patent — apply to Patent Office under Patents Act 1970. Takes 2-5 years. (4) Cybersquatting — file complaint with INDRP (Indian Domain Name Dispute Resolution Policy). (5) NALSA 15100 for free lawyer.`
    },

    // ───── MEDICAL NEGLIGENCE ─────
    {
        id: "medical-1", tags: ["medical negligence", "doctor", "hospital", "malpractice", "surgery", "treatment", "nmc", "consumer medical"],
        text: `Medical Negligence — Consumer Protection Act 2019 + National Medical Commission: (1) File complaint at District Consumer Commission (claims up to 1 crore) within 2 years. Supreme Court in Indian Medical Assn v VP Shantha 1995 held medical services are covered under Consumer Protection Act. (2) Simultaneously file complaint with National Medical Commission (nmc.org.in) or State Medical Council for doctor's licence action. (3) Criminal negligence causing death — file FIR under BNS Section 106 (death by negligence, up to 5 years imprisonment). (4) Collect all medical records first — hospital must provide records within 72 hours under NMC regulations. (5) NALSA 15100 for free lawyer.`
    },

    // ───── EDUCATION LAW ─────
    {
        id: "education-1", tags: ["education", "school", "college", "fees", "fee dispute", "ragging", "admission", "university", "ugc", "aicte", "senior", "junior"],
        text: `Education Law — Ragging (college/university seniors harassing juniors) and Fee Disputes: (1) Ragging — UGC Anti-Ragging Regulations 2009. College seniors ragging juniors is a criminal offence. File complaint on antiragging.in or call 1800-180-5522 (24x7 toll free). College must act within 24 hours. FIR under BNS Section 115 (voluntarily causing hurt) at nearest police station. Ragging leading to injury — IPC Section 323/325/307. (2) RTE Act 2009 — private schools must reserve 25% EWS seats; state must reimburse. Denial of EWS admission — complaint to DEO (District Education Officer). (3) Fee dispute — state fee regulatory committees for private colleges; writ petition in High Court for unreasonable fees. Capitation fee is illegal — complaint to state education department. (4) Admission irregularity — file complaint with UGC (ugc.ac.in), AICTE, or NMC.`
    },

    // ───── GST LITIGATION ─────
    {
        id: "gst-1", tags: ["gst", "goods services tax", "input tax credit", "itc", "show cause notice", "gst appeal", "refund gst", "fake invoice", "cgst", "scn", "gst notice"],
        text: `GST Disputes — CGST Act 2017: GST notice (show cause notice / SCN) reply and appeals — (1) Show Cause Notice (SCN) under CGST Section 73 (non-fraud) or Section 74 (fraud) — respond within 30 days with a detailed written reply. Adjudicating authority passes order after hearing. Do NOT ignore the notice — file reply even if you disagree. (2) Appeal against GST demand order — file before Appellate Authority under CGST Section 107 within 3 months. Pre-deposit 10% of disputed tax required to file GST appeal. (3) Second appeal — GST Appellate Tribunal (GSTAT) under CGST Section 112. (4) Input Tax Credit (ITC) mismatch — file rectification on GST portal. (5) Fake invoice fraud — report to DGGI helpline 1800-103-4786. Also file FIR under BNS Section 336 (forgery). (6) GST refund delayed beyond 60 days — portal grievance + writ in High Court. Refund carries 6% interest annually.`
    },

    // ───── FOOD SAFETY ─────
    {
        id: "fssai-1", tags: ["food safety", "fssai", "food adulteration", "restaurant", "food poisoning", "food quality", "packaged food"],
        text: `Food Safety — Food Safety and Standards Act 2006 (FSSAI): (1) Food adulteration or poisoning — file complaint with local Food Safety Officer (district level) or FSSAI helpline 1800-11-2100 (toll free). (2) Restaurant/hotel complaint — State Food Safety Commissioner. (3) Packaged food mislabelling — file complaint with FSSAI (fssai.gov.in) or Consumer Forum. (4) Criminal prosecution under FSS Act Section 59: selling unsafe food — 6 months imprisonment + 1 lakh fine; death caused — imprisonment up to life. (5) Online food delivery complaints — file with platform + Food Safety Commissioner. (6) Consumer Forum also has jurisdiction over defective food products under Consumer Protection Act 2019.`
    },

    // ───── NGT DETAILED PROCEDURES ─────
    {
        id: "ngt-1", tags: ["ngt", "national green tribunal", "pollution", "waste", "trees", "mining", "coastal", "wetland", "environmental damage", "ngt complaint"],
        text: `National Green Tribunal (NGT) — NGT Act 2010: (1) Any person can file application before NGT for environmental violations — no court fee for individuals. (2) NGT jurisdiction: air, water, soil pollution; forests, wetlands, biodiversity; Environment Protection Act 1986, Forest Conservation Act 1980, Water Act 1974, Air Act 1981. (3) File at NGT Principal Bench (Delhi) or regional benches (Pune, Bhopal, Kolkata, Chennai). (4) NGT must dispose of cases within 6 months. (5) Interim relief (stay on illegal construction/mining) available immediately. (6) Compensation claims for environmental damage filed directly before NGT. Helpline: 1800-11-0035. (7) Illegal felling of trees — Forest Conservation Act + NGT + FIR under BNS.`
    },

    // ───── SENIOR CITIZEN PROTECTION ─────
    {
        id: "senior-1", tags: ["senior citizen", "elderly", "old age", "parents", "maintenance parents", "abandoned", "elder", "children duty"],
        text: `Senior Citizen Rights — Maintenance and Welfare of Parents and Senior Citizens Act 2007: (1) Children MUST maintain parents who cannot support themselves. If they refuse — file application before Sub-Divisional Magistrate (SDM). Maintenance up to 10,000 rupees per month can be ordered. (2) Speedy remedy — SDM must pass order within 90 days. (3) Property transfer under duress — if senior gifted property but children stop maintenance, transfer can be CANCELLED by SDM. (4) Abandonment of senior — BNS Section 88. FIR can be filed. (5) Elder Line helpline: 14567 (toll free, 24x7). (6) States must provide old age homes in every district. (7) NALSA 15100 for free legal aid.`
    },

    // ───── DISABILITY RIGHTS ─────
    {
        id: "disability-1", tags: ["disability", "disabled", "rpwd", "benchmark disability", "differently abled", "divyang", "reservation disabled", "accessibility"],
        text: `Disability Rights — Rights of Persons with Disabilities Act 2016 (RPWD Act): (1) 21 types of disabilities recognized. Benchmark disability (40% or more impairment) gets government benefits and reservation. (2) 4% reservation in government jobs (1% each: blindness/low vision, deaf/hard of hearing, locomotor, others). (3) 5% reservation in higher education government institutions. (4) Accessibility — all public buildings and transport must be accessible by June 2022. File complaint with State Commissioner for Disabilities. (5) Discrimination complaint — Chief Commissioner of Disabilities: 011-23386054. (6) Disability certificate — apply at District Hospital through CMO office. (7) NALSA 15100 for free legal aid.`
    },

    // ───── TRIBAL / FOREST RIGHTS ─────
    {
        id: "tribal-1", tags: ["tribal", "adivasi", "forest rights", "fra", "forest rights act", "pesa", "gram sabha", "minor forest produce", "traditional forest", "jungle", "zameen", "tribe"],
        text: `Forest Rights Act 2006 (FRA) — Tribal and Adivasi jungle zameen rights: Scheduled Tribes and Other Traditional Forest Dwellers living on jungle zameen (forest land) are protected. (1) Gram Sabha has FINAL authority to recognize individual and community tribal forest rights. Forced eviction or kheda (displacement) from jungle zameen without Gram Sabha consent is illegal. (2) Individual rights — up to 4 hectares of forest/jungle land actually cultivated. File claim before Gram Sabha → Sub-Divisional Level Committee → District Level Committee. (3) Community rights — collective rights over forests, grazing, water bodies, minor forest produce (MFP). (4) Eviction of tribe/adivasi without due FRA process is illegal — file FIR + writ petition in High Court (Supreme Court in Wildlife First v MoEF 2019). (5) PESA Act 1996 — gram sabhas in Scheduled Areas (tribal areas) have self-governance powers over jungle land; any project needs gram sabha consent. (6) State Tribal Commissioner for tribal complaints. NALSA 15100 for free legal aid.`
    },

    // ───── ANTI-CORRUPTION & WHISTLEBLOWER ─────
    {
        id: "anticorruption-1", tags: ["corruption", "bribery", "bribe", "whistleblower", "rti corruption", "lokpal", "vigilance", "cvc", "anticorruption", "acb", "rishwat", "officer", "sarkari"],
        text: `Anti-Corruption — Sarkari officer rishwat (bribery) complaint: Prevention of Corruption Act 1988. (1) If a sarkari (government) officer demands rishwat (bribe) — immediately contact ACB (Anti-Corruption Bureau) for a trap operation. File FIR under Prevention of Corruption Act 1988 Sections 7-12. (2) Central government officer corruption — file complaint with Central Vigilance Commission (CVC: cvc.nic.in), helpline 1800-11-0180. (3) State government officer — State Vigilance Commission or Lokayukta. (4) Lokpal (lokpal.nic.in) — accepts complaints against Group A/B central government officers. (5) Whistle Blowers Protection Act 2014 — identity kept confidential. File complaint with Competent Authority (CVC). (6) RTI Act combined with vigilance complaint is a powerful anti-corruption tool. NALSA 15100 for free legal aid.`
    },

    // ───── MATERNITY & WOMEN'S LABOUR ─────
    {
        id: "maternity-1", tags: ["maternity", "maternity benefit", "pregnancy", "leave", "creche", "nursing", "equal pay", "maternity leave"],
        text: `Maternity and Women's Labour Rights — Maternity Benefit Act 1961 (amended 2017): (1) 26 weeks paid maternity leave for first 2 children; 12 weeks for 3rd child onwards. (2) Employer cannot terminate during pregnancy or maternity leave — illegal dismissal. (3) Creche mandatory in establishments with 50 or more employees. (4) Nursing breaks — 2 breaks per day until child is 15 months old. (5) Work from home option where nature of work permits. (6) Equal pay — Equal Remuneration Act 1976: same pay for same work regardless of gender. File complaint with Labour Commissioner. (7) Violation — file complaint with Labour Commissioner or Inspector under Maternity Benefit Act. Punishment: up to 1 year imprisonment + 5000 rupees fine.`
    },

    // ───── RIGHT TO EDUCATION ─────
    {
        id: "rte-1", tags: ["rte", "right to education", "school admission", "free education", "ews admission", "private school admission", "elementary education"],
        text: `Right to Education — Right of Children to Free and Compulsory Education Act 2009 (RTE Act): (1) Free and compulsory education for children aged 6 to 14 years. Every neighbourhood school must admit. (2) 25% seats reserved in private unaided schools for EWS/disadvantaged children. Apply through state RTE portal. If denied — file complaint with Block Education Officer. (3) No child can be expelled or held back till Class 8 (no detention policy). (4) No capitation fee or screening for admission — illegal under RTE. File complaint with District Education Officer if demanded. (5) Corporal punishment banned under RTE Section 17. File FIR under BNS + complaint to school management + education department. (6) Out-of-school children — report to District Education Officer for mainstreaming.`
    },

    // ───── LAND REVENUE & MUTATION ─────
    {
        id: "revenue-1", tags: ["mutation", "daakhil kharij", "khata", "patta", "khasra", "khatian", "land record", "patwari", "tehsildar", "revenue court", "zameen", "tehsil"],
        text: `Land Revenue Records — Zameen Mutation (Daakhil Kharij) and Tehsil Appeals: (1) Mutation (Daakhil Kharij/Intkal/Khata transfer) — after buying zameen (land), inheritance, or gift, apply at Tehsil/Revenue office with sale deed, previous records, death certificate (for inheritance). (2) If tehsil office refuses mutation without valid reason — appeal to Revenue Court (Assistant Collector/SDM) within 30 days. Tehsildar must give written reason for rejection. (3) Zameen land record correction — file application before Tehsildar with supporting documents. (4) Khatauni/Khasra copies — available online on state land record portals and at Tehsil office. (5) Fraudulent mutation of zameen — file FIR under BNS Section 336 (forgery) + complaint to Revenue Divisional Officer. (6) Revenue court zameen appeal chain: SDO → District Collector → Board of Revenue → High Court.`
    },

    // ───── INSURANCE DISPUTES ─────
    {
        id: "insurance-1", tags: ["insurance", "claim rejection", "life insurance", "health insurance", "irdai", "ombudsman insurance", "policy", "nominee insurance", "insurance complaint"],
        text: `Insurance Disputes — Insurance Act 1938 + IRDAI Regulations: (1) Claim rejected — first file grievance with insurer's Grievance Redressal Officer (GRO) within 30 days of rejection. (2) If not resolved in 30 days — file complaint with Insurance Ombudsman (free, must decide within 3 months). Find nearest Ombudsman at irdai.gov.in. Claims up to 50 lakh covered. (3) IRDAI helpline: 155255 or 1800-4254-732 (toll free). (4) Life insurance claim after death — nominee must submit death certificate + policy document. Company cannot reject without written reasons. (5) Health insurance cashless denial — hospital can still treat; settle and claim reimbursement later. Insurer must give written reasons for denial. (6) Consumer Forum also has jurisdiction over insurance disputes under Consumer Protection Act 2019.`
    },

    // ═══════════════════════════════════════════════════════════════════════
    //  LANDMARK SUPREME COURT JUDGMENTS — Real case law for bulletproof advice
    // ═══════════════════════════════════════════════════════════════════════

    // ───── CRIMINAL LAW LANDMARK CASES ─────
    {
        id: "sc-arrest-1", tags: ["arrest", "498a", "cruelty", "husband", "dowry", "arnesh kumar", "guidelines arrest"],
        text: `Arnesh Kumar v. State of Bihar (2014) — Supreme Court Judgment on Arrest Guidelines: SC held that police should NOT automatically arrest accused in cases under IPC Section 498A (now BNS Section 85 — cruelty by husband). Guidelines: (1) Police must be satisfied that arrest is necessary under BNSS parameters. (2) Magistrate must independently verify necessity before authorizing detention. (3) Every arrest must show reasons recorded — failure is contempt of court. (4) Police officers who arrest without compliance are liable for departmental action. (5) These guidelines apply to all offences punishable up to 7 years. Cite this judgment when accused of 498A/dowry harassment to prevent wrongful arrest.`
    },
    {
        id: "sc-arrest-2", tags: ["arrest", "family", "inform", "rights arrested", "joginder kumar"],
        text: `Joginder Kumar v. State of UP (1994) — Supreme Court on Rights During Arrest: SC held every arrested person has right to: (1) Inform a family member or friend about arrest immediately. (2) Police MUST inform the arrested person of this right. (3) Arrested person must be produced before magistrate within 24 hours (Article 22 of Constitution). (4) Memo of arrest must be prepared — attested by witness and countersigned by arrestee. (5) Arrested person has right to consult and be defended by a legal practitioner of choice. Violation of these rights makes the arrest illegal. File habeas corpus petition under Article 226 in High Court.`
    },
    {
        id: "sc-arrest-3", tags: ["custodial torture", "police brutality", "dk basu", "arrest guidelines", "lockup death"],
        text: `DK Basu v. State of West Bengal (1997) — Supreme Court 11 Guidelines for Arrest: SC laid down mandatory guidelines to prevent custodial violence: (1) Police must wear accurate name tags. (2) Memo of arrest prepared at time of arrest — with one witness from arrestee's family and countersigned by arrestee. (3) Arrestee has right to inform relative/friend immediately. (4) Time, place of arrest, and custody recorded in police diary. (5) Medical examination of arrestee within 48 hours by trained doctor. (6) Copies of arrest memo sent to magistrate. (7) Arrestee may be permitted to meet lawyer during interrogation. (8) Police control room to display information of arrested persons. Non-compliance is punishable as contempt + departmental action. File complaint with Human Rights Commission if violated.`
    },
    {
        id: "sc-rights-1", tags: ["article 21", "right to life", "due process", "fair trial", "maneka gandhi", "passport"],
        text: `Maneka Gandhi v. Union of India (1978) — Supreme Court on Article 21 Right to Life: SC expanded Article 21 to include right to live with dignity. Key holdings: (1) Right to life is not merely animal existence — includes right to live with dignity. (2) Procedure established by law must be fair, just, and reasonable — not arbitrary. (3) Article 21 covers right to livelihood, right to shelter, right to education, right to health. (4) No person can be deprived of life or personal liberty except by procedure established by law. (5) This judgment is the foundation for most fundamental rights cases in India. Cite when government action affects basic dignity — denial of benefits, arbitrary detention, or unfair procedures.`
    },
    {
        id: "sc-sedition-1", tags: ["sedition", "free speech", "kedar nath", "article 19", "bns 150", "government criticism"],
        text: `Kedar Nath Singh v. State of Bihar (1962) — Supreme Court on Sedition: SC upheld sedition law but with strict limitations: (1) Mere criticism of government is NOT sedition. (2) Only speech that incites actual violence or public disorder is sedition. (3) Strong words used to express disapproval of government policies without inciting violence = protected free speech under Article 19. (4) Under new BNS 2023 Section 150, sedition requires proof of acts endangering sovereignty, unity, or integrity of India. (5) Peaceful protests, social media criticism, and political opposition are NOT sedition. If charged with sedition for criticism — cite this judgment + apply for quashing under BNSS.`
    },

    // ───── FAMILY LAW LANDMARK CASES ─────
    {
        id: "sc-talaq-1", tags: ["triple talaq", "muslim divorce", "shayara bano", "instant talaq", "muslim women"],
        text: `Shayara Bano v. Union of India (2017) — Supreme Court Struck Down Triple Talaq: SC declared instant triple talaq (talaq-e-biddat) unconstitutional. Key holdings: (1) Instant triple talaq is arbitrary and violates Article 14 (right to equality). (2) Parliament enacted Muslim Women (Protection of Rights on Marriage) Act 2019 — triple talaq is now a criminal offence punishable with 3 years jail. (3) Muslim women can file FIR if husband pronounces triple talaq — police must register FIR. (4) Wife entitled to maintenance + custody of children. (5) Husband can apply for bail only from Magistrate (not police). If given triple talaq — file FIR, apply for maintenance under Section 125 CrPC/BNSS, seek custody through Family Court.`
    },
    {
        id: "sc-bigamy-1", tags: ["bigamy", "second marriage", "sarla mudgal", "conversion marriage", "ipc 494", "bns 82"],
        text: `Sarla Mudgal v. Union of India (1995) — Supreme Court on Bigamy Through Conversion: SC held that Hindu man cannot convert to Islam to marry second wife without dissolving first marriage. Holdings: (1) Second marriage without valid divorce from first wife is void — even after religious conversion. (2) Such marriage = offence of bigamy under IPC Section 494 (now BNS Section 82). (3) First wife can file FIR for bigamy + seek maintenance. (4) Children from void second marriage are legitimate but first marriage remains valid. (5) Divorce must be obtained through proper legal process under Hindu Marriage Act Section 13 before remarriage. File FIR for bigamy + petition Family Court if husband marries again without divorce.`
    },
    {
        id: "sc-divorce-1", tags: ["divorce", "cruelty", "irretrievable breakdown", "vinita saxena", "mental cruelty"],
        text: `Vinita Saxena v. Pankaj Pandit (2006) — Supreme Court on Cruelty as Ground for Divorce: SC clarified what constitutes cruelty under Hindu Marriage Act Section 13(1)(ia): (1) Mental cruelty = conduct that makes it impossible for spouse to live together. (2) Filing false criminal cases against spouse = cruelty. (3) Persistent refusal of physical relationship without reason = cruelty. (4) Making false allegations of adultery or character assassination = cruelty. (5) Irretrievable breakdown of marriage — SC can grant divorce under Article 142 even if no specific ground proved. File divorce petition citing cruelty with evidence of mental/physical cruelty before Family Court.`
    },

    // ───── CONSTITUTIONAL & FUNDAMENTAL RIGHTS ─────
    {
        id: "sc-basic-1", tags: ["basic structure", "constitution", "kesavananda bharati", "amendment", "parliament power"],
        text: `Kesavananda Bharati v. State of Kerala (1973) — Basic Structure Doctrine: SC held that Parliament's power to amend Constitution under Article 368 is NOT unlimited. Holdings: (1) Parliament cannot destroy or damage the basic structure of Constitution. (2) Basic structure includes: supremacy of Constitution, republican form of government, secular character, separation of powers, federal character, fundamental rights, judicial review. (3) Any constitutional amendment violating basic structure can be struck down by Supreme Court. (4) This protects citizens' fundamental rights from being taken away by constitutional amendment. Most important constitutional law judgment — cite when fundamental rights are threatened by legislation.`
    },
    {
        id: "sc-privacy-1", tags: ["privacy", "right to privacy", "puttaswamy", "aadhaar", "data protection", "surveillance"],
        text: `KS Puttaswamy v. Union of India (2017) — Right to Privacy: SC declared right to privacy as fundamental right under Article 21. Holdings: (1) Privacy is intrinsic to right to life and personal liberty. (2) Covers informational privacy (data protection), bodily privacy, and decisional privacy. (3) Any invasion of privacy must satisfy triple test: (a) legality — backed by law, (b) necessity — legitimate state aim, (c) proportionality — no excessive intrusion. (4) Aadhaar linking cannot be mandatory for bank accounts or phone numbers. (5) Led to Digital Personal Data Protection Act 2023. Cite when government or private entities collect data without consent, mandate biometrics, or conduct surveillance.`
    },
    {
        id: "sc-377-1", tags: ["section 377", "lgbtq", "homosexuality", "navtej johar", "decriminalization"],
        text: `Navtej Singh Johar v. Union of India (2018) — Decriminalizing Homosexuality: SC struck down Section 377 IPC insofar as it criminalized consensual sexual acts between adults. Holdings: (1) Consensual sex between adults in private — regardless of gender — is NOT a criminal offence. (2) Section 377 violated Articles 14 (equality), 15 (non-discrimination), 19 (freedom of expression), and 21 (right to life with dignity). (3) LGBTQ persons have equal rights as citizens. (4) However, Section 377 still applies to non-consensual acts and acts with minors. (5) If harassed for sexual orientation — file complaint under BNS + cite this judgment. Workplace discrimination against LGBTQ = violation of fundamental rights.`
    },
    {
        id: "sc-transgender-1", tags: ["transgender", "third gender", "nalsa", "hijra", "gender identity"],
        text: `NALSA v. Union of India (2014) — Transgender Rights: SC recognized transgender persons as third gender. Holdings: (1) Right to self-identify gender — no requirement of surgery. (2) Transgender persons entitled to reservation as socially/educationally backward class (OBC). (3) Fundamental rights under Articles 14, 15, 16, 19, 21 apply equally. (4) Led to Transgender Persons (Protection of Rights) Act 2019 — prohibits discrimination in employment, education, healthcare. (5) National portal for transgender certificate: transgender.dosje.gov.in. If discriminated — file complaint with District Magistrate under Transgender Act + Human Rights Commission.`
    },
    {
        id: "sc-vishaka-1", tags: ["sexual harassment", "workplace", "vishaka", "posh", "icc", "women safety office"],
        text: `Vishaka v. State of Rajasthan (1997) — Sexual Harassment at Workplace Guidelines: SC laid down guidelines for prevention of sexual harassment at workplace (later codified as POSH Act 2013). Key rules: (1) Every employer with 10+ employees MUST constitute Internal Complaints Committee (ICC). (2) Sexual harassment includes unwelcome physical contact, demand for sexual favours, sexually coloured remarks, showing pornography, any unwelcome conduct of sexual nature. (3) ICC must complete inquiry within 90 days. (4) If employer fails to constitute ICC — fine up to 50,000 rupees. (5) Complaint can be filed within 3 months of incident (extendable by 3 months). File written complaint to ICC. If no ICC exists — file with Local Complaints Committee at District level. SHe-Box portal: shebox.nic.in.`
    },
    {
        id: "sc-bommai-1", tags: ["president rule", "article 356", "governor", "state government", "sr bommai", "floor test"],
        text: `SR Bommai v. Union of India (1994) — Judicial Review of President's Rule: SC held that imposition of President's Rule under Article 356 is subject to judicial review. Holdings: (1) President's satisfaction under Art 356 must be based on objective material — not political considerations. (2) Floor test is the only way to determine majority — Governor cannot decide on own. (3) SC can restore dismissed state government if President's Rule found unconstitutional. (4) Secularism is basic feature — state government promoting communal politics can be dismissed. (5) If your state government is unconstitutionally dismissed — challenge in Supreme Court under Article 32. Political parties can file writ petition.`
    },

    // ───── CONSUMER & COMMERCIAL LANDMARK CASES ─────
    {
        id: "sc-consumer-1", tags: ["housing delay", "builder delay", "deficiency service", "lucknow development", "flat delay", "possession delay"],
        text: `Lucknow Development Authority v. MK Gupta (1994) — Housing Delay = Consumer Rights Violation: SC held that delay in providing possession of housing/flat amounts to deficiency in service under Consumer Protection Act. Holdings: (1) Government authorities and builders are "service providers" under Consumer Act. (2) Delay in handing over possession = deficiency in service — entitled to compensation. (3) Buyer can claim refund with interest + compensation for mental agony. (4) Builder cannot take advantage of one-sided agreement clauses. (5) Consumer forum has jurisdiction even against government housing authorities. File complaint in Consumer Commission — District (up to 1 crore), State (1-10 crore), National (above 10 crore). Limitation: 2 years from cause of action.`
    },
    {
        id: "sc-medical-2", tags: ["medical negligence", "doctor liability", "patient rights", "vp shantha", "hospital consumer"],
        text: `Indian Medical Association v. VP Shantha (1995) — Medical Services Under Consumer Law: SC held that medical professionals and hospitals are covered under Consumer Protection Act. Holdings: (1) Private hospitals providing paid services = service providers. (2) Even government hospitals where patients pay charges are covered. (3) Only free services in government hospitals are excluded. (4) Patient can file consumer complaint for medical negligence without proving criminal intent. (5) Compensation for negligent treatment, wrong diagnosis, unnecessary surgery. File consumer complaint within 2 years. For criminal negligence — FIR under BNS Section 106 (death by negligence). Medical Council complaint for professional misconduct.`
    },

    // ───── LABOUR & WORKERS LANDMARK CASES ─────
    {
        id: "sc-labour-1", tags: ["bonded labour", "minimum wage", "forced labour", "pudr", "child labour", "article 23"],
        text: `PUDR v. Union of India (1982) — Right Against Forced Labour: SC held that paying less than minimum wage = forced labour under Article 23 of Constitution. Holdings: (1) Every worker has right to minimum wage — non-payment violates fundamental rights. (2) Bonded labour in any form is illegal under Bonded Labour System (Abolition) Act 1976. (3) Contractors exploiting migrant workers = forced labour. (4) Government must actively enforce minimum wage laws. (5) Child labour violates Article 24 — no child below 14 can work in hazardous employment. Report bonded labour to District Magistrate — mandatory rescue + rehabilitation. Labour helpline: 14434. Child labour helpline: 1098 (Childline).`
    },

    // ───── PROPERTY & ENVIRONMENT LANDMARK CASES ─────
    {
        id: "sc-environment-1", tags: ["pollution", "absolute liability", "mc mehta", "hazardous industry", "bhopal", "factory pollution"],
        text: `MC Mehta v. Union of India (1986) — Absolute Liability for Hazardous Industries: SC established stricter standard than negligence for hazardous industries. Holdings: (1) Enterprise engaged in hazardous activity has absolute liability for damage — no need to prove negligence. (2) Compensation must be proportional to magnitude and capacity of enterprise. (3) This goes beyond "strict liability" — no exceptions allowed. (4) Pollution victims can claim compensation under NGT Act 2010 + this judgment. (5) Any person affected by industrial pollution — file complaint with National Green Tribunal (NGT) or State Pollution Control Board. NGT filing fee: 1000 rupees. SPCB complaint: free. Environment helpline available through district administration.`
    },
    {
        id: "sc-livelihood-1", tags: ["right to livelihood", "street vendor", "hawker", "eviction", "olga tellis", "pavement dweller"],
        text: `Olga Tellis v. Bombay Municipal Corporation (1985) — Right to Livelihood Under Article 21: SC held that right to life includes right to livelihood. Holdings: (1) Pavement dwellers and street vendors cannot be evicted without following due process. (2) Eviction without notice and opportunity of hearing = violation of Article 21. (3) Government must provide alternative before eviction. (4) Street Vendors (Protection of Livelihood) Act 2014 — every vendor has right to vend in designated area. Certificate of vending must be issued. (5) If municipality evicts without notice — file writ petition in High Court under Article 226. NALSA 15100 for free legal aid.`
    },

    // ═══════════════════════════════════════════════════════════════════════
    //  BNS ↔ IPC CONVERSION TABLE — Old laws to new laws mapping
    // ═══════════════════════════════════════════════════════════════════════
    {
        id: "bns-ipc-map-1", tags: ["bns", "ipc", "conversion", "old law new law", "bharatiya nyaya sanhita", "section mapping", "ipc to bns"],
        text: `BNS (Bharatiya Nyaya Sanhita 2023) replaces IPC (Indian Penal Code 1860). Key section mappings — IPC → BNS: Murder: IPC 302 → BNS 103. Culpable homicide: IPC 304 → BNS 105. Death by negligence: IPC 304A → BNS 106. Dowry death: IPC 304B → BNS 80. Abetment to suicide: IPC 306 → BNS 108. Attempt to murder: IPC 307 → BNS 109. Kidnapping: IPC 363 → BNS 137. Rape: IPC 376 → BNS 65. Outraging modesty: IPC 354 → BNS 74. Voyeurism: IPC 354C → BNS 77. Stalking: IPC 354D → BNS 78. Sexual harassment: IPC 354A → BNS 75. Insult to modesty: IPC 509 → BNS 79. Cruelty by husband: IPC 498A → BNS 85. Bigamy: IPC 494 → BNS 82.`
    },
    {
        id: "bns-ipc-map-2", tags: ["bns", "ipc", "conversion", "theft", "cheating", "forgery", "criminal intimidation"],
        text: `BNS ↔ IPC Property & Financial Crime Mapping: Theft: IPC 379 → BNS 303. House theft: IPC 380 → BNS 305. Robbery: IPC 392 → BNS 309. Dacoity: IPC 395 → BNS 310. Criminal breach of trust: IPC 406 → BNS 316. Cheating: IPC 420 → BNS 318. Forgery: IPC 463 → BNS 336. Counterfeiting: IPC 489A → BNS 345. Hurt: IPC 323 → BNS 115. Grievous hurt: IPC 326 → BNS 117. Wrongful restraint: IPC 341 → BNS 126. Wrongful confinement: IPC 342 → BNS 127. Criminal intimidation: IPC 506 → BNS 351. Defamation: IPC 499 → BNS 356. Trespass: IPC 441 → BNS 329. Mischief: IPC 425 → BNS 324. BNSS 2023 replaces CrPC 1973. BSA 2023 replaces Indian Evidence Act 1872.`
    },

    // ═══════════════════════════════════════════════════════════════════════
    //  CONSTITUTIONAL ARTICLES — Quick reference for citizens
    // ═══════════════════════════════════════════════════════════════════════
    {
        id: "constitution-1", tags: ["fundamental rights", "constitution", "article 14", "article 19", "article 21", "equality", "freedom", "right to life"],
        text: `Constitution of India — Fundamental Rights (Part III): Article 14: Equality before law — no discrimination by state. Article 15: Prohibition of discrimination on grounds of religion, race, caste, sex, place of birth. Article 16: Equal opportunity in public employment. Article 17: Abolition of untouchability — practicing untouchability is criminal offence. Article 19: Six freedoms — (a) speech and expression, (b) assemble peacefully, (c) form associations, (d) move freely, (e) reside anywhere in India, (f) practise any profession. Article 20: Protection against conviction (no ex post facto law, no double jeopardy, no self-incrimination). Article 21: Right to life and personal liberty — includes dignity, livelihood, shelter, education, health, clean environment, privacy. Article 22: Protection against arbitrary arrest and detention.`
    },
    {
        id: "constitution-2", tags: ["writ", "article 32", "article 226", "habeas corpus", "mandamus", "high court", "supreme court", "fundamental rights enforcement"],
        text: `Constitution of India — Writs and Remedies: Article 32: Right to move Supreme Court for enforcement of fundamental rights — itself a fundamental right. Article 226: High Court can issue writs for fundamental rights AND any other purpose. Five types of writs: (1) Habeas Corpus — produce detained person before court (use when illegally arrested). (2) Mandamus — order public authority to perform duty (use when government office refuses to act). (3) Certiorari — quash unlawful order of lower court/tribunal. (4) Prohibition — prevent lower court from exceeding jurisdiction. (5) Quo Warranto — challenge person holding public office without authority. Article 32 = Supreme Court only. Article 226 = High Court (wider scope). Free legal aid available through NALSA for filing writs.`
    },
    {
        id: "constitution-3", tags: ["directive principles", "article 39", "article 41", "article 46", "legal aid", "free lawyer", "education right", "worker rights"],
        text: `Constitution of India — Directive Principles and Legal Aid: Article 38: State shall secure social order promoting welfare. Article 39: Equal pay for equal work for men and women. Article 39A: Free legal aid — state must ensure equal justice and free legal aid for poor. This led to Legal Services Authorities Act 1987 and NALSA. Article 41: Right to work, education, and public assistance in cases of unemployment, old age, sickness. Article 43: Living wage for workers. Article 45: Free and compulsory education for children (now Article 21A — fundamental right). Article 46: Promotion of educational and economic interests of SC, ST, and weaker sections. Article 47: Duty of state to raise nutrition level and standard of living. These are not enforceable in court BUT guide government policy and can support fundamental rights claims.`
    },

];

// ═══════════════════════════════════════════════════════════════════════
//  RETRIEVAL ENGINE — BM25-inspired scoring
// ═══════════════════════════════════════════════════════════════════════
const K1 = 1.5, B = 0.75;
const CORPUS_LOWER = CORPUS.map(c => ({ ...c, textLower: c.text.toLowerCase(), tagsSet: new Set(c.tags) }));
const AVG_LEN = CORPUS.reduce((s, c) => s + c.text.split(" ").length, 0) / CORPUS.length;

function tokenize(str) {
    return str.toLowerCase()
        .replace(/[^\w\s]/g, " ")
        .split(/\s+/)
        .filter(w => w.length > 1);
}

/**
 * Phrase-level bonus: bigram (+2.0) and trigram (+4.0) exact matches
 * Rewards documents that contain multi-word query phrases verbatim.
 */
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

// Hindi→English keyword map for cross-lingual retrieval
const HINDI_MAP = {
    "kanoon": "law", "kanooni": "legal", "thana": "police", "daroga": "police", "vakeel": "lawyer",
    "kiraya": "rent", "makaan": "house", "ghar": "house", "zameen": "land", "jamin": "land",
    "naukri": "job", "salary": "salary", "boss": "employer", "karz": "loan", "udhaar": "loan",
    "shaadi": "marriage", "talaq": "divorce", "baccha": "child", "paisa": "money", "rupay": "money",
    "fareb": "fraud", "dhoka": "fraud", "loot": "theft", "chori": "theft", "maar": "assault",
    "peet": "assault", "dhamki": "threat", "gaali": "abuse", "adalat": "court", "nyayalay": "court",
    "madad": "help", "haq": "right", "adhikar": "right", "samasya": "problem", "pareshani": "problem",
    "naksha": "map", "tehsildar": "revenue", "patwari": "revenue", "sarpanch": "panchayat",
    "gaon": "village", "sheher": "city", "gali": "street",
    // common transliterations
    "fir": "fir", "bail": "bail", "court": "court", "police": "police", "lawyer": "lawyer",
    "property": "property", "accident": "accident", "fraud": "fraud", "divorce": "divorce",
    // v10 — expanded synonyms for better retrieval
    "wakeel": "lawyer", "advocate": "lawyer", "waqeel": "lawyer",
    "rishwat": "bribe", "ghoos": "bribe",
    "vivad": "dispute", "jhagda": "dispute", "ladai": "dispute",
    "khoon": "murder", "hatya": "murder", "qatl": "murder",
    "aurat": "woman", "mahila": "woman", "patni": "wife", "pati": "husband",
    "buzurg": "senior", "budhapa": "senior", "vridh": "senior",
    "kisan": "farmer", "kheti": "agriculture",
    "bijli": "electricity", "batti": "electricity",
    "ilaj": "treatment", "aspatal": "hospital", "dawai": "medicine",
    "pension": "pension", "retire": "retirement",
    "sangathan": "union", "mazdoor": "labour", "kaamgar": "worker",
    "hathiyar": "weapon", "bandook": "gun",
    "nashila": "drug", "nasha": "drug", "sharab": "liquor",
    "parchi": "receipt", "raseed": "receipt",
    "samvidhan": "constitution", "adhikaar": "right",
    "nijta": "privacy", "gairkanuni": "illegal",
    "saza": "punishment", "jail": "prison", "kaid": "prison",
    "gawah": "witness", "saboot": "evidence", "sabut": "evidence",
    "insaaf": "justice", "nyay": "justice",
    "bima": "insurance", "dava": "claim",
    "aadhaar": "aadhaar", "pan": "pan",
    "panchayat": "panchayat", "gram": "village",
    "sarkar": "government", "sarkari": "government",
};

function expandHindi(tokens) {
    const expanded = new Set(tokens);
    for (const t of tokens) {
        if (HINDI_MAP[t]) expanded.add(HINDI_MAP[t]);
    }
    return [...expanded];
}

// RAG Result Cache — corpus is static, so results are deterministic
const ragCache = new Map();
const RAG_CACHE_MAX = 100;

/**
 * Retrieve top-k relevant legal chunks for a query
 * @param {string} query
 * @param {number} k
 * @returns {Array<{id, text, score}>}
 */
function retrieve(query, k = 3) {
    const cacheKey = query.toLowerCase().trim();
    if (ragCache.has(cacheKey)) return ragCache.get(cacheKey);

    const rawTokens = tokenize(query);
    const tokens = expandHindi(rawTokens);

    const scores = CORPUS_LOWER.map(doc => {
        const words = doc.textLower.split(/\s+/);
        const N = words.length;
        let score = 0;

        // BM25 term scoring — FIXED: exact word match (was substring match)
        for (const t of tokens) {
            const tf = words.filter(w => w === t).length;
            if (tf === 0) continue;
            const idf = Math.log((CORPUS.length + 1) / (CORPUS.filter(c => c.text.toLowerCase().split(/\s+/).includes(t)).length + 0.5));
            const bm = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * N / AVG_LEN));
            score += idf * bm;
        }

        // Tag exact match bonus (high weight)
        for (const t of tokens) {
            if (doc.tagsSet.has(t)) score += 3.0;
        }
        // Partial tag match
        for (const t of tokens) {
            for (const tag of doc.tags) {
                if (tag.includes(t) || t.includes(tag)) score += 1.0;
            }
        }

        // Phrase-level bonus for multi-word query matches
        score += phraseScore(doc.text, query);

        return { id: doc.id, text: doc.text, score };
    });

    // Anti-hallucination: require score > 1.0 and at least 2 qualifying chunks.
    // Exception: single chunk with score > 5.0 = very high confidence, return it alone.
    const qualifying = scores
        .filter(s => s.score > 1.0)
        .sort((a, b) => b.score - a.score)
        .slice(0, k);

    let result;
    if (qualifying.length >= 2) result = qualifying;
    else if (qualifying.length === 1 && qualifying[0].score > 5.0) result = qualifying;
    else result = [];

    // Cache the result
    if (ragCache.size >= RAG_CACHE_MAX) ragCache.delete(ragCache.keys().next().value);
    ragCache.set(cacheKey, result);
    return result;
}

/**
 * Build grounded context for LLM injection.
 * Uses multi-tier RAG (SC/HC/DC) first, falls back to original corpus.
 * Returns both the formatted string AND raw chunk texts so citation-guard can verify claims.
 * @returns {{ contextString: string, chunks: string[] }}
 */
function buildContext(query) {
    // Try multi-tier retrieval first (SC/HC/DC corpora)
    let hits;
    try {
        const { retrieveMultiTier } = require("./rag-tiers");
        hits = retrieveMultiTier(query, 3);
    } catch {
        hits = [];
    }

    // Fallback to original corpus if multi-tier returned nothing
    if (!hits || hits.length === 0) {
        hits = retrieve(query, 3);
    }

    // Merge: if multi-tier found something but original corpus also has strong matches, combine
    if (hits.length > 0 && hits.length < 3) {
        const originalHits = retrieve(query, 3);
        const existingIds = new Set(hits.map(h => h.id));
        for (const oh of originalHits) {
            if (!existingIds.has(oh.id) && hits.length < 3) {
                hits.push(oh);
            }
        }
        hits.sort((a, b) => b.score - a.score);
    }

    if (!hits.length) return { contextString: "", chunks: [] };
    const contextString = hits.map((h, i) => `--- Reference ${i + 1} ---\n${h.text}`).join("\n\n");
    const chunks = hits.map(h => h.text);
    return { contextString, chunks };
}

module.exports = { retrieve, buildContext };