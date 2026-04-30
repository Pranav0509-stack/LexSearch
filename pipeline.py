"""
BharatLLM Dataset Pipeline
============================
Downloads, cleans, deduplicates, and prepares Indian language datasets
for training BharatLLM.

Datasets supported:
  - IndicCorp v2 (AI4Bharat) — 22 Indian languages, 8.5B tokens
  - Sangraha (Sarvam AI) — curated multilingual Indian web data
  - Samanantar — 11 Indian language pairs, 49.7M sentence pairs
  - IIT Hindi Corpus — academic Hindi text
  - Custom domain datasets (legal, agri, medical)

Usage:
    python dataset_pipeline/pipeline.py \
        --languages hi bn ta mr te \
        --datasets indiccorp sangraha samanantar \
        --output-dir ./data/processed
"""

import os
import re
import json
import hashlib
import unicodedata
from pathlib import Path
from typing import List, Dict, Generator, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from loguru import logger
from tqdm import tqdm


# ─────────────────────────────────────────
# Dataset Registry
# ─────────────────────────────────────────

DATASET_REGISTRY = {
    "indiccorp": {
        "name": "IndicCorp v2",
        "hf_id": "ai4bharat/IndicCorp",
        "languages": ["as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks",
                     "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat",
                     "sd", "ta", "te", "ur"],
        "size_tokens": "8.5B",
        "quality": "high",
        "license": "CC0",
    },
    "sangraha": {
        "name": "Sangraha",
        "hf_id": "ai4bharat/sangraha",
        "languages": ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"],
        "size_tokens": "2.4B",
        "quality": "curated",
        "license": "CC-BY-4.0",
    },
    "samanantar": {
        "name": "Samanantar",
        "hf_id": "ai4bharat/samanantar",
        "languages": ["bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"],
        "size_pairs": "49.7M",
        "quality": "parallel",
        "license": "CC-BY-4.0",
        "note": "Translation pairs (en-XX) — good for multilingual alignment"
    },
    "ai4bharat_nlp": {
        "name": "AI4Bharat NLP Suite",
        "hf_id": "ai4bharat/indic-instruct-data-v0.1",
        "languages": ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa"],
        "size_tokens": "1.2B",
        "quality": "instruction-tuned",
        "license": "CC-BY-4.0",
    },
}

LANG_NAMES = {
    "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "ur": "Urdu", "or": "Odia", "as": "Assamese",
    "ne": "Nepali", "sa": "Sanskrit", "mai": "Maithili",
}


# ─────────────────────────────────────────
# Text Cleaner
# ─────────────────────────────────────────

class IndicTextCleaner:
    """
    Cleans Indian language text for LLM training.
    Handles Unicode normalization, encoding issues, and noise removal.
    """

    # Regex patterns for noise removal
    NOISE_PATTERNS = [
        re.compile(r'http\S+'),                  # URLs
        re.compile(r'www\.\S+'),                 # WWW links
        re.compile(r'\S+@\S+\.\S+'),             # Emails
        re.compile(r'#{2,}'),                    # Multiple hashes
        re.compile(r'\*{3,}'),                   # Multiple asterisks
        re.compile(r'_{3,}'),                    # Multiple underscores
        re.compile(r'\.{4,}'),                   # Multiple dots
        re.compile(r'<[^>]+>'),                  # HTML tags
        re.compile(r'\[.*?\]'),                  # Markdown links
        re.compile(r'^\s*[\-\*\+]\s+', re.M),   # Bullet points at line start
    ]

    # Indic-specific cleaning
    DEVANAGARI_PUNCT = str.maketrans({
        '।': '.', '।': '.', '॥': '.',
        '॰': '.', '‌': '', '‍': '',  # ZWNJ, ZWJ removal
    })

    def clean(self, text: str, lang: str = "hi") -> Optional[str]:
        """
        Full cleaning pipeline for a single text sample.
        Returns None if the text should be discarded.
        """
        if not text or not isinstance(text, str):
            return None

        # 1. Unicode normalization (NFC)
        text = unicodedata.normalize("NFC", text)

        # 2. Fix common encoding issues
        text = self._fix_encoding(text)

        # 3. Remove noise patterns
        for pattern in self.NOISE_PATTERNS:
            text = pattern.sub(' ', text)

        # 4. Language-specific cleaning
        if lang in ("hi", "mr", "ne", "sa", "mai"):
            text = text.translate(self.DEVANAGARI_PUNCT)

        # 5. Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        # 6. Quality checks
        if not self._passes_quality_check(text, lang):
            return None

        return text

    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding artifacts in Indian language text."""
        # Fix Windows-1252 / ISO-8859-1 artifacts
        replacements = {
            'â€™': "'", 'â€œ': '"', 'â€': '"',
            'Ã ': 'à', 'Ã©': 'é',
            '\x00': '', '\ufffd': '',  # Null bytes and replacement chars
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _passes_quality_check(self, text: str, lang: str) -> bool:
        """Basic quality heuristics."""
        words = text.split()

        # Minimum length
        if len(words) < 5:
            return False

        # Maximum length (avoid runaway text)
        if len(words) > 5000:
            return False

        # Check character ratio (avoid text that's mostly symbols/numbers)
        alpha_chars = sum(1 for c in text if c.isalpha())
        total_chars = len(text.replace(' ', ''))
        if total_chars > 0 and alpha_chars / total_chars < 0.5:
            return False

        # Check for suspiciously repetitive text
        unique_words = set(words)
        if len(words) > 10 and len(unique_words) / len(words) < 0.2:
            return False

        return True


# ─────────────────────────────────────────
# Language Detector
# ─────────────────────────────────────────

class IndicLangDetector:
    """
    Rule-based Indic language detector using Unicode script ranges.
    Falls back to fasttext for ambiguous cases.
    """

    SCRIPT_TO_LANG = {
        # Devanagari — check for Hindi/Marathi/Nepali specific chars
        (0x0900, 0x097F): "hi",   # Default Devanagari → Hindi
        (0x0980, 0x09FF): "bn",   # Bengali
        (0x0A00, 0x0A7F): "pa",   # Gurmukhi → Punjabi
        (0x0A80, 0x0AFF): "gu",   # Gujarati
        (0x0B00, 0x0B7F): "or",   # Oriya
        (0x0B80, 0x0BFF): "ta",   # Tamil
        (0x0C00, 0x0C7F): "te",   # Telugu
        (0x0C80, 0x0CFF): "kn",   # Kannada
        (0x0D00, 0x0D7F): "ml",   # Malayalam
        (0x0D80, 0x0DFF): "si",   # Sinhala
        (0x0E00, 0x0E7F): "th",   # Thai (not Indic but common in datasets)
    }

    # Devanagari language disambiguation
    DEVANAGARI_MARKERS = {
        "mr": ["आहे", "आहेत", "नाही", "हे", "ती", "तो", "आणि"],  # Marathi
        "ne": ["छ", "छन्", "हो", "भयो", "गर्नु", "गरेको"],        # Nepali
        "hi": ["है", "हैं", "नहीं", "यह", "वह", "और", "में"],    # Hindi
    }

    def detect(self, text: str, expected_lang: Optional[str] = None) -> Tuple[str, float]:
        """
        Detect language of text.
        Returns (language_code, confidence)
        """
        if not text:
            return "unknown", 0.0

        # Count chars per script
        script_counts = defaultdict(int)
        for ch in text:
            cp = ord(ch)
            for (start, end), lang in self.SCRIPT_TO_LANG.items():
                if start <= cp <= end:
                    script_counts[lang] += 1
                    break

        if not script_counts:
            # Likely Latin/English
            return "en", 0.8

        # Get dominant script language
        dominant_lang = max(script_counts, key=script_counts.get)
        total_indic = sum(script_counts.values())
        confidence = script_counts[dominant_lang] / max(len(text), 1)

        # Disambiguate Devanagari
        if dominant_lang == "hi":
            dominant_lang = self._disambiguate_devanagari(text)

        return dominant_lang, min(confidence, 1.0)

    def _disambiguate_devanagari(self, text: str) -> str:
        """Distinguish Hindi, Marathi, Nepali from Devanagari text."""
        scores = defaultdict(int)
        words = text.split()
        for lang, markers in self.DEVANAGARI_MARKERS.items():
            for marker in markers:
                if marker in words:
                    scores[lang] += 1
        if scores:
            return max(scores, key=scores.get)
        return "hi"  # Default to Hindi


# ─────────────────────────────────────────
# MinHash Deduplicator
# ─────────────────────────────────────────

class MinHashDeduplicator:
    """
    MinHash LSH deduplication for large-scale text corpora.
    Removes near-duplicate documents efficiently.
    """

    def __init__(self, num_perm: int = 128, threshold: float = 0.8, ngram: int = 13):
        """
        Args:
            num_perm: Number of permutations for MinHash
            threshold: Jaccard similarity threshold for near-duplicates
            ngram: N-gram size for shingling
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram = ngram
        self.seen_hashes = set()

    def get_shingles(self, text: str) -> set:
        """Get character n-grams (shingles) from text."""
        text = text.lower().replace(' ', '')
        if len(text) < self.ngram:
            return {text}
        return {text[i:i+self.ngram] for i in range(len(text) - self.ngram + 1)}

    def get_minhash_signature(self, text: str) -> List[int]:
        """
        Compute MinHash signature.
        Uses multiple hash functions to approximate Jaccard similarity.
        """
        shingles = self.get_shingles(text)
        if not shingles:
            return [0] * self.num_perm

        # Simulate MinHash with multiple hash functions
        # In production, use datasketch.MinHash for efficiency
        signature = []
        for seed in range(self.num_perm):
            min_hash = float('inf')
            for shingle in shingles:
                # Hash with different seeds
                h = int(hashlib.md5(f"{seed}:{shingle}".encode()).hexdigest(), 16)
                min_hash = min(min_hash, h)
            signature.append(min_hash)

        return signature

    def is_duplicate(self, text: str) -> bool:
        """Check if text is a near-duplicate of something we've seen."""
        # Fast exact dedup first
        exact_hash = hashlib.sha256(text.encode()).hexdigest()
        if exact_hash in self.seen_hashes:
            return True

        # MinHash approximate dedup
        signature = self.get_minhash_signature(text)
        sig_key = tuple(sorted(signature[:16]))  # Use subset as key

        if sig_key in self.seen_hashes:
            return True

        self.seen_hashes.add(exact_hash)
        self.seen_hashes.add(sig_key)
        return False

    def deduplicate_batch(self, texts: List[str]) -> Tuple[List[str], Dict]:
        """
        Deduplicate a batch of texts.
        Returns (unique_texts, stats)
        """
        unique = []
        stats = {"total": len(texts), "duplicates": 0, "kept": 0}

        for text in texts:
            if not self.is_duplicate(text):
                unique.append(text)
                stats["kept"] += 1
            else:
                stats["duplicates"] += 1

        stats["dedup_rate"] = stats["duplicates"] / max(stats["total"], 1)
        return unique, stats


# ─────────────────────────────────────────
# Quality Filter
# ─────────────────────────────────────────

class QualityFilter:
    """
    Multi-stage quality filtering for Indian language text.
    Combines heuristic rules with optional perplexity filtering.
    """

    def __init__(
        self,
        min_words: int = 20,
        max_words: int = 8000,
        min_avg_word_len: float = 2.0,
        max_avg_word_len: float = 25.0,
        max_digit_ratio: float = 0.3,
        max_punct_ratio: float = 0.3,
        max_repeat_ratio: float = 0.5,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.min_avg_word_len = min_avg_word_len
        self.max_avg_word_len = max_avg_word_len
        self.max_digit_ratio = max_digit_ratio
        self.max_punct_ratio = max_punct_ratio
        self.max_repeat_ratio = max_repeat_ratio

    def filter(self, text: str, lang: str = "hi") -> Tuple[bool, str]:
        """
        Apply quality filters.
        Returns (passed, reason_if_rejected)
        """
        words = text.split()
        n_words = len(words)

        # Word count
        if n_words < self.min_words:
            return False, f"too_short:{n_words}"
        if n_words > self.max_words:
            return False, f"too_long:{n_words}"

        # Average word length
        avg_len = sum(len(w) for w in words) / n_words
        if avg_len < self.min_avg_word_len:
            return False, f"avg_word_too_short:{avg_len:.1f}"
        if avg_len > self.max_avg_word_len:
            return False, f"avg_word_too_long:{avg_len:.1f}"

        # Digit ratio
        digits = sum(1 for c in text if c.isdigit())
        if digits / max(len(text), 1) > self.max_digit_ratio:
            return False, f"too_many_digits:{digits/len(text):.2f}"

        # Punctuation ratio
        punct = sum(1 for c in text if unicodedata.category(c).startswith('P'))
        if punct / max(len(text), 1) > self.max_punct_ratio:
            return False, f"too_much_punct:{punct/len(text):.2f}"

        # Repetition check
        unique_words = set(words)
        repeat_ratio = 1.0 - len(unique_words) / n_words
        if repeat_ratio > self.max_repeat_ratio:
            return False, f"too_repetitive:{repeat_ratio:.2f}"

        return True, "passed"

    def score(self, text: str) -> float:
        """
        Quality score between 0-1 (higher is better).
        Useful for ranking/sorting samples.
        """
        passed, reason = self.filter(text)
        if not passed:
            return 0.0

        words = text.split()
        n_words = len(words)

        scores = []

        # Length score (prefer medium-length texts)
        ideal_words = 200
        length_score = 1.0 - abs(n_words - ideal_words) / max(ideal_words, n_words)
        scores.append(max(0, length_score))

        # Vocabulary richness
        vocab_richness = len(set(words)) / n_words
        scores.append(vocab_richness)

        # Alpha ratio (prefer text-heavy over symbol-heavy)
        alpha = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
        scores.append(alpha)

        return sum(scores) / len(scores)


# ─────────────────────────────────────────
# Dataset Downloader
# ─────────────────────────────────────────

class DatasetDownloader:
    """Downloads Indian language datasets from HuggingFace Hub."""

    def __init__(self, cache_dir: str = "./data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        dataset_id: str,
        languages: List[str],
        max_samples_per_lang: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """
        Download a dataset for specified languages.

        Args:
            dataset_id: Registry key (e.g., "indiccorp")
            languages: List of language codes
            max_samples_per_lang: Limit samples per language (for testing)

        Returns:
            Dict mapping language code → list of text samples
        """
        if dataset_id not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_id}. Available: {list(DATASET_REGISTRY.keys())}")

        dataset_info = DATASET_REGISTRY[dataset_id]
        hf_id = dataset_info["hf_id"]
        supported_langs = dataset_info["languages"]

        logger.info(f"Downloading {dataset_info['name']} ({hf_id})")
        logger.info(f"Requested languages: {languages}")

        # Filter to supported languages
        valid_langs = [l for l in languages if l in supported_langs]
        if not valid_langs:
            logger.warning(f"No requested languages supported by {dataset_id}")
            logger.info(f"Supported: {supported_langs}")
            return {}

        results = {}

        try:
            from datasets import load_dataset

            for lang in valid_langs:
                logger.info(f"  Downloading {LANG_NAMES.get(lang, lang)} ({lang})...")

                try:
                    # Try config-based loading (IndicCorp style)
                    ds = load_dataset(hf_id, lang, split="train", streaming=True)
                except Exception:
                    try:
                        # Try direct loading
                        ds = load_dataset(hf_id, split="train", streaming=True)
                    except Exception as e:
                        logger.error(f"  Failed to load {lang}: {e}")
                        continue

                texts = []
                for i, sample in enumerate(ds):
                    if max_samples_per_lang and i >= max_samples_per_lang:
                        break

                    text = sample.get("text") or sample.get("sentence") or ""
                    if text:
                        texts.append(text)

                results[lang] = texts
                logger.success(f"  {LANG_NAMES.get(lang, lang)}: {len(texts)} samples")

        except ImportError:
            logger.error("'datasets' package not installed. Run: pip install datasets")
            raise

        return results

    def download_from_url(self, url: str, output_path: str):
        """Download a file from a URL."""
        import urllib.request

        logger.info(f"Downloading from {url}")
        urllib.request.urlretrieve(url, output_path)
        logger.success(f"Downloaded to {output_path}")

    def get_download_commands(self) -> str:
        """Get shell commands to download all datasets."""
        cmds = [
            "# Download Indian language datasets",
            "# Run these commands to get the data",
            "",
            "# Install required packages",
            "pip install datasets huggingface_hub",
            "",
            "# Download IndicCorp v2 (largest, most languages)",
            "python -c \"",
            "from datasets import load_dataset",
            "for lang in ['hi', 'bn', 'ta', 'mr', 'te', 'gu', 'kn', 'ml']:",
            "    ds = load_dataset('ai4bharat/IndicCorp', lang, split='train')",
            "    ds.save_to_disk(f'data/raw/indiccorp/{lang}')",
            "\"",
            "",
            "# Download Samanantar (translation pairs)",
            "python -c \"",
            "from datasets import load_dataset",
            "for lang in ['hi', 'bn', 'ta', 'mr', 'te']:",
            "    ds = load_dataset('ai4bharat/samanantar', f'en-{lang}', split='train')",
            "    ds.save_to_disk(f'data/raw/samanantar/{lang}')",
            "\"",
            "",
            "# Alternative: Use HuggingFace Hub CLI",
            "huggingface-cli download ai4bharat/IndicCorp",
            "huggingface-cli download ai4bharat/sangraha",
        ]
        return "\n".join(cmds)


# ─────────────────────────────────────────
# Full Pipeline Orchestrator
# ─────────────────────────────────────────

class DatasetPipeline:
    """
    Orchestrates the full Indian language dataset pipeline.
    """

    def __init__(
        self,
        languages: List[str] = None,
        datasets: List[str] = None,
        output_dir: str = "./data/processed",
        max_samples_per_lang: Optional[int] = None,
    ):
        self.languages = languages or ["hi", "bn", "ta", "mr"]
        self.datasets = datasets or ["indiccorp", "sangraha"]
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples_per_lang

        # Pipeline components
        self.cleaner = IndicTextCleaner()
        self.detector = IndicLangDetector()
        self.deduplicator = MinHashDeduplicator(num_perm=128, threshold=0.8)
        self.quality_filter = QualityFilter()

    def run(self):
        """Execute the full pipeline."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        downloader = DatasetDownloader()

        all_stats = {}

        for dataset_id in self.datasets:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing dataset: {DATASET_REGISTRY[dataset_id]['name']}")
            logger.info(f"{'='*50}")

            # Download
            raw_data = downloader.download(dataset_id, self.languages, self.max_samples)

            # Process each language
            for lang, texts in raw_data.items():
                logger.info(f"\n  Processing {LANG_NAMES.get(lang, lang)} ({lang}): {len(texts)} samples")

                stats = {
                    "raw": len(texts),
                    "after_clean": 0,
                    "after_langdetect": 0,
                    "after_dedup": 0,
                    "after_quality": 0,
                }

                # Stage 1: Clean
                cleaned = []
                for text in texts:
                    c = self.cleaner.clean(text, lang)
                    if c:
                        cleaned.append(c)
                stats["after_clean"] = len(cleaned)

                # Stage 2: Language verification
                verified = []
                for text in cleaned:
                    detected_lang, conf = self.detector.detect(text)
                    if detected_lang == lang or conf < 0.3:  # Keep if confident match
                        verified.append(text)
                stats["after_langdetect"] = len(verified)

                # Stage 3: Deduplication
                unique, dedup_stats = self.deduplicator.deduplicate_batch(verified)
                stats["after_dedup"] = len(unique)
                logger.debug(f"    Dedup: {dedup_stats['dedup_rate']:.1%} duplicates removed")

                # Stage 4: Quality filter
                quality_filtered = []
                for text in unique:
                    passed, reason = self.quality_filter.filter(text, lang)
                    if passed:
                        quality_filtered.append(text)
                stats["after_quality"] = len(quality_filtered)

                # Save
                output_file = self.output_dir / dataset_id / lang / "train.jsonl"
                output_file.parent.mkdir(parents=True, exist_ok=True)

                with open(output_file, "w", encoding="utf-8") as f:
                    for text in quality_filtered:
                        json.dump({"text": text, "lang": lang, "source": dataset_id}, f, ensure_ascii=False)
                        f.write("\n")

                # Log stats
                retention = stats["after_quality"] / max(stats["raw"], 1)
                logger.success(
                    f"  {LANG_NAMES.get(lang, lang)}: "
                    f"{stats['raw']} → {stats['after_quality']} "
                    f"({retention:.1%} retained)"
                )
                all_stats[f"{dataset_id}_{lang}"] = stats

        # Save overall stats
        stats_file = self.output_dir / "pipeline_stats.json"
        with open(stats_file, "w") as f:
            json.dump(all_stats, f, indent=2)

        logger.success(f"\nPipeline complete! Output: {self.output_dir}")
        self._print_summary(all_stats)

    def _print_summary(self, stats: Dict):
        """Print a summary table of pipeline results."""
        total_raw = sum(s["raw"] for s in stats.values())
        total_final = sum(s["after_quality"] for s in stats.values())
        overall_retention = total_final / max(total_raw, 1)

        logger.info(f"\n{'='*50}")
        logger.info(f"PIPELINE SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total raw samples:      {total_raw:,}")
        logger.info(f"Total after processing: {total_final:,}")
        logger.info(f"Overall retention rate: {overall_retention:.1%}")
        logger.info(f"Output dir: {self.output_dir}")


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="BharatLLM Dataset Pipeline")
    parser.add_argument("--languages", nargs="+", default=["hi", "bn", "ta", "mr"],
                        help="Language codes to process")
    parser.add_argument("--datasets", nargs="+", default=["indiccorp"],
                        choices=list(DATASET_REGISTRY.keys()),
                        help="Datasets to download and process")
    parser.add_argument("--output-dir", default="./data/processed")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per language (for testing)")
    parser.add_argument("--show-download-commands", action="store_true",
                        help="Show shell commands to download datasets")
    args = parser.parse_args()

    if args.show_download_commands:
        downloader = DatasetDownloader()
        print(downloader.get_download_commands())
        return

    pipeline = DatasetPipeline(
        languages=args.languages,
        datasets=args.datasets,
        output_dir=args.output_dir,
        max_samples_per_lang=args.max_samples,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
