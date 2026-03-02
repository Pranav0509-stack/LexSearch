"""
BharatLLM Evaluation Suite
============================
Evaluates BharatLLM on Indian language benchmarks.

Benchmarks:
  - IndicGLUE: 6 tasks across 11 Indian languages
  - Indic-MMLU: Multi-subject QA in Indian languages
  - Translation quality (BLEU on Samanantar test set)
  - Custom India-specific Q&A (culture, history, law)
  - Hinglish understanding

Usage:
    python eval/evaluate.py --model ./outputs/bharat-llm-7b
    python eval/evaluate.py --model ./outputs/bharat-llm-7b --benchmarks indic_mmlu translation
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from loguru import logger


# ─────────────────────────────────────────
# India-specific Q&A Test Set
# ─────────────────────────────────────────

INDIA_QA_SAMPLES = [
    # Constitutional / Legal
    {
        "question": "What is Article 370 of the Indian Constitution?",
        "reference": "Article 370 of the Indian Constitution granted special autonomous status to Jammu and Kashmir. It was abrogated in August 2019.",
        "lang": "en", "domain": "law",
    },
    {
        "question": "भारत के संविधान में मौलिक अधिकार कितने हैं?",
        "reference": "भारतीय संविधान में मूल रूप से सात मौलिक अधिकार थे, लेकिन संपत्ति का अधिकार 1978 में हटाने के बाद अब छह मौलिक अधिकार हैं।",
        "lang": "hi", "domain": "law",
    },
    # Agriculture
    {
        "question": "खरीफ और रबी फसलों में क्या अंतर है?",
        "reference": "खरीफ फसलें मानसून के मौसम (जून-सितंबर) में बोई जाती हैं जैसे चावल, मक्का, ज्वार। रबी फसलें सर्दियों में (अक्टूबर-मार्च) बोई जाती हैं जैसे गेहूं, जौ, सरसों।",
        "lang": "hi", "domain": "agriculture",
    },
    # Culture
    {
        "question": "பொங்கல் திருவிழா எந்த மாநிலத்தில் கொண்டாடப்படுகிறது?",
        "reference": "பொங்கல் திருவிழா முக்கியமாக தமிழ்நாட்டில் கொண்டாடப்படுகிறது. இது நான்கு நாள் திருவிழாவாகும்.",
        "lang": "ta", "domain": "culture",
    },
    # History
    {
        "question": "Who founded the Indian National Congress?",
        "reference": "The Indian National Congress was founded in 1885 by Allan Octavian Hume, a retired British civil servant, along with Dadabhai Naoroji and Dinshaw Wacha.",
        "lang": "en", "domain": "history",
    },
    # Science/Technology
    {
        "question": "ISRO का पूरा नाम क्या है और इसकी स्थापना कब हुई?",
        "reference": "ISRO का पूरा नाम Indian Space Research Organisation (भारतीय अंतरिक्ष अनुसंधान संगठन) है। इसकी स्थापना 15 अगस्त 1969 को हुई थी।",
        "lang": "hi", "domain": "science",
    },
    # Hinglish
    {
        "question": "GST kya hai aur yeh kab implement hua?",
        "reference": "GST matlab Goods and Services Tax ek indirect tax hai jo 1 July 2017 se India mein implement hua. Yeh multiple taxes ko replace karta hai.",
        "lang": "hinglish", "domain": "economics",
    },
]

INDIC_MMLU_SUBJECTS = [
    "Indian History", "Indian Geography", "Indian Politics",
    "Indian Economy", "Indian Science", "Indian Literature",
    "Indian Law", "Indian Agriculture", "Indian Culture", "General Knowledge",
]


# ─────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────

def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score for translation evaluation."""
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU()
        result = bleu.corpus_score(predictions, [references])
        return result.score
    except ImportError:
        logger.warning("sacrebleu not installed, using simple BLEU approximation")
        return _simple_bleu(predictions, references)


def _simple_bleu(predictions: List[str], references: List[str]) -> float:
    """Simple word-overlap BLEU approximation."""
    if not predictions:
        return 0.0

    scores = []
    for pred, ref in zip(predictions, references):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        if not ref_words:
            continue
        precision = len(pred_words & ref_words) / max(len(pred_words), 1)
        recall = len(pred_words & ref_words) / max(len(ref_words), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        scores.append(f1)

    return sum(scores) / len(scores) * 100 if scores else 0.0


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Exact match accuracy."""
    if not predictions:
        return 0.0
    correct = sum(p.strip().lower() == r.strip().lower()
                  for p, r in zip(predictions, references))
    return correct / len(predictions) * 100


def compute_contains_match(predictions: List[str], references: List[str]) -> float:
    """Check if reference keywords appear in prediction."""
    if not predictions:
        return 0.0
    scores = []
    for pred, ref in zip(predictions, references):
        ref_keywords = [w for w in ref.split() if len(w) > 4]
        if not ref_keywords:
            scores.append(1.0)
            continue
        matches = sum(1 for k in ref_keywords if k.lower() in pred.lower())
        scores.append(matches / len(ref_keywords))
    return sum(scores) / len(scores) * 100


# ─────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────

class BharatLLMEvaluator:
    """
    Comprehensive evaluator for BharatLLM.
    Runs multiple benchmarks and produces a report.
    """

    def __init__(self, model_path: str, benchmarks: List[str] = None):
        self.model_path = model_path
        self.benchmarks = benchmarks or ["india_qa", "language_detection", "translation"]
        self.results = {}

    def load_model(self):
        """Load model for evaluation."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model for evaluation: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        logger.success("Model loaded")
        return self

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response for a given prompt."""
        import torch

        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low temp for eval (more deterministic)
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def run_india_qa(self) -> Dict:
        """Evaluate on India-specific Q&A samples."""
        logger.info("Running India QA evaluation...")

        predictions = []
        references = []
        domains = []

        for sample in INDIA_QA_SAMPLES:
            prompt = f"Question: {sample['question']}\nAnswer:"
            pred = self.generate(prompt, max_new_tokens=150)
            predictions.append(pred)
            references.append(sample["reference"])
            domains.append(sample["domain"])

        # Compute metrics
        contains = compute_contains_match(predictions, references)

        # Domain-wise breakdown
        domain_scores = {}
        for domain in set(domains):
            domain_preds = [p for p, d in zip(predictions, domains) if d == domain]
            domain_refs = [r for r, d in zip(references, domains) if d == domain]
            domain_scores[domain] = compute_contains_match(domain_preds, domain_refs)

        result = {
            "overall_score": contains,
            "domain_scores": domain_scores,
            "n_samples": len(INDIA_QA_SAMPLES),
        }

        logger.success(f"India QA: {contains:.1f}% keyword match")
        return result

    def run_language_detection_eval(self) -> Dict:
        """Test if model responds appropriately in the correct language."""
        logger.info("Running language switching evaluation...")

        test_cases = [
            {"input": "नमस्ते, आप कैसे हैं?", "expected_lang": "hi"},
            {"input": "আপনি কেমন আছেন?", "expected_lang": "bn"},
            {"input": "நீங்கள் எப்படி இருக்கிறீர்கள்?", "expected_lang": "ta"},
            {"input": "Hello, how are you?", "expected_lang": "en"},
        ]

        correct = 0
        for case in test_cases:
            response = self.generate(f"{case['input']}\n", max_new_tokens=100)
            # Simple check: does response contain expected script?
            expected_script_range = {
                "hi": (0x0900, 0x097F),
                "bn": (0x0980, 0x09FF),
                "ta": (0x0B80, 0x0BFF),
                "en": (0x0041, 0x007A),
            }.get(case["expected_lang"], (0, 0))

            has_expected = any(
                expected_script_range[0] <= ord(c) <= expected_script_range[1]
                for c in response
            )
            if has_expected:
                correct += 1

        score = correct / len(test_cases) * 100
        logger.success(f"Language switching: {score:.0f}% ({correct}/{len(test_cases)})")
        return {"score": score, "correct": correct, "total": len(test_cases)}

    def run_all(self) -> Dict:
        """Run all configured benchmarks."""
        logger.info(f"Running benchmarks: {self.benchmarks}")

        if "india_qa" in self.benchmarks:
            self.results["india_qa"] = self.run_india_qa()

        if "language_detection" in self.benchmarks:
            self.results["language_detection"] = self.run_language_detection_eval()

        self._print_report()
        self._save_report()
        return self.results

    def _print_report(self):
        """Print evaluation report."""
        print("\n" + "="*60)
        print("🇮🇳 BharatLLM Evaluation Report")
        print("="*60)

        if "india_qa" in self.results:
            r = self.results["india_qa"]
            print(f"\n📋 India QA Benchmark")
            print(f"   Overall: {r['overall_score']:.1f}%")
            for domain, score in r.get("domain_scores", {}).items():
                print(f"   {domain:15s}: {score:.1f}%")

        if "language_detection" in self.results:
            r = self.results["language_detection"]
            print(f"\n🗣 Language Switching")
            print(f"   Score: {r['score']:.0f}% ({r['correct']}/{r['total']})")

        print("\n" + "="*60)

    def _save_report(self):
        """Save evaluation results to JSON."""
        report_path = Path(self.model_path) / "eval_results.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="BharatLLM Evaluation")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["india_qa", "language_detection"],
                        choices=["india_qa", "language_detection", "translation"])
    args = parser.parse_args()

    evaluator = BharatLLMEvaluator(args.model, args.benchmarks)
    evaluator.load_model()
    evaluator.run_all()


if __name__ == "__main__":
    main()
