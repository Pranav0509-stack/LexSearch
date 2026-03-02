"""
Indic Vocabulary Fusion for BharatLLM
======================================
Extends an existing model's vocabulary with Indic script tokens without
retraining the full model.

Method:
  1. Train a SentencePiece model on Indic text
  2. Find tokens unique to Indic model not in base model
  3. Initialize new token embeddings using:
     - Subword composition: average existing subword embeddings
     - Transliteration: use romanized equivalent embeddings
     - Random init (fallback) with small std
  4. Resize model's embedding matrix
  5. Fine-tune embeddings while keeping rest frozen
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger


INDIC_SCRIPTS = {
    "devanagari": (0x0900, 0x097F),  # Hindi, Marathi, Sanskrit
    "bengali":    (0x0980, 0x09FF),
    "gurmukhi":   (0x0A00, 0x0A7F),  # Punjabi
    "gujarati":   (0x0A80, 0x0AFF),
    "oriya":      (0x0B00, 0x0B7F),
    "tamil":      (0x0B80, 0x0BFF),
    "telugu":     (0x0C00, 0x0C7F),
    "kannada":    (0x0C80, 0x0CFF),
    "malayalam":  (0x0D00, 0x0D7F),
}


class IndicVocabFusion:
    """
    Fuses Indic vocabulary into an existing English-dominant LLM.
    """

    def __init__(
        self,
        base_tokenizer_dir: str,
        indic_corpus_path: Optional[str] = None,
        vocab_size_extension: int = 16000,  # How many new tokens to add
        embed_init_strategy: str = "subword_avg",  # or "random"
    ):
        self.base_tokenizer_dir = Path(base_tokenizer_dir)
        self.indic_corpus_path = indic_corpus_path
        self.vocab_size_extension = vocab_size_extension
        self.embed_init_strategy = embed_init_strategy

    def train_indic_spm(
        self,
        corpus_path: str,
        model_prefix: str = "indic_spm",
        vocab_size: int = 32000,
        character_coverage: float = 0.9995,
    ):
        """
        Train a SentencePiece model on Indic text data.
        character_coverage=0.9995 ensures good coverage of all Indic scripts.
        """
        import sentencepiece as spm

        logger.info(f"Training SentencePiece model on {corpus_path}")
        logger.info(f"Vocab size: {vocab_size}, Coverage: {character_coverage}")

        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type="bpe",
            pad_id=3,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            # Important for Indic: don't normalize unicode aggressively
            normalization_rule_name="nmt_nfkc_cf",
            add_dummy_prefix=False,
            # Include whitespace for word boundary
            user_defined_symbols=["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
        )

        logger.success(f"SentencePiece model saved: {model_prefix}.model")
        return f"{model_prefix}.model"

    def find_new_tokens(
        self,
        base_tokenizer,
        indic_spm_model: str,
        min_freq: int = 10,
    ) -> List[str]:
        """
        Find tokens in the Indic SPM that are not in the base tokenizer.
        These are the tokens we need to add.
        """
        import sentencepiece as spm

        indic_sp = spm.SentencePieceProcessor()
        indic_sp.load(indic_spm_model)

        base_vocab = set(base_tokenizer.get_vocab().keys())
        new_tokens = []

        for token_id in range(indic_sp.get_piece_size()):
            token = indic_sp.id_to_piece(token_id)
            if token not in base_vocab and self._is_indic_token(token):
                new_tokens.append(token)

        logger.info(f"Found {len(new_tokens)} new Indic tokens not in base vocabulary")
        return new_tokens[:self.vocab_size_extension]

    def _is_indic_token(self, token: str) -> bool:
        """Check if a token contains any Indic script characters."""
        for ch in token:
            cp = ord(ch)
            for script_name, (start, end) in INDIC_SCRIPTS.items():
                if start <= cp <= end:
                    return True
        return False

    def initialize_new_embeddings(
        self,
        new_tokens: List[str],
        base_tokenizer,
        base_embeddings: torch.Tensor,  # shape: [vocab_size, embed_dim]
        strategy: str = "subword_avg",
    ) -> torch.Tensor:
        """
        Initialize embeddings for new Indic tokens.

        Strategies:
          - subword_avg: Average embeddings of constituent characters/subwords
          - random: Small random initialization

        Returns:
            new_embeddings: shape [n_new_tokens, embed_dim]
        """
        embed_dim = base_embeddings.shape[-1]
        new_embeddings = torch.zeros(len(new_tokens), embed_dim)

        for i, token in enumerate(new_tokens):
            if strategy == "subword_avg":
                # Tokenize the Indic token using base tokenizer
                # This gives us existing subword pieces to average
                try:
                    subword_ids = base_tokenizer.encode(
                        token,
                        add_special_tokens=False,
                        return_tensors="pt"
                    ).squeeze()

                    if subword_ids.numel() > 0:
                        # Average the embeddings of constituent subwords
                        subword_embeds = base_embeddings[subword_ids]
                        new_embeddings[i] = subword_embeds.mean(dim=0)
                    else:
                        # Fallback: use <unk> embedding + small noise
                        unk_id = base_tokenizer.unk_token_id or 0
                        new_embeddings[i] = base_embeddings[unk_id] + \
                            torch.randn(embed_dim) * 0.01
                except Exception:
                    new_embeddings[i] = torch.randn(embed_dim) * 0.02
            else:
                # Random init with small std (standard practice)
                new_embeddings[i] = torch.randn(embed_dim) * 0.02

        logger.info(f"Initialized {len(new_tokens)} new embeddings using '{strategy}'")
        return new_embeddings

    def extend_model_vocabulary(
        self,
        model,
        tokenizer,
        new_tokens: List[str],
        init_strategy: str = "subword_avg",
    ):
        """
        Add new tokens to both the tokenizer and the model's embedding matrices.

        This modifies:
          - tokenizer: adds new tokens
          - model.embed_tokens: extends embedding matrix
          - model.lm_head: extends output projection
        """
        logger.info(f"Extending vocabulary with {len(new_tokens)} Indic tokens...")

        # Add tokens to tokenizer
        num_added = tokenizer.add_tokens(new_tokens)
        logger.info(f"Added {num_added} tokens to tokenizer")

        if num_added == 0:
            logger.warning("No new tokens were added (all already in vocabulary)")
            return model, tokenizer

        # Get current embeddings before resize
        old_embed = model.get_input_embeddings().weight.data.clone()

        # Resize model embeddings (this adds zero rows)
        model.resize_token_embeddings(len(tokenizer))
        new_embed = model.get_input_embeddings().weight.data

        # Initialize the new token rows intelligently
        new_token_embeddings = self.initialize_new_embeddings(
            new_tokens[:num_added],
            tokenizer,
            old_embed,
            strategy=init_strategy,
        )

        # Fill in new token rows
        old_vocab_size = old_embed.shape[0]
        with torch.no_grad():
            new_embed[old_vocab_size:old_vocab_size + num_added] = \
                new_token_embeddings.to(new_embed.dtype)

        logger.success(f"Vocabulary extended: {old_vocab_size} → {len(tokenizer)} tokens")
        return model, tokenizer

    def build_indic_tokenizer(
        self,
        output_dir: str,
        languages: List[str] = ["hi", "bn", "ta", "mr", "te", "gu", "pa"],
    ) -> Dict:
        """
        Build and save configuration for a combined Indic+English tokenizer.
        Returns metadata about the tokenizer.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Script ranges for requested languages
        lang_to_script = {
            "hi": "devanagari", "mr": "devanagari", "ne": "devanagari",
            "bn": "bengali",
            "pa": "gurmukhi",
            "gu": "gujarati",
            "or": "oriya",
            "ta": "tamil",
            "te": "telugu",
            "kn": "kannada",
            "ml": "malayalam",
        }

        scripts_needed = set()
        for lang in languages:
            if lang in lang_to_script:
                scripts_needed.add(lang_to_script[lang])

        metadata = {
            "languages": languages,
            "scripts": list(scripts_needed),
            "unicode_ranges": {
                script: {"start": hex(INDIC_SCRIPTS[script][0]), "end": hex(INDIC_SCRIPTS[script][1])}
                for script in scripts_needed if script in INDIC_SCRIPTS
            },
            "vocab_extension": self.vocab_size_extension,
            "init_strategy": self.embed_init_strategy,
        }

        # Save metadata
        with open(out / "indic_tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.success(f"Indic tokenizer config saved to {out}")
        return metadata


# ─────────────────────────────────────────────
# Hinglish / Code-Switch Handler
# ─────────────────────────────────────────────

class CodeSwitchHandler:
    """
    Handle Hinglish and other Indian code-switching patterns.
    Ensures the merged model can handle mixed-language text naturally.

    Code-switching examples:
      "Main aaj bahut busy tha, isliye call nahi kar saka"
      "Let's plan karte hain for tomorrow"
      "Yaar, this movie was ekdum solid!"
    """

    HINGLISH_PATTERNS = [
        # Common Hindi words in Latin script (Romanized Hindi)
        "acha", "theek", "yaar", "bhai", "didi", "kya", "nahi",
        "bahut", "aaj", "kal", "abhi", "phir", "matlab", "matlab",
        "ekdum", "bilkul", "zaroor", "shukriya", "namaste",
    ]

    def detect_code_switch(self, text: str) -> Dict:
        """Detect if text contains code-switching and which scripts are present."""
        result = {
            "is_code_switched": False,
            "scripts": set(),
            "language_spans": [],
        }

        current_script = None
        span_start = 0

        for i, ch in enumerate(text):
            cp = ord(ch)
            script = self._detect_char_script(cp)

            if script != current_script:
                if current_script is not None:
                    result["language_spans"].append({
                        "script": current_script,
                        "start": span_start,
                        "end": i,
                        "text": text[span_start:i],
                    })
                current_script = script
                span_start = i

        result["scripts"] = {span["script"] for span in result["language_spans"]}
        result["is_code_switched"] = len(result["scripts"]) > 1

        return result

    def _detect_char_script(self, codepoint: int) -> str:
        """Detect Unicode script for a codepoint."""
        if 0x0041 <= codepoint <= 0x007A:
            return "latin"
        for script_name, (start, end) in INDIC_SCRIPTS.items():
            if start <= codepoint <= end:
                return script_name
        return "other"
