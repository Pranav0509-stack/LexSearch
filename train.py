"""
BharatLLM QLoRA Fine-Tuning
=============================
Fine-tunes the merged BharatLLM base on Indian language datasets
using QLoRA (Quantized LoRA) — runs on a single 16-24GB GPU.

Techniques:
  - 4-bit NF4 quantization (bitsandbytes)
  - LoRA rank-64 adapters on attention layers
  - Gradient checkpointing
  - Flash Attention 2 (if available)
  - Mixed-language instruction tuning

Training Data Format:
  {"instruction": "...", "input": "...", "output": "...", "lang": "hi"}

Usage:
    python finetune/train.py --config configs/finetune_bharat.yaml

    # Quick test on single GPU:
    python finetune/train.py \
        --model-path ./outputs/bharat-base-7b \
        --data-dir ./data/processed \
        --output-dir ./outputs/bharat-llm-7b \
        --languages hi bn ta \
        --max-steps 5000
"""

import os
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import yaml

from loguru import logger


# ─────────────────────────────────────────
# Training Config
# ─────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Full training configuration."""

    # Model
    model_path: str = "./outputs/bharat-base-7b"
    output_dir: str = "./outputs/bharat-llm-7b"

    # Data
    data_dir: str = "./data/processed"
    languages: List[str] = field(default_factory=lambda: ["hi", "bn", "ta", "mr"])
    max_seq_length: int = 2048
    max_samples_per_lang: Optional[int] = None  # None = use all

    # LoRA
    lora_rank: int = 64
    lora_alpha: int = 128        # Usually 2x rank
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",       # MLP/FFN
    ])

    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True

    # Training
    num_train_epochs: int = 3
    max_steps: int = -1             # -1 means use num_train_epochs
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8   # Effective batch = 4*8 = 32
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05

    # Optimization
    optim: str = "paged_adamw_8bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True

    # Logging & Saving
    logging_steps: int = 25
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    report_to: str = "wandb"       # "wandb", "tensorboard", "none"

    # Dataset format
    dataset_format: str = "instruction"  # "instruction", "text", "chat"

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        config = cls()
        for k, v in data.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config


# ─────────────────────────────────────────
# Dataset Loader
# ─────────────────────────────────────────

class IndicDatasetLoader:
    """
    Loads and formats Indian language data for instruction tuning.
    Supports multiple formats and mixes languages.
    """

    # Instruction templates for different languages
    INSTRUCTION_TEMPLATES = {
        "hi": "### निर्देश:\n{instruction}\n\n### इनपुट:\n{input}\n\n### उत्तर:\n{output}",
        "bn": "### নির্দেশ:\n{instruction}\n\n### ইনপুট:\n{input}\n\n### উত্তর:\n{output}",
        "ta": "### வழிமுறை:\n{instruction}\n\n### உள்ளீடு:\n{input}\n\n### வெளியீடு:\n{output}",
        "te": "### సూచన:\n{instruction}\n\n### ఇన్‌పుట్:\n{input}\n\n### అవుట్‌పుట్:\n{output}",
        "mr": "### सूचना:\n{instruction}\n\n### इनपुट:\n{input}\n\n### उत्तर:\n{output}",
        "en": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
        # Hinglish
        "hinglish": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
    }

    # Default to English template for unknown languages
    DEFAULT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Output:\n{output}"

    def load_from_dir(
        self,
        data_dir: str,
        languages: List[str],
        max_samples_per_lang: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> List[Dict]:
        """
        Load all JSONL files from the processed data directory.
        Mix languages proportionally.
        """
        import random
        random.seed(seed)

        all_samples = []
        data_path = Path(data_dir)

        for lang in languages:
            lang_samples = []

            # Search for data files
            patterns = [
                f"**/{lang}/train.jsonl",
                f"**/{lang}/*.jsonl",
                f"{lang}.jsonl",
            ]

            files = []
            for pattern in patterns:
                files.extend(data_path.glob(pattern))

            if not files:
                logger.warning(f"No data files found for language: {lang}")
                continue

            for file in files:
                with open(file, encoding="utf-8") as f:
                    for line in f:
                        try:
                            sample = json.loads(line.strip())
                            if "text" in sample:
                                lang_samples.append(sample)
                        except json.JSONDecodeError:
                            continue

            if max_samples_per_lang:
                random.shuffle(lang_samples)
                lang_samples = lang_samples[:max_samples_per_lang]

            logger.info(f"Loaded {len(lang_samples)} samples for {lang}")
            all_samples.extend(lang_samples)

        if shuffle:
            random.shuffle(all_samples)

        return all_samples

    def format_for_training(
        self,
        samples: List[Dict],
        format_type: str = "text",
        add_eos: bool = True,
        eos_token: str = "</s>",
    ) -> List[str]:
        """
        Format samples for causal language modeling.

        Formats:
          - "text": raw text, simple continuation
          - "instruction": Alpaca-style instruction tuning
          - "chat": ChatML format (<|im_start|>user...)
        """
        formatted = []

        for sample in samples:
            lang = sample.get("lang", "en")

            if format_type == "text":
                text = sample.get("text", "")

            elif format_type == "instruction":
                template = self.INSTRUCTION_TEMPLATES.get(lang, self.DEFAULT_TEMPLATE)
                text = template.format(
                    instruction=sample.get("instruction", ""),
                    input=sample.get("input", ""),
                    output=sample.get("output", sample.get("text", "")),
                )

            elif format_type == "chat":
                # ChatML format
                messages = sample.get("messages", [])
                if not messages:
                    # Convert text to chat format
                    messages = [
                        {"role": "user", "content": sample.get("instruction", "")},
                        {"role": "assistant", "content": sample.get("output", sample.get("text", ""))},
                    ]

                text = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    text += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            else:
                text = sample.get("text", "")

            if text:
                if add_eos:
                    text = text + eos_token
                formatted.append(text)

        return formatted

    def build_instruction_dataset(
        self,
        languages: List[str],
        n_samples: int = 10000,
    ) -> List[Dict]:
        """
        Build a synthetic instruction dataset in multiple Indian languages.
        Uses templates to create instruction-following samples.
        """
        templates = [
            # Translation tasks
            {
                "instruction_en": "Translate the following Hindi text to English:",
                "instruction_hi": "निम्नलिखित अंग्रेजी पाठ का हिंदी में अनुवाद करें:",
                "type": "translation",
            },
            # Summarization
            {
                "instruction_en": "Summarize the following text in 2-3 sentences:",
                "instruction_hi": "निम्नलिखित पाठ को 2-3 वाक्यों में सारांशित करें:",
                "type": "summarization",
            },
            # Q&A
            {
                "instruction_en": "Answer the following question:",
                "instruction_hi": "निम्नलिखित प्रश्न का उत्तर दें:",
                "type": "qa",
            },
        ]

        samples = []
        logger.info(f"Building instruction dataset for {languages}")
        return samples  # In production: populate from actual data sources


# ─────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────

class BharatLLMTrainer:
    """
    QLoRA trainer for BharatLLM.
    Wraps HuggingFace transformers + peft + trl.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        logger.info(f"BharatLLMTrainer | model={config.model_path} | langs={config.languages}")

    def setup(self):
        """Load model, tokenizer, and prepare for training."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            use_fast=True,
            padding_side="right",
        )

        # Set pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token = eos_token")

        logger.info("Loading model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.use_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.use_double_quant,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_flash_attention_2=self._check_flash_attn(),
        )

        # Prepare model for QLoRA training
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config.gradient_checkpointing,
        )

        # Apply LoRA
        logger.info(f"Applying LoRA: rank={self.config.lora_rank}, alpha={self.config.lora_alpha}")
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        logger.success("Model and LoRA adapters ready for training")
        return self

    def prepare_dataset(self):
        """Load and format training data."""
        from datasets import Dataset

        loader = IndicDatasetLoader()

        # Load raw samples
        samples = loader.load_from_dir(
            self.config.data_dir,
            self.config.languages,
            self.config.max_samples_per_lang,
        )
        logger.info(f"Total samples loaded: {len(samples)}")

        # Format for training
        texts = loader.format_for_training(
            samples,
            format_type=self.config.dataset_format,
        )
        logger.info(f"Total formatted samples: {len(texts)}")

        # Tokenize
        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
            )

        dataset = Dataset.from_dict({"text": texts})
        dataset = dataset.map(tokenize, batched=True, num_proc=4, remove_columns=["text"])

        logger.success(f"Dataset prepared: {len(dataset)} tokenized samples")
        self.dataset = dataset
        return self

    def train(self):
        """Run the training loop."""
        from transformers import TrainingArguments
        from trl import SFTTrainer

        logger.info("Starting training...")

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            report_to=self.config.report_to,
            gradient_checkpointing=self.config.gradient_checkpointing,
            group_by_length=True,        # Group similar lengths → less padding waste
            dataloader_num_workers=4,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            args=training_args,
            max_seq_length=self.config.max_seq_length,
            packing=True,   # Pack multiple short sequences → better GPU utilization
        )

        logger.info("Training started!")
        trainer.train()

        # Save final model (merge LoRA into base weights)
        logger.info("Saving final model (merging LoRA adapters)...")
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        logger.success(f"Training complete! Model saved to: {self.config.output_dir}")

    def _check_flash_attn(self) -> bool:
        """Check if Flash Attention 2 is available."""
        try:
            import flash_attn
            logger.info("Flash Attention 2 available — using for faster training")
            return True
        except ImportError:
            logger.info("Flash Attention 2 not available, using standard attention")
            return False


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BharatLLM QLoRA Fine-Tuning")
    parser.add_argument("--config", type=str, help="YAML config path")
    parser.add_argument("--model-path", type=str, default="./outputs/bharat-base-7b")
    parser.add_argument("--data-dir", type=str, default="./data/processed")
    parser.add_argument("--output-dir", type=str, default="./outputs/bharat-llm-7b")
    parser.add_argument("--languages", nargs="+", default=["hi", "bn", "ta", "mr"])
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            languages=args.languages,
            max_steps=args.max_steps,
            lora_rank=args.lora_rank,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
        )

    trainer = BharatLLMTrainer(config)
    trainer.setup()
    trainer.prepare_dataset()
    trainer.train()


if __name__ == "__main__":
    main()
