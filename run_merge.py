"""
BharatLLM Forge - Main Merge Runner
=====================================
Orchestrates the complete merge pipeline:
  1. Load source models
  2. Apply TIES-DARE merge
  3. Extend vocabulary with Indic tokens
  4. Save merged model
  5. Basic validation

Usage:
    python merge_engine/run_merge.py --config configs/bharat_base_merge.yaml

    # Or programmatically:
    from merge_engine.run_merge import MergePipeline
    pipeline = MergePipeline.from_yaml("configs/bharat_base_merge.yaml")
    pipeline.run()
"""

import os
import sys
import json
import time
import shutil
import argparse
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import torch
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from ties_dare import TIESDareMerger, SLERPMerger, FrankenMerger, MergeConfig
from ties_dare import load_model_state_dict, save_merged_model
from vocab_fusion import IndicVocabFusion

console = Console()


# ─────────────────────────────────────────
# Pipeline Config
# ─────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Full configuration for the merge pipeline."""

    # Source models
    base_model_path: str = ""
    source_models: List[Dict] = field(default_factory=list)
    # Each item: {"path": "...", "name": "...", "weight": 0.5}

    # Merge strategy
    merge_strategy: str = "ties_dare"  # "ties_dare", "slerp", "franken"

    # TIES-DARE params
    dare_drop_rate: float = 0.15
    ties_density: float = 0.3
    ties_lambda: float = 1.0
    merge_lambda: float = 0.5  # For SLERP

    # Indic vocabulary
    extend_vocab: bool = True
    indic_languages: List[str] = field(default_factory=lambda: ["hi", "bn", "ta", "mr", "te"])
    vocab_extension_size: int = 16000

    # Output
    output_dir: str = "./outputs/bharat-base-7b"

    # System
    device: str = "auto"  # "auto", "cuda", "cpu"
    dtype: str = "float16"  # "float16", "bfloat16", "float32"

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        config = cls()
        for k, v in data.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        config = cls()
        for k, v in data.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config


# ─────────────────────────────────────────
# Merge Pipeline
# ─────────────────────────────────────────

class MergePipeline:
    """
    Full merge pipeline for creating BharatLLM from source models.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = self._resolve_device(config.device)
        self.dtype = self._resolve_dtype(config.dtype)
        self.start_time = None

        logger.info(f"MergePipeline initialized | device={self.device} | strategy={config.merge_strategy}")

    def run(self):
        """Execute the full merge pipeline."""
        self.start_time = time.time()
        self._print_header()

        # Step 1: Validate inputs
        console.rule("[bold cyan]Step 1: Validation")
        self._validate_config()

        # Step 2: Load base model
        console.rule("[bold cyan]Step 2: Load Base Model")
        base_state_dict = load_model_state_dict(self.config.base_model_path, self.dtype)
        self._print_model_stats("Base Model", base_state_dict)

        # Step 3: Load source models
        console.rule("[bold cyan]Step 3: Load Source Models")
        source_state_dicts = []
        source_names = []
        source_weights = []

        for model_cfg in self.config.source_models:
            name = model_cfg.get("name", Path(model_cfg["path"]).name)
            weight = model_cfg.get("weight", 1.0)
            logger.info(f"Loading {name} (weight={weight})")
            sd = load_model_state_dict(model_cfg["path"], self.dtype)
            self._print_model_stats(name, sd)
            source_state_dicts.append(sd)
            source_names.append(name)
            source_weights.append(weight)

        # Step 4: Run merge
        console.rule(f"[bold cyan]Step 4: Merge ({self.config.merge_strategy.upper()})")
        merged_state_dict = self._run_merge(
            base_state_dict,
            source_state_dicts,
            source_names,
            source_weights,
        )

        # Free memory
        del base_state_dict
        for sd in source_state_dicts:
            del sd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 5: Vocabulary extension
        if self.config.extend_vocab:
            console.rule("[bold cyan]Step 5: Indic Vocabulary Extension")
            merged_state_dict = self._extend_vocabulary(merged_state_dict)

        # Step 6: Save
        console.rule("[bold cyan]Step 6: Save Merged Model")
        self._save_model(merged_state_dict)

        # Step 7: Validate output
        console.rule("[bold cyan]Step 7: Validation")
        self._validate_output()

        elapsed = time.time() - self.start_time
        self._print_success(elapsed)

    def _run_merge(
        self,
        base_state_dict,
        source_state_dicts,
        source_names,
        source_weights,
    ) -> Dict[str, torch.Tensor]:
        """Run the selected merge strategy."""

        if self.config.merge_strategy == "ties_dare":
            merge_config = MergeConfig(
                model_weights=source_weights,
                dare_drop_rate=self.config.dare_drop_rate,
                ties_density=self.config.ties_density,
                ties_lambda=self.config.ties_lambda,
                device=str(self.device),
                dtype=self.dtype,
            )
            merger = TIESDareMerger(merge_config)
            return merger.merge(source_state_dicts, base_state_dict, source_names)

        elif self.config.merge_strategy == "slerp":
            if len(source_state_dicts) != 2:
                raise ValueError("SLERP requires exactly 2 source models")
            merger = SLERPMerger(t=self.config.merge_lambda)
            return merger.merge(source_state_dicts[0], source_state_dicts[1])

        elif self.config.merge_strategy == "franken":
            if len(source_state_dicts) != 2:
                raise ValueError("FrankenMerge requires exactly 2 source models")
            merger = FrankenMerger(crossover_layer=16)
            return merger.merge(source_state_dicts[0], source_state_dicts[1])

        else:
            raise ValueError(f"Unknown merge strategy: {self.config.merge_strategy}")

    def _extend_vocabulary(self, state_dict: Dict) -> Dict:
        """Add Indic tokens to the vocabulary and resize embeddings."""
        logger.info(f"Extending vocabulary for languages: {self.config.indic_languages}")

        # Note: In a real pipeline, you'd load the actual tokenizer here
        # and use IndicVocabFusion to add tokens.
        # This is a placeholder that logs the intent.

        fusion = IndicVocabFusion(
            base_tokenizer_dir=self.config.base_model_path,
            vocab_size_extension=self.config.vocab_extension_size,
        )

        metadata = fusion.build_indic_tokenizer(
            output_dir=str(Path(self.config.output_dir) / "indic_tokenizer"),
            languages=self.config.indic_languages,
        )

        logger.info(f"Vocabulary extension config: {metadata}")
        logger.info("NOTE: Load the merged model with transformers to apply embedding resize:")
        logger.info("  from transformers import AutoModelForCausalLM, AutoTokenizer")
        logger.info("  model = AutoModelForCausalLM.from_pretrained(output_dir)")
        logger.info("  tokenizer = AutoTokenizer.from_pretrained(output_dir)")
        logger.info("  fusion.extend_model_vocabulary(model, tokenizer, new_tokens)")

        return state_dict

    def _save_model(self, state_dict: Dict):
        """Save merged model with metadata."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy config files from base model
        config_src = Path(self.config.base_model_path)
        for f in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "tokenizer.model", "special_tokens_map.json", "generation_config.json"]:
            src = config_src / f
            if src.exists():
                shutil.copy2(str(src), str(out_dir / f))

        # Update config.json with merge metadata
        config_file = out_dir / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                cfg = json.load(f)

            cfg["_merge_metadata"] = {
                "merged_by": "bharatlm_forge",
                "merge_strategy": self.config.merge_strategy,
                "source_models": [m["name"] for m in self.config.source_models],
                "indic_languages": self.config.indic_languages,
                "ties_density": self.config.ties_density,
                "dare_drop_rate": self.config.dare_drop_rate,
            }

            with open(config_file, "w") as f:
                json.dump(cfg, f, indent=2)

        # Save weights
        try:
            from safetensors.torch import save_file
            save_file(state_dict, str(out_dir / "model.safetensors"))
            logger.success(f"Saved as safetensors: {out_dir}/model.safetensors")
        except ImportError:
            torch.save(state_dict, str(out_dir / "pytorch_model.bin"))
            logger.success(f"Saved as pytorch: {out_dir}/pytorch_model.bin")

    def _validate_output(self):
        """Quick validation of the saved model."""
        out_dir = Path(self.config.output_dir)

        checks = []

        # Check files exist
        has_weights = (out_dir / "model.safetensors").exists() or \
                     (out_dir / "pytorch_model.bin").exists()
        checks.append(("Model weights", has_weights))
        checks.append(("Config file", (out_dir / "config.json").exists()))
        checks.append(("Tokenizer", (out_dir / "tokenizer.json").exists() or
                                    (out_dir / "tokenizer.model").exists()))

        table = Table(title="Output Validation")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="green")

        for check_name, passed in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            table.add_row(check_name, status)

        console.print(table)

    def _validate_config(self):
        """Validate the merge config before starting."""
        assert self.config.base_model_path, "base_model_path is required"
        assert self.config.source_models, "At least one source model required"
        assert self.config.merge_strategy in ["ties_dare", "slerp", "franken"]

        # Normalize weights
        total_weight = sum(m.get("weight", 1.0) for m in self.config.source_models)
        for m in self.config.source_models:
            m["weight"] = m.get("weight", 1.0) / total_weight

        logger.success("Config validation passed")

    def _print_model_stats(self, name: str, state_dict: Dict):
        """Print model parameter statistics."""
        total_params = sum(t.numel() for t in state_dict.values())
        total_size_gb = sum(t.numel() * t.element_size() for t in state_dict.values()) / 1e9
        logger.info(f"{name}: {total_params/1e9:.2f}B params | {total_size_gb:.1f} GB | {len(state_dict)} tensors")

    def _print_header(self):
        console.print(Panel.fit(
            "[bold saffron1]🇮🇳 BharatLLM Forge[/bold saffron1]\n"
            "[dim]Building India's LLM via model merging[/dim]\n"
            f"[cyan]Strategy: {self.config.merge_strategy.upper()}[/cyan] | "
            f"[green]Models: {len(self.config.source_models)}[/green] | "
            f"[magenta]Languages: {', '.join(self.config.indic_languages)}[/magenta]",
            title="BharatLLM Forge v0.1",
            border_style="orange1",
        ))

    def _print_success(self, elapsed: float):
        console.print(Panel.fit(
            f"[bold green]✅ Merge Complete![/bold green]\n"
            f"[dim]Time elapsed: {elapsed/60:.1f} minutes[/dim]\n"
            f"[cyan]Output: {self.config.output_dir}[/cyan]\n\n"
            "[bold]Next steps:[/bold]\n"
            "  1. Fine-tune: [cyan]python finetune/train.py --config configs/finetune_bharat.yaml[/cyan]\n"
            "  2. Chat: [cyan]python scripts/chat.py --model outputs/bharat-base-7b[/cyan]\n"
            "  3. Evaluate: [cyan]python eval/evaluate.py --model outputs/bharat-base-7b[/cyan]",
            title="Success",
            border_style="green",
        ))

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        return {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}.get(dtype, torch.float16)


# ─────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BharatLLM Forge — Merge Engine")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--base-model", type=str, help="Base model path (overrides config)")
    parser.add_argument("--output-dir", type=str, default="./outputs/bharat-base-7b")
    parser.add_argument("--strategy", type=str, default="ties_dare",
                        choices=["ties_dare", "slerp", "franken"])
    parser.add_argument("--dry-run", action="store_true", help="Validate config without running")
    args = parser.parse_args()

    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        # Build config from command line for quick testing
        logger.warning("No config file provided. Using defaults.")
        config = PipelineConfig(
            base_model_path=args.base_model or "./models/mistral-7b",
            source_models=[
                {"name": "mistral-7b", "path": "./models/mistral-7b", "weight": 0.6},
                {"name": "llama3-8b", "path": "./models/llama3-8b", "weight": 0.4},
            ],
            merge_strategy=args.strategy,
            output_dir=args.output_dir,
        )

    if args.dry_run:
        logger.info("DRY RUN — validating config only")
        pipeline = MergePipeline(config)
        pipeline._validate_config()
        logger.success("Config is valid!")
        return

    pipeline = MergePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
