"""
TIES-DARE Merge Algorithm for BharatLLM Forge
==============================================
Implements:
  - DARE: Drop And REscale delta weights (Yu et al., 2024)
  - TIES: Trim, Elect Sign, Merge (Yadav et al., 2023)
  - Indic-aware weight protection for embedding layers

Paper references:
  TIES: https://arxiv.org/abs/2306.01708
  DARE: https://arxiv.org/abs/2311.03099

Usage:
    from merge_engine.ties_dare import TIESDareMerger
    merger = TIESDareMerger(config)
    merged_model = merger.merge(models, base_model)
"""

import os
import copy
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn


# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────

@dataclass
class MergeConfig:
    """Configuration for TIES-DARE merge."""

    # Model weights (must sum to 1.0 or will be normalized)
    model_weights: List[float] = field(default_factory=lambda: [0.5, 0.4, 0.1])

    # DARE parameters
    dare_drop_rate: float = 0.15        # Fraction of delta weights to drop
    dare_rescale: bool = True           # Rescale after dropping
    dare_seed: int = 42                 # Reproducibility

    # TIES parameters
    ties_density: float = 0.3          # Fraction of top-magnitude params to keep
    ties_lambda: float = 1.0           # Scaling factor for merged deltas

    # Indic-specific: protect these layer name patterns from pruning
    indic_protect_layers: List[str] = field(default_factory=lambda: [
        "embed_tokens",     # Token embeddings - critical for new Indic vocab
        "lm_head",          # Output projection
        "embed_positions",  # Position embeddings
    ])

    # Memory management
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    offload_to_cpu: bool = True  # Offload during merge to save VRAM


# ─────────────────────────────────────────
# Core Merge Implementation
# ─────────────────────────────────────────

class TIESDareMerger:
    """
    TIES-DARE merger: the best method for combining heterogeneous models.

    Pipeline:
      1. Compute delta weights = model_i - base_model
      2. DARE: randomly drop & rescale low-signal deltas
      3. TIES: trim to top-density, elect signs, merge
      4. Add back to base model
    """

    def __init__(self, config: MergeConfig):
        self.config = config
        logger.info(f"TIESDareMerger initialized | device={config.device} | density={config.ties_density} | drop_rate={config.dare_drop_rate}")

    def merge(
        self,
        models: List[Dict[str, torch.Tensor]],    # List of model state_dicts
        base_model: Dict[str, torch.Tensor],       # Base model state_dict
        model_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full TIES-DARE merge.

        Args:
            models: List of fine-tuned model state dicts
            base_model: The base/reference model state dict
            model_names: Optional names for logging

        Returns:
            Merged model state dict
        """
        n = len(models)
        if model_names is None:
            model_names = [f"model_{i}" for i in range(n)]

        # Normalize weights
        weights = np.array(self.config.model_weights[:n], dtype=float)
        weights = weights / weights.sum()
        logger.info(f"Merging {n} models: {list(zip(model_names, weights.round(3)))}")

        # Get all parameter keys
        all_keys = set(base_model.keys())
        for m in models:
            all_keys.update(m.keys())

        merged_state = {}
        torch.manual_seed(self.config.dare_seed)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TextColumn("[green]{task.completed}/{task.total}"),
        ) as progress:
            task = progress.add_task("TIES-DARE Merging layers...", total=len(all_keys))

            for key in all_keys:
                progress.advance(task)

                # Skip non-tensor params
                if key not in base_model:
                    # New param from a fine-tuned model — take from first model that has it
                    for m in models:
                        if key in m:
                            merged_state[key] = m[key].clone()
                            break
                    continue

                base_param = base_model[key].float()

                # Check if this layer should be protected (Indic embeddings)
                is_protected = any(p in key for p in self.config.indic_protect_layers)

                # Gather delta weights from all models
                deltas = []
                for i, m in enumerate(models):
                    if key in m:
                        delta = m[key].float() - base_param
                        deltas.append((delta, weights[i], model_names[i]))
                    else:
                        # Model doesn't have this param — zero delta
                        deltas.append((torch.zeros_like(base_param), 0.0, model_names[i]))

                # Skip if no real deltas
                if all(d[1] == 0.0 for d in deltas):
                    merged_state[key] = base_model[key].clone()
                    continue

                # Apply DARE to each delta (skip protected layers)
                dare_deltas = []
                for delta, w, name in deltas:
                    if not is_protected and delta.numel() > 100:
                        dare_delta = self._apply_dare(delta, self.config.dare_drop_rate)
                    else:
                        dare_delta = delta
                    dare_deltas.append((dare_delta, w))

                # Apply TIES merge
                merged_delta = self._apply_ties(dare_deltas, is_protected)

                # Reconstruct: base + merged_delta * lambda
                merged_param = base_param + self.config.ties_lambda * merged_delta

                # Cast back to original dtype
                merged_state[key] = merged_param.to(base_model[key].dtype)

                # Offload to CPU if needed
                if self.config.offload_to_cpu:
                    merged_state[key] = merged_state[key].cpu()

        logger.success(f"Merge complete! {len(merged_state)} parameters merged.")
        return merged_state

    def _apply_dare(self, delta: torch.Tensor, drop_rate: float) -> torch.Tensor:
        """
        DARE: Drop And REscale
        - Randomly zero out (drop_rate * 100)% of delta weights
        - Rescale remaining weights to preserve expected value

        This reduces interference between models by removing low-signal updates.
        """
        if drop_rate <= 0:
            return delta

        # Random binary mask (1 = keep, 0 = drop)
        mask = torch.bernoulli(torch.full_like(delta, 1.0 - drop_rate))

        # Apply drop
        dropped = delta * mask

        # Rescale: E[dropped] = E[delta] * (1 - drop_rate)
        # So multiply by 1/(1-drop_rate) to restore expected magnitude
        if self.config.dare_rescale and drop_rate < 1.0:
            dropped = dropped / (1.0 - drop_rate)

        return dropped

    def _apply_ties(
        self,
        weighted_deltas: List[Tuple[torch.Tensor, float]],
        is_protected: bool = False
    ) -> torch.Tensor:
        """
        TIES: Trim, Elect Sign, Merge

        Step 1 — TRIM: Keep only top-density magnitude parameters per model
        Step 2 — ELECT: Majority vote on sign per parameter position
        Step 3 — MERGE: Average only models that agree with elected sign
        """
        deltas = [d for d, _ in weighted_deltas]
        weights = torch.tensor([w for _, w in weighted_deltas])

        stacked = torch.stack(deltas, dim=0)  # [n_models, ...]

        # ── Step 1: TRIM ──────────────────────────────────────────
        if not is_protected:
            trimmed = self._trim_to_density(stacked, self.config.ties_density)
        else:
            # Protected layers: skip trimming, use all weights
            trimmed = stacked

        # ── Step 2: ELECT SIGN ─────────────────────────────────────
        # For each parameter, sum signed weights — positive or negative wins
        weighted_sum = (trimmed * weights.view(-1, *([1] * (trimmed.dim() - 1)))).sum(dim=0)
        elected_sign = torch.sign(weighted_sum)
        # Where sum is 0, default to +1
        elected_sign = torch.where(elected_sign == 0, torch.ones_like(elected_sign), elected_sign)

        # ── Step 3: MERGE (disjoint mean) ──────────────────────────
        # Only include models whose sign agrees with elected sign
        agrees = (torch.sign(trimmed) == elected_sign.unsqueeze(0)).float()  # [n, ...]
        weighted_agrees = agrees * weights.view(-1, *([1] * (trimmed.dim() - 1)))

        # Weighted mean of agreeing models
        numerator = (trimmed * weighted_agrees).sum(dim=0)
        denominator = weighted_agrees.sum(dim=0).clamp(min=1e-8)
        merged = numerator / denominator

        # Zero out positions where nobody agreed
        merged = merged * (denominator > 0).float()

        return merged

    def _trim_to_density(self, stacked: torch.Tensor, density: float) -> torch.Tensor:
        """
        Keep only top-density fraction of parameters per model (by absolute magnitude).
        Zero out the rest — this removes small noisy updates.
        """
        n_models = stacked.shape[0]
        trimmed = stacked.clone()

        for i in range(n_models):
            flat = stacked[i].abs().flatten()
            if flat.numel() == 0:
                continue
            k = max(1, int(flat.numel() * density))
            threshold, _ = torch.kthvalue(flat, flat.numel() - k + 1)
            mask = stacked[i].abs() >= threshold
            trimmed[i] = stacked[i] * mask

        return trimmed


# ─────────────────────────────────────────
# SLERP (for 2-model merges)
# ─────────────────────────────────────────

class SLERPMerger:
    """
    Spherical Linear Interpolation between two models.
    Better than linear averaging for combining two similar models.
    """

    def __init__(self, t: float = 0.5):
        """
        Args:
            t: Interpolation factor (0.0 = model A, 1.0 = model B)
        """
        self.t = t

    def merge(
        self,
        model_a: Dict[str, torch.Tensor],
        model_b: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Merge two models using SLERP."""
        merged = {}
        for key in model_a:
            if key not in model_b:
                merged[key] = model_a[key]
                continue

            a = model_a[key].float()
            b = model_b[key].float()

            if a.shape != b.shape:
                logger.warning(f"Shape mismatch at {key}: {a.shape} vs {b.shape}, using linear")
                merged[key] = ((1 - self.t) * a + self.t * b).to(model_a[key].dtype)
                continue

            # Flatten for SLERP, then reshape
            a_flat = a.flatten()
            b_flat = b.flatten()

            merged_flat = self._slerp(a_flat, b_flat, self.t)
            merged[key] = merged_flat.reshape(a.shape).to(model_a[key].dtype)

        return merged

    def _slerp(self, v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
        """Core SLERP formula on flattened weight vectors."""
        # Normalize
        v0_norm = v0 / (v0.norm() + 1e-8)
        v1_norm = v1 / (v1.norm() + 1e-8)

        # Compute angle
        dot = torch.clamp((v0_norm * v1_norm).sum(), -1.0, 1.0)
        omega = torch.acos(dot.abs())

        # If nearly parallel, fall back to linear
        if omega.abs() < 1e-6:
            return (1 - t) * v0 + t * v1

        # SLERP formula
        result = (torch.sin((1 - t) * omega) / torch.sin(omega)) * v0 + \
                 (torch.sin(t * omega) / torch.sin(omega)) * v1
        return result


# ─────────────────────────────────────────
# FrankenMerge (layer-wise composition)
# ─────────────────────────────────────────

class FrankenMerger:
    """
    FrankenMerge: Take early layers from one model, later layers from another.

    Concept: LLM layers have functional specialization:
      - Early layers (0-10): syntax, basic language understanding
      - Middle layers (11-20): semantic understanding, facts
      - Late layers (21-31): reasoning, generation quality

    Strategy for BharatLLM:
      - Early layers from Indic model: better language base
      - Late layers from strong English model: better reasoning
    """

    def __init__(self, crossover_layer: int = 16, total_layers: int = 32):
        self.crossover_layer = crossover_layer
        self.total_layers = total_layers

    def merge(
        self,
        model_early: Dict[str, torch.Tensor],   # Source for early layers
        model_late: Dict[str, torch.Tensor],     # Source for late layers
    ) -> Dict[str, torch.Tensor]:
        """Compose layers from two models."""
        merged = {}

        for key in set(list(model_early.keys()) + list(model_late.keys())):
            # Parse layer number from key
            layer_num = self._extract_layer_num(key)

            if layer_num is None:
                # Non-layer param (embeddings, etc.) — use early model
                merged[key] = model_early.get(key, model_late.get(key))
            elif layer_num < self.crossover_layer:
                merged[key] = model_early.get(key, model_late.get(key))
            else:
                merged[key] = model_late.get(key, model_early.get(key))

        logger.info(f"FrankenMerge: layers 0-{self.crossover_layer-1} from model_early, "
                    f"{self.crossover_layer}+ from model_late")
        return merged

    def _extract_layer_num(self, key: str) -> Optional[int]:
        """Extract transformer layer number from parameter key name."""
        import re
        # Handles: layers.12.*, model.layers.12.*, transformer.h.12.*
        patterns = [
            r'\.layers\.(\d+)\.',
            r'\.h\.(\d+)\.',
            r'\.block\.(\d+)\.',
        ]
        for pattern in patterns:
            match = re.search(pattern, key)
            if match:
                return int(match.group(1))
        return None


# ─────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────

def load_model_state_dict(
    model_path: str,
    dtype: torch.dtype = torch.float16
) -> Dict[str, torch.Tensor]:
    """
    Load a model's state dict from HuggingFace format.
    Handles sharded models (model-00001-of-00003.safetensors, etc.)
    """
    from pathlib import Path

    model_dir = Path(model_path)
    logger.info(f"Loading model from: {model_dir}")

    # Try safetensors first (faster, safer)
    safetensor_files = list(model_dir.glob("*.safetensors"))
    if safetensor_files:
        try:
            from safetensors.torch import load_file
            state_dict = {}
            for f in sorted(safetensor_files):
                logger.debug(f"Loading shard: {f.name}")
                shard = load_file(str(f), device="cpu")
                state_dict.update(shard)
            logger.success(f"Loaded {len(state_dict)} tensors from {len(safetensor_files)} safetensor shards")
            return state_dict
        except ImportError:
            logger.warning("safetensors not installed, falling back to PyTorch format")

    # Fall back to pytorch_model.bin
    bin_files = list(model_dir.glob("pytorch_model*.bin"))
    if bin_files:
        state_dict = {}
        for f in sorted(bin_files):
            logger.debug(f"Loading shard: {f.name}")
            shard = torch.load(str(f), map_location="cpu", weights_only=True)
            state_dict.update(shard)
        logger.success(f"Loaded {len(state_dict)} tensors from {len(bin_files)} pytorch shards")
        return state_dict

    raise FileNotFoundError(f"No model weights found in {model_dir}")


def save_merged_model(
    state_dict: Dict[str, torch.Tensor],
    output_dir: str,
    config_source_dir: str,
):
    """
    Save merged model in HuggingFace format with config files copied over.
    """
    import shutil
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save weights as safetensors
    try:
        from safetensors.torch import save_file
        save_file(state_dict, str(out / "model.safetensors"))
        logger.success(f"Saved merged model to {out}/model.safetensors")
    except ImportError:
        torch.save(state_dict, str(out / "pytorch_model.bin"))
        logger.success(f"Saved merged model to {out}/pytorch_model.bin")

    # Copy config files from source model
    config_src = Path(config_source_dir)
    for config_file in ["config.json", "tokenizer.json", "tokenizer_config.json",
                         "tokenizer.model", "special_tokens_map.json", "generation_config.json"]:
        src = config_src / config_file
        if src.exists():
            shutil.copy2(str(src), str(out / config_file))
            logger.debug(f"Copied {config_file}")

    logger.success(f"Merged model saved to: {output_dir}")
