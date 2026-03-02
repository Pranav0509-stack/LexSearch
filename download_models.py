"""
BharatLLM Model Downloader
============================
Downloads open-weight models from HuggingFace for merging.

Usage:
    python scripts/download_models.py --models mistral-7b llama3-8b indicbert
    python scripts/download_models.py --list-models
"""

import os
import argparse
from pathlib import Path

from loguru import logger

# Registry of downloadable models
MODEL_REGISTRY = {
    # General purpose (strong English + reasoning)
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-v0.3",
        "size_gb": 14.5,
        "type": "base",
        "gated": False,
        "description": "Mistral 7B base — excellent reasoning, good starting point",
    },
    "mistral-7b-instruct": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "size_gb": 14.5,
        "type": "instruct",
        "gated": False,
        "description": "Mistral 7B instruction-tuned",
    },
    "llama3-8b": {
        "hf_id": "meta-llama/Meta-Llama-3-8B",
        "size_gb": 16.1,
        "type": "base",
        "gated": True,   # Requires HF account + license acceptance
        "description": "LLaMA 3 8B — Meta's best open model, strong benchmarks",
    },
    "llama3-8b-instruct": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "size_gb": 16.1,
        "type": "instruct",
        "gated": True,
        "description": "LLaMA 3 8B instruction-tuned",
    },
    "gemma2-9b": {
        "hf_id": "google/gemma-2-9b",
        "size_gb": 18.0,
        "type": "base",
        "gated": True,
        "description": "Google Gemma 2 9B — strong multilingual capabilities",
    },
    "gemma2-2b": {
        "hf_id": "google/gemma-2-2b",
        "size_gb": 5.0,
        "type": "base",
        "gated": True,
        "description": "Google Gemma 2 2B — small, efficient",
    },
    "qwen2-7b": {
        "hf_id": "Qwen/Qwen2.5-7B",
        "size_gb": 15.0,
        "type": "base",
        "gated": False,
        "description": "Qwen 2.5 7B — excellent multilingual, good for Indian languages via transfer",
    },
    # Indic-specific
    "indicbert": {
        "hf_id": "ai4bharat/indic-bert",
        "size_gb": 0.7,
        "type": "encoder",
        "gated": False,
        "description": "AI4Bharat IndicBERT — trained on 12 Indian languages",
    },
    "ai4bharat-instruct": {
        "hf_id": "ai4bharat/Airavata",
        "size_gb": 14.5,
        "type": "instruct",
        "gated": False,
        "description": "AI4Bharat Airavata — Hindi instruction-tuned LLaMA (7B)",
    },
    "navarasa": {
        "hf_id": "Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0",
        "size_gb": 17.0,
        "type": "instruct",
        "gated": False,
        "description": "Navarasa — multilingual Indian LLM (9 Indian languages)",
    },
    "krutrim": {
        "hf_id": "krutrim-si-designs/Krutrim-spectre-v2",
        "size_gb": 14.5,
        "type": "instruct",
        "gated": True,
        "description": "Ola Krutrim — Indian startup's multilingual LLM",
    },
    # Small/efficient options
    "phi3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "size_gb": 7.6,
        "type": "instruct",
        "gated": False,
        "description": "Microsoft Phi-3 Mini 3.8B — punches above its weight",
    },
}


def list_models():
    """Print all available models."""
    print("\n🇮🇳 BharatLLM — Available Models for Merging\n")
    print(f"{'Model':<25} {'Size':<10} {'Type':<12} {'Gated':<8} {'Description'}")
    print("─" * 90)

    for name, info in MODEL_REGISTRY.items():
        gated = "⚠ Yes" if info["gated"] else "✅ No"
        print(f"{name:<25} {str(info['size_gb'])+'GB':<10} {info['type']:<12} {gated:<8} {info['description']}")

    print("\n⚠ Gated models require: huggingface-cli login + license acceptance")
    print("📋 Full list: https://huggingface.co/models?search=llama+hindi")


def download_model(model_key: str, output_dir: str = "./models"):
    """Download a model from HuggingFace Hub."""
    if model_key not in MODEL_REGISTRY:
        logger.error(f"Unknown model: {model_key}")
        logger.info(f"Available: {list(MODEL_REGISTRY.keys())}")
        return False

    info = MODEL_REGISTRY[model_key]
    hf_id = info["hf_id"]
    output_path = Path(output_dir) / model_key

    if output_path.exists() and any(output_path.glob("*.safetensors")):
        logger.info(f"Model already downloaded: {output_path}")
        return True

    logger.info(f"Downloading {model_key} ({info['size_gb']}GB)...")
    logger.info(f"  HuggingFace ID: {hf_id}")
    logger.info(f"  Output: {output_path}")

    if info["gated"]:
        logger.warning(f"⚠ This model requires license acceptance on HuggingFace!")
        logger.warning(f"  Visit: https://huggingface.co/{hf_id}")
        logger.warning(f"  Then run: huggingface-cli login")

    try:
        from huggingface_hub import snapshot_download

        output_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=hf_id,
            local_dir=str(output_path),
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )

        logger.success(f"Downloaded {model_key} to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Manual download:")
        logger.info(f"  huggingface-cli download {hf_id} --local-dir {output_path}")
        return False


def main():
    parser = argparse.ArgumentParser(description="BharatLLM Model Downloader")
    parser.add_argument("--models", nargs="+", help="Models to download")
    parser.add_argument("--output-dir", default="./models")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models or not args.models:
        list_models()
        return

    for model_key in args.models:
        download_model(model_key, args.output_dir)

    print(f"\n✅ Downloads complete! Models in: {args.output_dir}")
    print("\nNext step — run the merge:")
    print("  python merge_engine/run_merge.py --config configs/bharat_base_merge.yaml")


if __name__ == "__main__":
    main()
