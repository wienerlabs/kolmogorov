"""Model loading utilities with quantization and device mapping support."""

from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from kolmogorov.utils.logging import get_logger

logger = get_logger(__name__)


def load_tokenizer(model_name_or_path: str, **kwargs) -> PreTrainedTokenizer:
    """Load tokenizer with sensible defaults."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=kwargs.get("trust_remote_code", True),
        **{k: v for k, v in kwargs.items() if k != "trust_remote_code"}
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


def get_quantization_config(config: dict[str, Any]) -> BitsAndBytesConfig | None:
    """Create BitsAndBytesConfig from config dict."""
    quant_config = config.get("quantization", {})
    
    if not quant_config.get("enabled", False):
        return None
    
    compute_dtype = quant_config.get("bnb_4bit_compute_dtype", "bfloat16")
    if isinstance(compute_dtype, str):
        compute_dtype = getattr(torch, compute_dtype)
    
    return BitsAndBytesConfig(
        load_in_4bit=quant_config.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
    )


def load_model(
    model_name_or_path: str,
    config: dict[str, Any] | None = None,
    model_type: str = "causal_lm",
) -> PreTrainedModel:
    """Load model with optional quantization."""
    config = config or {}
    model_config = config.get("model", {})
    
    quantization_config = get_quantization_config(config)
    
    dtype = model_config.get("dtype", "auto")
    if dtype != "auto":
        dtype = getattr(torch, dtype)
    
    model_class = (
        AutoModelForSequenceClassification 
        if model_type == "reward" 
        else AutoModelForCausalLM
    )
    
    model_kwargs = {
        "pretrained_model_name_or_path": model_name_or_path,
        "device_map": model_config.get("device_map", "auto"),
        "trust_remote_code": model_config.get("trust_remote_code", True),
        "use_cache": model_config.get("use_cache", False),
    }
    
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    if model_type == "reward":
        model_kwargs["num_labels"] = 1
    
    logger.info(f"Loading model: {model_name_or_path}")
    model = model_class.from_pretrained(**model_kwargs)
    
    return model


def load_model_for_training(
    model_name_or_path: str,
    config: dict[str, Any],
    model_type: str = "causal_lm",
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer configured for training."""
    model = load_model(model_name_or_path, config, model_type)
    tokenizer = load_tokenizer(model_name_or_path)
    
    if config.get("training", {}).get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    
    return model, tokenizer

