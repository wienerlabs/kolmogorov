"""LoRA configuration and application utilities."""

from typing import Any

from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import PreTrainedModel

from kolmogorov.utils.logging import get_logger

logger = get_logger(__name__)


def get_lora_config(config: dict[str, Any], task_type: str = "CAUSAL_LM") -> LoraConfig | None:
    """Create LoraConfig from configuration dictionary."""
    lora_config = config.get("lora", {})
    
    if not lora_config.get("enabled", True):
        return None
    
    task_type_enum = getattr(TaskType, task_type.upper(), TaskType.CAUSAL_LM)
    
    lora_kwargs = {
        "r": lora_config.get("r", 16),
        "lora_alpha": lora_config.get("lora_alpha", 32),
        "lora_dropout": lora_config.get("lora_dropout", 0.05),
        "bias": lora_config.get("bias", "none"),
        "task_type": task_type_enum,
    }
    
    if "target_modules" in lora_config:
        lora_kwargs["target_modules"] = lora_config["target_modules"]
    
    if "modules_to_save" in lora_config:
        lora_kwargs["modules_to_save"] = lora_config["modules_to_save"]
    
    return LoraConfig(**lora_kwargs)


def apply_lora(model: PreTrainedModel, config: dict[str, Any], task_type: str = "CAUSAL_LM") -> PeftModel:
    """Apply LoRA adapters to model."""
    lora_config = get_lora_config(config, task_type)
    
    if lora_config is None:
        logger.info("LoRA disabled, returning original model")
        return model
    
    logger.info(f"Applying LoRA with r={lora_config.r}, alpha={lora_config.lora_alpha}")
    peft_model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )
    
    return peft_model


def print_trainable_parameters(model: PreTrainedModel) -> dict[str, Any]:
    """Print and return trainable parameter statistics."""
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    stats = {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": 100 * trainable_params / all_params if all_params > 0 else 0,
    }
    
    logger.info(
        f"trainable params: {stats['trainable_params']:,} || "
        f"all params: {stats['all_params']:,} || "
        f"trainable%: {stats['trainable_percent']:.4f}"
    )
    
    return stats

