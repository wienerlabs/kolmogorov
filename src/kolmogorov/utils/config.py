"""Configuration loading and management utilities."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file with inheritance support."""
    config_path = Path(config_path)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if "_extends" in config:
        base_path = config_path.parent / config.pop("_extends")
        base_config = load_config(base_path)
        config = _deep_merge(base_config, config)
    
    return config


def save_config(config: dict[str, Any], config_path: str | Path) -> None:
    """Save configuration to YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def get_training_args(config: dict[str, Any], trainer_type: str = "base") -> dict[str, Any]:
    """Extract training arguments from config for specific trainer type."""
    training_config = config.get("training", {}).copy()
    
    type_specific = config.get(trainer_type, {})
    for key, value in type_specific.items():
        if key not in ["max_length", "max_prompt_length"]:
            training_config[key] = value
    
    return training_config

