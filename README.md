
<img width="300" height="300" alt="kolmo" src="https://github.com/user-attachments/assets/5e8280b4-72a7-4ebb-a37a-613c07b0a618" />

---

**Build a reinforcement learning system that enables an LLM to learn continuously from real-world human interactions using preference-based feedback mechanisms.**

## Overview

Kolmogorov implements a complete RLHF (Reinforcement Learning from Human Feedback) pipeline with:

- **Preference Data Collection** — Structured logging for queries, response candidates, and user preferences
- **Reward Model Training** — Bradley-Terry preference learning using TRL's RewardTrainer
- **DPO (Direct Preference Optimization)** — Align models with human preferences without a separate reward model
- **LoRA Fine-Tuning** — Parameter-efficient training for memory efficiency

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Train with DPO

```bash
python scripts/train_dpo.py \
    --dataset trl-lib/ultrafeedback_binarized \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --max_samples 1000
```

### Train a Reward Model

```bash
python scripts/train_reward.py \
    --dataset trl-lib/ultrafeedback_binarized \
    --model Qwen/Qwen2.5-0.5B-Instruct
```

### Supervised Fine-Tuning

```bash
python scripts/train_sft.py \
    --dataset trl-lib/Capybara \
    --model Qwen/Qwen2.5-0.5B
```

## Python API

```python
from kolmogorov.trainers import DPOTrainerWrapper
from kolmogorov.data import PreferenceDataset
from datasets import load_dataset

# Load preference data
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# Train with DPO
trainer = DPOTrainerWrapper(
    config_path="configs/dpo_config.yaml",
    train_dataset=dataset,
)
trainer.setup()
trainer.train()
trainer.save()
```

## Collecting Preference Data

```python
from kolmogorov.data import PreferenceCollector

collector = PreferenceCollector(storage_path="data/preferences")

# Add pairwise comparison
collector.add_comparison(
    prompt="What is machine learning?",
    response_a="ML is a subset of AI...",
    response_b="Machine learning is...",
    preferred="a",
    domain="education",
)

# Export for training
training_data = collector.export_for_training("data/train.json")
```

## Project Structure

```
kolmogorov/
├── configs/                # YAML configuration files
│   ├── base_config.yaml
│   ├── dpo_config.yaml
│   ├── reward_config.yaml
│   └── sft_config.yaml
├── scripts/                # Training scripts
│   ├── train_dpo.py
│   ├── train_reward.py
│   └── train_sft.py
├── src/kolmogorov/
│   ├── data/              # Data collection & processing
│   ├── models/            # Model loading & LoRA
│   ├── trainers/          # DPO, Reward, SFT trainers
│   ├── evaluation/        # Metrics & evaluation
│   └── utils/             # Config & logging
└── pyproject.toml
```

## Configuration

All training parameters are configurable via YAML files in `configs/`. Key settings:

```yaml
model:
  name_or_path: "Qwen/Qwen2.5-0.5B-Instruct"

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

dpo:
  beta: 0.1
  loss_type: "sigmoid"
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.1.0
- TRL >= 0.8.0
- PEFT >= 0.10.0
- Transformers >= 4.40.0

## License

MIT
