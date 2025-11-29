"""SFT (Supervised Fine-Tuning) trainer wrapper."""

from typing import Any
from pathlib import Path

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import SFTTrainer, SFTConfig

from kolmogorov.models.loader import load_model_for_training
from kolmogorov.models.lora_config import get_lora_config
from kolmogorov.utils.config import load_config
from kolmogorov.utils.logging import get_logger, generate_run_name

logger = get_logger(__name__)


class SFTTrainerWrapper:
    """Wrapper around TRL's SFTTrainer with config-driven setup."""
    
    def __init__(
        self,
        config_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
        model: PreTrainedModel | str | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
    ):
        self.config = config or (load_config(config_path) if config_path else {})
        self._model = model
        self._tokenizer = tokenizer
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self._trainer: SFTTrainer | None = None
    
    def _create_training_args(self) -> SFTConfig:
        """Create SFTConfig from configuration."""
        training = self.config.get("training", {})
        sft = self.config.get("sft", {})
        logging_config = self.config.get("logging", {})
        
        run_name = logging_config.get("run_name") or generate_run_name("sft")
        
        return SFTConfig(
            output_dir=training.get("output_dir", "./checkpoints/sft"),
            num_train_epochs=training.get("num_train_epochs", 3),
            per_device_train_batch_size=training.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=training.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=training.get("gradient_accumulation_steps", 4),
            learning_rate=training.get("learning_rate", 2e-5),
            weight_decay=training.get("weight_decay", 0.01),
            warmup_ratio=training.get("warmup_ratio", 0.1),
            lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
            max_grad_norm=training.get("max_grad_norm", 1.0),
            logging_steps=training.get("logging_steps", 10),
            save_steps=training.get("save_steps", 500),
            eval_steps=training.get("eval_steps", 100),
            save_total_limit=training.get("save_total_limit", 3),
            bf16=training.get("bf16", True),
            gradient_checkpointing=training.get("gradient_checkpointing", True),
            remove_unused_columns=training.get("remove_unused_columns", False),
            max_seq_length=sft.get("max_length", 2048),
            packing=sft.get("packing", False),
            report_to=logging_config.get("report_to", "none"),
            run_name=run_name,
        )
    
    def setup(self) -> "SFTTrainerWrapper":
        """Initialize the trainer with all components."""
        model_config = self.config.get("model", {})
        model_name = model_config.get("name_or_path", "Qwen/Qwen2.5-0.5B")
        
        if isinstance(self._model, str):
            model_name = self._model
            self._model = None
        
        if self._model is None or self._tokenizer is None:
            self._model, self._tokenizer = load_model_for_training(
                model_name, self.config, model_type="causal_lm"
            )
        
        training_args = self._create_training_args()
        peft_config = get_lora_config(self.config, task_type="CAUSAL_LM")
        
        logger.info("Initializing SFT trainer")
        
        self._trainer = SFTTrainer(
            model=self._model,
            args=training_args,
            train_dataset=self._train_dataset,
            eval_dataset=self._eval_dataset,
            processing_class=self._tokenizer,
            peft_config=peft_config,
        )
        
        return self
    
    def train(self) -> dict[str, Any]:
        """Run training and return metrics."""
        if self._trainer is None:
            self.setup()
        
        logger.info("Starting SFT training...")
        result = self._trainer.train()
        
        return {"train_loss": result.training_loss, "metrics": result.metrics}
    
    def save(self, output_dir: str | Path | None = None) -> None:
        """Save the trained model."""
        if self._trainer is None:
            raise ValueError("Trainer not initialized")
        
        save_path = output_dir or self.config.get("training", {}).get("output_dir")
        self._trainer.save_model(save_path)
        logger.info(f"Model saved to {save_path}")
    
    @property
    def trainer(self) -> SFTTrainer:
        if self._trainer is None:
            raise ValueError("Trainer not initialized. Call setup() first.")
        return self._trainer

