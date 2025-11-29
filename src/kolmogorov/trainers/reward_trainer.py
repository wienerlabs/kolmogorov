"""Reward Model trainer wrapper."""

from typing import Any
from pathlib import Path

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig

from kolmogorov.models.loader import load_model_for_training
from kolmogorov.models.lora_config import get_lora_config
from kolmogorov.utils.config import load_config
from kolmogorov.utils.logging import get_logger, generate_run_name

logger = get_logger(__name__)


class RewardTrainerWrapper:
    """Wrapper around TRL's RewardTrainer with config-driven setup."""
    
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
        self._trainer: RewardTrainer | None = None
    
    def _create_training_args(self) -> RewardConfig:
        """Create RewardConfig from configuration."""
        training = self.config.get("training", {})
        reward = self.config.get("reward", {})
        logging_config = self.config.get("logging", {})
        
        run_name = logging_config.get("run_name") or generate_run_name("reward")
        
        return RewardConfig(
            output_dir=training.get("output_dir", "./checkpoints/reward"),
            num_train_epochs=training.get("num_train_epochs", 1),
            per_device_train_batch_size=training.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=training.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=training.get("gradient_accumulation_steps", 4),
            learning_rate=training.get("learning_rate", 1e-5),
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
            max_length=reward.get("max_length", 1024),
            report_to=logging_config.get("report_to", "none"),
            run_name=run_name,
        )
    
    def setup(self) -> "RewardTrainerWrapper":
        """Initialize the trainer with all components."""
        model_config = self.config.get("model", {})
        model_name = model_config.get("name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
        
        if isinstance(self._model, str):
            model_name = self._model
            self._model = None
        
        if self._model is None or self._tokenizer is None:
            self._model, self._tokenizer = load_model_for_training(
                model_name, self.config, model_type="reward"
            )
        
        training_args = self._create_training_args()
        
        # For reward model, ensure score head is saved
        lora_config = get_lora_config(self.config, task_type="SEQ_CLS")
        if lora_config:
            if lora_config.modules_to_save is None:
                lora_config.modules_to_save = ["score"]
            elif "score" not in lora_config.modules_to_save:
                lora_config.modules_to_save.append("score")
        
        logger.info("Initializing Reward Model trainer")
        
        self._trainer = RewardTrainer(
            model=self._model,
            args=training_args,
            train_dataset=self._train_dataset,
            eval_dataset=self._eval_dataset,
            processing_class=self._tokenizer,
            peft_config=lora_config,
        )
        
        return self
    
    def train(self) -> dict[str, Any]:
        """Run training and return metrics."""
        if self._trainer is None:
            self.setup()
        
        logger.info("Starting Reward Model training...")
        result = self._trainer.train()
        
        return {"train_loss": result.training_loss, "metrics": result.metrics}
    
    def save(self, output_dir: str | Path | None = None) -> None:
        """Save the trained model."""
        if self._trainer is None:
            raise ValueError("Trainer not initialized")
        
        save_path = output_dir or self.config.get("training", {}).get("output_dir")
        self._trainer.save_model(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def predict(self, text: str) -> float:
        """Get reward score for a given text."""
        if self._trainer is None:
            raise ValueError("Trainer not initialized")
        
        inputs = self._tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._trainer.model.device) for k, v in inputs.items()}
        
        outputs = self._trainer.model(**inputs)
        return outputs.logits[0].item()
    
    @property
    def trainer(self) -> RewardTrainer:
        if self._trainer is None:
            raise ValueError("Trainer not initialized. Call setup() first.")
        return self._trainer

