"""Comprehensive evaluation pipeline for RLHF models."""

from typing import Any, Callable
from pathlib import Path

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset

from kolmogorov.evaluation.metrics import (
    compute_win_rate,
    compute_reward_accuracy,
    compute_kl_divergence,
)
from kolmogorov.utils.logging import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Evaluator for RLHF models with support for multiple metrics."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reference_model: PreTrainedModel | None = None,
        reward_model: PreTrainedModel | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.device = next(model.parameters()).device
    
    @torch.no_grad()
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """Generate a response for a given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
    
    @torch.no_grad()
    def compute_response_logprobs(
        self,
        prompt: str,
        response: str,
        model: PreTrainedModel | None = None,
    ) -> float:
        """Compute log probability of response given prompt."""
        model = model or self.model
        
        full_text = prompt + response
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        outputs = model(**inputs, labels=inputs["input_ids"])
        return -outputs.loss.item()
    
    def evaluate_against_reference(
        self,
        eval_dataset: Dataset,
        prompt_column: str = "prompt",
        num_samples: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate model against reference model on preference data."""
        if self.reference_model is None:
            raise ValueError("Reference model required for this evaluation")
        
        samples = eval_dataset
        if num_samples:
            samples = samples.select(range(min(num_samples, len(samples))))
        
        policy_logprobs = []
        reference_logprobs = []
        
        for example in samples:
            prompt = example[prompt_column]
            response = self.generate_response(prompt)
            
            policy_lp = self.compute_response_logprobs(prompt, response, self.model)
            ref_lp = self.compute_response_logprobs(prompt, response, self.reference_model)
            
            policy_logprobs.append(policy_lp)
            reference_logprobs.append(ref_lp)
        
        kl_metrics = compute_kl_divergence(policy_logprobs, reference_logprobs)
        
        return {
            "kl_divergence": kl_metrics,
            "num_samples": len(samples),
        }
    
    def evaluate_reward_model(
        self,
        eval_dataset: Dataset,
        chosen_column: str = "chosen",
        rejected_column: str = "rejected",
    ) -> dict[str, Any]:
        """Evaluate reward model accuracy on held-out preference data."""
        if self.reward_model is None:
            raise ValueError("Reward model required for this evaluation")
        
        predictions = []
        true_labels = []
        
        for example in eval_dataset:
            chosen_inputs = self.tokenizer(example[chosen_column], return_tensors="pt")
            rejected_inputs = self.tokenizer(example[rejected_column], return_tensors="pt")
            
            chosen_inputs = {k: v.to(self.device) for k, v in chosen_inputs.items()}
            rejected_inputs = {k: v.to(self.device) for k, v in rejected_inputs.items()}
            
            with torch.no_grad():
                chosen_reward = self.reward_model(**chosen_inputs).logits[0].item()
                rejected_reward = self.reward_model(**rejected_inputs).logits[0].item()
            
            predictions.append(chosen_reward - rejected_reward)
            true_labels.append(1)  # chosen is always preferred
        
        return compute_reward_accuracy(predictions, true_labels)
    
    def run_full_evaluation(
        self,
        eval_dataset: Dataset,
        include_generation: bool = True,
        num_samples: int = 100,
    ) -> dict[str, Any]:
        """Run comprehensive evaluation suite."""
        results = {"num_samples": min(num_samples, len(eval_dataset))}
        
        if self.reference_model:
            logger.info("Evaluating against reference model...")
            results["reference_comparison"] = self.evaluate_against_reference(
                eval_dataset, num_samples=num_samples
            )
        
        if self.reward_model:
            logger.info("Evaluating reward model accuracy...")
            results["reward_model"] = self.evaluate_reward_model(eval_dataset)
        
        logger.info(f"Evaluation complete: {results}")
        return results

