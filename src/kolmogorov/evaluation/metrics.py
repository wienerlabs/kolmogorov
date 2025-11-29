"""Evaluation metrics for RLHF models."""

from typing import Any
import numpy as np
from scipy import stats


def compute_win_rate(
    model_responses: list[str],
    baseline_responses: list[str],
    preferences: list[int],  # 1 if model wins, 0 if baseline wins
) -> dict[str, float]:
    """Compute win rate of model against baseline.
    
    Args:
        model_responses: Responses from the trained model
        baseline_responses: Responses from baseline/reference model
        preferences: Binary labels (1 = model preferred, 0 = baseline preferred)
    
    Returns:
        Dictionary with win_rate, confidence_interval, and sample_size
    """
    wins = sum(preferences)
    total = len(preferences)
    
    if total == 0:
        return {"win_rate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "sample_size": 0}
    
    win_rate = wins / total
    
    # Wilson score interval for confidence
    z = 1.96  # 95% confidence
    denominator = 1 + z**2 / total
    center = (win_rate + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt((win_rate * (1 - win_rate) + z**2 / (4 * total)) / total) / denominator
    
    return {
        "win_rate": win_rate,
        "ci_lower": max(0, center - spread),
        "ci_upper": min(1, center + spread),
        "sample_size": total,
    }


def compute_reward_accuracy(
    predicted_rewards: list[float],
    true_preferences: list[int],  # 1 if first is better, 0 if second is better
    threshold: float = 0.0,
) -> dict[str, float]:
    """Compute accuracy of reward model predictions.
    
    Args:
        predicted_rewards: Reward differences (reward_a - reward_b)
        true_preferences: Binary labels (1 = first preferred, 0 = second preferred)
        threshold: Threshold for considering predictions equal
    
    Returns:
        Dictionary with accuracy, precision, recall, f1
    """
    correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    
    for pred, true in zip(predicted_rewards, true_preferences):
        pred_label = 1 if pred > threshold else 0
        
        if pred_label == true:
            correct += 1
        
        if pred_label == 1 and true == 1:
            true_positive += 1
        elif pred_label == 1 and true == 0:
            false_positive += 1
        elif pred_label == 0 and true == 1:
            false_negative += 1
    
    total = len(predicted_rewards)
    accuracy = correct / total if total > 0 else 0
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sample_size": total,
    }


def compute_kl_divergence(
    policy_logprobs: list[float],
    reference_logprobs: list[float],
) -> dict[str, float]:
    """Compute KL divergence between policy and reference model.
    
    Args:
        policy_logprobs: Log probabilities from trained policy
        reference_logprobs: Log probabilities from reference model
    
    Returns:
        Dictionary with mean, std, and max KL divergence
    """
    kl_divs = [p - r for p, r in zip(policy_logprobs, reference_logprobs)]
    
    return {
        "kl_mean": float(np.mean(kl_divs)),
        "kl_std": float(np.std(kl_divs)),
        "kl_max": float(np.max(kl_divs)),
        "kl_min": float(np.min(kl_divs)),
    }


def compute_ranking_correlation(
    predicted_rankings: list[list[int]],
    true_rankings: list[list[int]],
) -> dict[str, float]:
    """Compute Kendall's tau correlation between predicted and true rankings.
    
    Args:
        predicted_rankings: List of predicted ranking orders
        true_rankings: List of true ranking orders
    
    Returns:
        Dictionary with mean tau, p-value, and sample statistics
    """
    taus = []
    pvalues = []
    
    for pred, true in zip(predicted_rankings, true_rankings):
        if len(pred) < 2:
            continue
        tau, pvalue = stats.kendalltau(pred, true)
        if not np.isnan(tau):
            taus.append(tau)
            pvalues.append(pvalue)
    
    if not taus:
        return {"tau_mean": 0.0, "tau_std": 0.0, "pvalue_mean": 1.0}
    
    return {
        "tau_mean": float(np.mean(taus)),
        "tau_std": float(np.std(taus)),
        "pvalue_mean": float(np.mean(pvalues)),
        "sample_size": len(taus),
    }

