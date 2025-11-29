"""
Kolmogorov: RLHF Implementation for LLM Preference Learning

A reinforcement learning system that enables LLMs to learn continuously
from real-world human interactions using preference-based feedback mechanisms.
"""

__version__ = "0.1.0"
__author__ = "Kolmogorov Team"

from kolmogorov.trainers import DPOTrainerWrapper, RewardTrainerWrapper, SFTTrainerWrapper
from kolmogorov.data import PreferenceDataset, PreferenceCollector
from kolmogorov.evaluation import Evaluator

__all__ = [
    "DPOTrainerWrapper",
    "RewardTrainerWrapper",
    "SFTTrainerWrapper",
    "PreferenceDataset",
    "PreferenceCollector",
    "Evaluator",
]

