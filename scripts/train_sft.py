#!/usr/bin/env python3
"""SFT training script for Kolmogorov."""

import argparse
from pathlib import Path

from datasets import load_dataset

from kolmogorov.trainers import SFTTrainerWrapper
from kolmogorov.utils.config import load_config
from kolmogorov.utils.logging import setup_logging

logger = setup_logging()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with SFT")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path (overrides config)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="trl-lib/Capybara",
        help="Dataset name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of training samples",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    if args.model:
        config["model"]["name_or_path"] = args.model
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    
    logger.info(f"Loading dataset: {args.dataset}")
    train_dataset = load_dataset(args.dataset, split="train")
    
    if args.max_samples:
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
        logger.info(f"Using {len(train_dataset)} samples")
    
    logger.info("Initializing SFT trainer...")
    trainer = SFTTrainerWrapper(
        config=config,
        train_dataset=train_dataset,
    )
    
    trainer.setup()
    
    logger.info("Starting training...")
    results = trainer.train()
    
    logger.info(f"Training complete. Loss: {results['train_loss']:.4f}")
    
    trainer.save()
    logger.info("Model saved successfully")


if __name__ == "__main__":
    main()

