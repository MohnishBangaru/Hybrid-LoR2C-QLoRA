"""Deterministic seeding of every random source used during training."""

import random

import torch


class TorchSeeder:
    """Seeds Python, torch CPU and torch CUDA generators."""

    def seed(self, *, value: int) -> None:
        """Apply `value` to all generators."""
        random.seed(value)
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
