#!/usr/bin/env python3
"""
4D fMRI Model Training LR Scheduler

This module provides a single scheduler:
  - WarmupCosineSchedule: linear warmup followed by cosine decay

Usage:
    from lr_scheduler import get_scheduler, WarmupCosineSchedule

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = get_scheduler(
        optimizer=optimizer,
        warmup_steps=500,
        total_steps=10000,
        cycles=0.5,
        restart_interval=-1
    )

    for step in range(total_steps):
        train_step(...)
        scheduler.step()
"""
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_scheduler(
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        cycles: float = 0.5,
        restart_interval: int = -1
) -> LambdaLR:
    """
    Create a WarmupCosineSchedule for 4D fMRI training.

    Args:
        optimizer: torch Optimizer
        warmup_steps: number of steps for linear warmup
        total_steps: total number of training steps
        cycles: number of full cosine oscillations (default 0.5)
        restart_interval: if >0, wrap step counter every restart_interval
    Returns:
        WarmupCosineSchedule instance
    """
    return WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        t_total=total_steps,
        cycles=cycles,
        restart_interval=restart_interval
    )

class WarmupCosineSchedule(LambdaLR):
    """
    Linear warmup followed by cosine decay.
    Designed specifically for 4D fMRI model training.
    """

    def __init__(
            self,
            optimizer: Optimizer,
            warmup_steps: int,
            t_total: int,
            cycles: float = 0.5,
            last_epoch: int = -1,
            restart_interval: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.restart_interval = restart_interval
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step: int) -> float:
        # optionally wrap around for manual restarts
        if self.restart_interval > 0:
            step = step % self.restart_interval
        # linear warmup phase
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        # cosine decay phase
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * 2.0 * self.cycles * progress))
        )
