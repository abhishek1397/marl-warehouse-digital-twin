"""Trainer utility helper functions for device detection and formatting."""

from typing import List, Sequence

import torch


def get_device(device_str: str = "cpu") -> torch.device:
    """Selects and returns PyTorch device (CPU/CUDA).

    Args:
        device_str: Preferred device string ('cpu', 'cuda', 'auto').

    Returns:
        torch.device instance.
    """
    if device_str.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str.lower() == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def compute_moving_average(values: Sequence[float], window: int = 100) -> float:
    """Computes moving average over trailing window elements."""
    if not values:
        return 0.0
    recent = list(values)[-window:]
    return float(sum(recent) / len(recent))


def format_time(seconds: float) -> str:
    """Formats duration seconds into HH:MM:SS string format."""
    total_sec = int(seconds)
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
