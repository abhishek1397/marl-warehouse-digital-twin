"""TensorBoardLogger wrapping PyTorch SummaryWriter for metric visualization."""

import os
from typing import Dict

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """Wrapper for PyTorch SummaryWriter recording scalar metrics to TensorBoard logs."""

    def __init__(self, log_dir: str) -> None:
        self.log_dir: str = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer: SummaryWriter = SummaryWriter(log_dir=self.log_dir)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Logs a single scalar metric value."""
        self.writer.add_scalar(tag=tag, scalar_value=float(value), global_step=step)

    def log_dict(self, metrics: Dict[str, float], step: int) -> None:
        """Logs a dictionary of scalar metrics."""
        for tag, val in metrics.items():
            if isinstance(val, (int, float)):
                self.writer.add_scalar(tag=tag, scalar_value=float(val), global_step=step)

    def flush(self) -> None:
        """Flushes buffered writer content."""
        self.writer.flush()

    def close(self) -> None:
        """Closes SummaryWriter instance."""
        self.writer.close()
