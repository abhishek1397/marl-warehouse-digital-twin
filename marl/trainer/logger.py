"""UnifiedLogger orchestrating TensorBoard, CSV, stdout Console, and File logging."""

import logging
import os
from typing import Any, Dict, Optional

from marl.trainer.config import LoggingSubConfig
from marl.trainer.csv_logger import CSVLogger
from marl.trainer.tensorboard_logger import TensorBoardLogger


class UnifiedLogger:
    """Orchestrates TensorBoard, CSV, Console, and File logging under a single interface."""

    def __init__(self, log_dir: str, config: Optional[LoggingSubConfig] = None) -> None:
        self.log_dir: str = os.path.abspath(log_dir)
        self.config: LoggingSubConfig = config or LoggingSubConfig()
        os.makedirs(self.log_dir, exist_ok=True)

        self.tb_logger: Optional[TensorBoardLogger] = None
        if self.config.tensorboard:
            self.tb_logger = TensorBoardLogger(log_dir=self.log_dir)

        self.csv_logger: Optional[CSVLogger] = None
        if self.config.csv:
            csv_path = os.path.join(self.log_dir, "metrics.csv")
            self.csv_logger = CSVLogger(csv_path=csv_path)

        # File and Console Python Logger
        self._python_logger = logging.getLogger(f"MARL_Logger_{id(self)}")
        self._python_logger.setLevel(logging.INFO)
        self._python_logger.handlers.clear()

        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

        if self.config.file:
            log_file_path = os.path.join(self.log_dir, "training.log")
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self._python_logger.addHandler(file_handler)

        if self.config.stdout:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self._python_logger.addHandler(stream_handler)

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Logs metrics dictionary to TensorBoard, CSV, and Console."""
        row_data = {"step": step}
        numeric_metrics: Dict[str, float] = {}

        for k, v in metrics.items():
            if isinstance(v, (int, float, bool)):
                val_float = float(v)
                row_data[k] = val_float
                numeric_metrics[k] = val_float
            else:
                row_data[k] = str(v)

        if self.tb_logger:
            self.tb_logger.log_dict(numeric_metrics, step)

        if self.csv_logger:
            self.csv_logger.log_row(row_data)

        if self.config.stdout and step % self.config.log_interval == 0:
            summary = " | ".join(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in numeric_metrics.items())
            self._python_logger.info(f"Step {step} -> {summary}")

    def log_info(self, message: str) -> None:
        """Logs an informational string message."""
        self._python_logger.info(message)

    def close(self) -> None:
        """Closes all logging resources."""
        if self.tb_logger:
            self.tb_logger.close()
        if self.csv_logger:
            self.csv_logger.close()
