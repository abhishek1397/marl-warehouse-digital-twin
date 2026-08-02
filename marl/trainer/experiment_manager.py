"""ExperimentManager class organizing experiment folder trees and tracking runs."""

import os
from typing import Optional

from marl.trainer.config import ExperimentConfig


class ExperimentManager:
    """Creates and manages experiment run folder structures (runs/experiment_001/...)."""

    def __init__(
        self,
        base_dir: str = "runs",
        experiment_name: str = "experiment",
        config: Optional[ExperimentConfig] = None,
    ) -> None:
        self.base_dir: str = os.path.abspath(base_dir)
        self.experiment_name: str = experiment_name
        self.config: ExperimentConfig = config or ExperimentConfig()

        os.makedirs(self.base_dir, exist_ok=True)

        # Generate unique experiment folder path (e.g., runs/experiment_001)
        self.exp_dir: str = self._create_experiment_dir()

        # Create subdirectories
        self.checkpoints_dir: str = os.path.join(self.exp_dir, "checkpoints")
        self.logs_dir: str = os.path.join(self.exp_dir, "logs")
        self.plots_dir: str = os.path.join(self.exp_dir, "plots")

        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Save configuration YAML in experiment directory
        self.config_path: str = os.path.join(self.exp_dir, "config.yaml")
        self.config.save_yaml(self.config_path)

    def _create_experiment_dir(self) -> str:
        """Finds next available index and creates runs/experiment_XXX directory."""
        index = 1
        while True:
            folder_name = f"{self.experiment_name}_{index:03d}"
            target_dir = os.path.join(self.base_dir, folder_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                return target_dir
            index += 1
