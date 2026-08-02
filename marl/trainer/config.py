"""Hierarchical configuration dataclasses and YAML loader/saver for MARL training."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class EnvSubConfig:
    """Environment configuration options."""

    grid_width: int = 20
    grid_height: int = 20
    num_robots: int = 3
    num_tasks: int = 10
    max_episode_steps: int = 200
    observation_mode: str = "local"
    reward_mode: str = "individual"
    comm_mode: str = "none"


@dataclass
class TrainingSubConfig:
    """Training hyperparameter options."""

    total_timesteps: int = 100000
    learning_rate: float = 0.0003
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 10000


@dataclass
class EvalSubConfig:
    """Evaluation evaluation options."""

    eval_interval: int = 5000
    eval_episodes: int = 10


@dataclass
class LoggingSubConfig:
    """Logging destination options."""

    tensorboard: bool = True
    csv: bool = True
    stdout: bool = True
    file: bool = True
    log_interval: int = 100


@dataclass
class CheckpointSubConfig:
    """Model checkpoint options."""

    save_interval: int = 5000
    save_best: bool = True
    max_checkpoints: int = 5


@dataclass
class NetworkSubConfig:
    """Neural network architecture configuration options."""

    hidden_dim: int = 128
    num_layers: int = 2
    activation: str = "relu"


@dataclass
class OptimizerSubConfig:
    """Optimizer configuration options."""

    name: str = "adam"
    lr: float = 0.0003
    weight_decay: float = 0.0


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration dataclass."""

    experiment_name: str = "default_marl_exp"
    seed: int = 42
    device: str = "cpu"

    env: EnvSubConfig = field(default_factory=EnvSubConfig)
    training: TrainingSubConfig = field(default_factory=TrainingSubConfig)
    eval: EvalSubConfig = field(default_factory=EvalSubConfig)
    logging: LoggingSubConfig = field(default_factory=LoggingSubConfig)
    checkpoint: CheckpointSubConfig = field(default_factory=CheckpointSubConfig)
    network: NetworkSubConfig = field(default_factory=NetworkSubConfig)
    optimizer: OptimizerSubConfig = field(default_factory=OptimizerSubConfig)

    @classmethod
    def load_yaml(cls, path: str) -> "ExperimentConfig":
        """Loads configuration from a YAML file.

        Args:
            path: Absolute or relative path to YAML file.

        Returns:
            ExperimentConfig instance populated from file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration YAML file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        env_cfg = EnvSubConfig(**data.get("env", {}))
        training_cfg = TrainingSubConfig(**data.get("training", {}))
        eval_cfg = EvalSubConfig(**data.get("eval", {}))
        logging_cfg = LoggingSubConfig(**data.get("logging", {}))
        checkpoint_cfg = CheckpointSubConfig(**data.get("checkpoint", {}))
        network_cfg = NetworkSubConfig(**data.get("network", {}))
        optimizer_cfg = OptimizerSubConfig(**data.get("optimizer", {}))

        return cls(
            experiment_name=data.get("experiment_name", "default_marl_exp"),
            seed=data.get("seed", 42),
            device=data.get("device", "cpu"),
            env=env_cfg,
            training=training_cfg,
            eval=eval_cfg,
            logging=logging_cfg,
            checkpoint=checkpoint_cfg,
            network=network_cfg,
            optimizer=optimizer_cfg,
        )

    def save_yaml(self, path: str) -> None:
        """Saves current configuration to a YAML file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False)
