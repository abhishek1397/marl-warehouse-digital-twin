# MARL Training Infrastructure Framework Documentation (`marl/trainer/`)

This document provides technical documentation for `marl/trainer/`, the algorithm-agnostic training infrastructure framework supporting reproducible MARL experimentation.

---

## 1. Directory Layout & Experiment Lifecycle

`ExperimentManager` automatically creates and organizes experiment runs under `runs/`:

```
runs/
└── experiment_001/
    ├── checkpoints/
    │   ├── checkpoint_000100.pt
    │   ├── latest.pt
    │   └── best_model.pt
    ├── logs/
    │   ├── events.out.tfevents...  (TensorBoard)
    │   ├── metrics.csv              (CSV Log)
    │   └── training.log             (Console & File Log)
    ├── plots/                       (Exported Evaluation Charts)
    └── config.yaml                  (Active Configuration Backup)
```

---

## 2. Configuration System (`configs/training_config.yaml`)

Configuration is managed via hierarchical dataclasses in `marl.trainer.config` and loaded from YAML:

```python
from marl.trainer import ExperimentConfig

# Load configuration from YAML
config = ExperimentConfig.load_yaml("configs/training_config.yaml")

# Modify settings programmatically
config.seed = 123
config.training.learning_rate = 1e-4

# Save back to YAML
config.save_yaml("runs/experiment_001/config.yaml")
```

---

## 3. PyTorch Checkpoint Management

`CheckpointManager` provides PyTorch state dict saving, loading, best-model tracking, and pruning:

```python
from marl.trainer import CheckpointManager

ckpt_mgr = CheckpointManager(checkpoint_dir="runs/experiment_001/checkpoints")

# Save step checkpoint (updates latest.pt and best_model.pt if is_best=True)
ckpt_mgr.save_checkpoint(
    state_dict={"policy": policy.state_dict(), "optimizer": optimizer.state_dict()},
    step=5000,
    is_best=True,
    metadata={"mean_reward": 142.5},
)

# Resume / Load latest or best model
latest_payload = ckpt_mgr.load_latest()
best_payload = ckpt_mgr.load_best()
```

---

## 4. Unified Multi-Logging System

`UnifiedLogger` routes metrics simultaneously to TensorBoard, CSV files, File logs, and stdout Console:

```python
from marl.trainer import UnifiedLogger

logger = UnifiedLogger(log_dir="runs/experiment_001/logs")

# Log metrics dictionary
logger.log_metrics({
    "mean_reward": 45.2,
    "success_rate": 0.85,
    "mean_collisions": 0.1,
}, step=100)

logger.close()
```

---

## 5. Model Registry for Future MARL Algorithms

`ModelRegistry` enables future algorithms (MAPPO, QMIX, IPPO, DQN) to plug in via decorators without modifying core infrastructure:

```python
from marl.trainer import ModelRegistry

# Register a custom policy class
@ModelRegistry.register_policy("mappo")
class MAMPPOPolicy:
    def __init__(self, config):
        pass

# Retrieve registered policy class
policy_cls = ModelRegistry.get_policy("mappo")
instance = policy_cls(config)
```

---

## 6. Callback Lifecycle Hooks

Callbacks hook into the training loop via `BaseCallback` and `CallbackList`:

```python
from marl.trainer import BaseCallback, CallbackList

class CustomLoggingCallback(BaseCallback):
    def on_episode_end(self, episode: int, metrics: dict) -> None:
        print(f"Episode {episode} finished with mean return {metrics.get('mean_reward'):.2f}")

callbacks = CallbackList([CustomLoggingCallback()])
callbacks.on_episode_end(1, {"mean_reward": 88.4})
```

---

## 7. Multi-Framework Seeding & Reproducibility

`seed_everything(seed)` seeds Python, NumPy, PyTorch (CPU and CUDA cuDNN deterministic flags), Gymnasium, PettingZoo, and the simulator:

```python
from marl.trainer import seed_everything

seed_everything(seed=42, torch_deterministic=True)
```
