"""Comprehensive test suite for marl/trainer infrastructure framework."""

import os
import shutil
import tempfile
import torch
import numpy as np
import pytest

from marl.trainer import (
    BaseCallback,
    CallbackList,
    CheckpointManager,
    CheckpointSubConfig,
    CSVLogger,
    EnvSubConfig,
    ExperimentConfig,
    ExperimentManager,
    ModelRegistry,
    TensorBoardLogger,
    TrainingMetricsTracker,
    UnifiedLogger,
    compute_moving_average,
    format_time,
    get_device,
    seed_everything,
)


@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


def test_seed_everything() -> None:
    seed_everything(42)
    py_val1 = np.random.rand()
    torch_val1 = torch.rand(1).item()

    seed_everything(42)
    py_val2 = np.random.rand()
    torch_val2 = torch.rand(1).item()

    assert py_val1 == py_val2
    assert torch_val1 == torch_val2


def test_experiment_config_yaml_io(temp_dir) -> None:
    config = ExperimentConfig(experiment_name="test_exp", seed=123)
    yaml_path = os.path.join(temp_dir, "test_config.yaml")

    config.save_yaml(yaml_path)
    assert os.path.exists(yaml_path)

    loaded_config = ExperimentConfig.load_yaml(yaml_path)
    assert loaded_config.experiment_name == "test_exp"
    assert loaded_config.seed == 123
    assert loaded_config.env.grid_width == 20

    with pytest.raises(FileNotFoundError):
        ExperimentConfig.load_yaml(os.path.join(temp_dir, "non_existent.yaml"))


def test_experiment_manager(temp_dir) -> None:
    exp_mgr = ExperimentManager(base_dir=temp_dir, experiment_name="exp_test")
    assert os.path.exists(exp_mgr.exp_dir)
    assert os.path.exists(exp_mgr.checkpoints_dir)
    assert os.path.exists(exp_mgr.logs_dir)
    assert os.path.exists(exp_mgr.plots_dir)
    assert os.path.exists(exp_mgr.config_path)

    exp_mgr2 = ExperimentManager(base_dir=temp_dir, experiment_name="exp_test")
    assert "exp_test_002" in exp_mgr2.exp_dir


def test_checkpoint_manager_and_pruning(temp_dir) -> None:
    cfg = CheckpointSubConfig(max_checkpoints=2)
    ckpt_mgr = CheckpointManager(checkpoint_dir=temp_dir, config=cfg)

    assert ckpt_mgr.load_latest() is None
    assert ckpt_mgr.load_best() is None

    dummy_model = torch.nn.Linear(10, 2)
    state = {"model": dummy_model.state_dict()}

    for step in range(1, 5):
        ckpt_mgr.save_checkpoint(state_dict=state, step=step, is_best=(step == 4))

    latest_payload = ckpt_mgr.load_latest()
    assert latest_payload["step"] == 4

    best_payload = ckpt_mgr.load_best()
    assert best_payload["step"] == 4

    with pytest.raises(FileNotFoundError):
        ckpt_mgr.load_checkpoint(os.path.join(temp_dir, "invalid.pt"))


def test_loggers(temp_dir) -> None:
    tb = TensorBoardLogger(log_dir=temp_dir)
    tb.log_scalar("reward", 10.5, step=1)
    tb.log_dict({"loss": 0.1, "accuracy": 0.9}, step=1)
    tb.flush()
    tb.close()

    csv_file = os.path.join(temp_dir, "test.csv")
    csv_log = CSVLogger(csv_path=csv_file)
    csv_log.log_row({"step": 1, "reward": 10.5})
    csv_log.close()
    assert os.path.exists(csv_file)

    unified = UnifiedLogger(log_dir=temp_dir)
    unified.log_metrics({"mean_reward": 50.0}, step=1)
    unified.log_info("Test info message")
    unified.close()


def test_metrics_tracker() -> None:
    tracker = TrainingMetricsTracker(window_size=10)
    assert tracker.get_summary()["episode_count"] == 0.0

    tracker.record_episode(reward=100.0, length=20, success=True, collisions=1)
    tracker.record_episode(reward=50.0, length=30, success=False, collisions=2)

    summary = tracker.get_summary()
    assert summary["episode_count"] == 2.0
    assert summary["mean_reward"] == 75.0
    assert summary["success_rate"] == 0.5
    assert summary["mean_collisions"] == 1.5

    tracker.reset()
    assert tracker.get_summary()["episode_count"] == 0.0


def test_callbacks() -> None:
    called_hooks = []

    class FullTestCB(BaseCallback):
        def on_training_start(self, config):
            called_hooks.append("tr_start")

        def on_episode_start(self, episode):
            called_hooks.append("ep_start")

        def on_step(self, step, step_data):
            called_hooks.append("step")

        def on_episode_end(self, episode, metrics):
            called_hooks.append("ep_end")

        def on_checkpoint(self, step, checkpoint_path):
            called_hooks.append("ckpt")

        def on_training_end(self):
            called_hooks.append("tr_end")

    cb_list = CallbackList([FullTestCB()])
    cb_list.on_training_start(None)
    cb_list.on_episode_start(1)
    cb_list.on_step(1, {})
    cb_list.on_episode_end(1, {})
    cb_list.on_checkpoint(1, "path")
    cb_list.on_training_end()

    assert called_hooks == ["tr_start", "ep_start", "step", "ep_end", "ckpt", "tr_end"]


def test_model_registry() -> None:
    @ModelRegistry.register_policy("dummy_policy")
    class DummyPolicy:
        pass

    @ModelRegistry.register_network("dummy_net")
    class DummyNet:
        pass

    @ModelRegistry.register_optimizer("dummy_opt")
    class DummyOpt:
        pass

    @ModelRegistry.register_evaluator("dummy_eval")
    class DummyEval:
        pass

    registered = ModelRegistry.list_registered()
    assert "dummy_policy" in registered["policies"]
    assert "dummy_net" in registered["networks"]
    assert "dummy_opt" in registered["optimizers"]
    assert "dummy_eval" in registered["evaluators"]

    assert ModelRegistry.get_policy("dummy_policy") == DummyPolicy
    assert ModelRegistry.get_network("dummy_net") == DummyNet
    assert ModelRegistry.get_optimizer("dummy_opt") == DummyOpt
    assert ModelRegistry.get_evaluator("dummy_eval") == DummyEval

    with pytest.raises(KeyError):
        ModelRegistry.get_policy("unknown_policy")
    with pytest.raises(KeyError):
        ModelRegistry.get_network("unknown_net")
    with pytest.raises(KeyError):
        ModelRegistry.get_optimizer("unknown_opt")
    with pytest.raises(KeyError):
        ModelRegistry.get_evaluator("unknown_eval")


def test_trainer_utils() -> None:
    assert get_device("cpu").type == "cpu"
    assert get_device("auto").type in ["cpu", "cuda"]
    assert get_device("cuda").type in ["cpu", "cuda"]

    assert compute_moving_average([]) == 0.0
    assert compute_moving_average([10.0, 20.0, 30.0], window=2) == 25.0

    assert format_time(3665) == "01:01:05"
