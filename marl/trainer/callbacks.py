"""Callback interface and CallbackList composite for training lifecycle hooks."""

from abc import ABC
from typing import Any, Dict, List, Optional


class BaseCallback(ABC):
    """Abstract base class for training callbacks."""

    def on_training_start(self, config: Any) -> None:
        """Called when training starts."""
        pass

    def on_episode_start(self, episode: int) -> None:
        """Called at the beginning of each episode."""
        pass

    def on_step(self, step: int, step_data: Dict[str, Any]) -> None:
        """Called after each environment step transition."""
        pass

    def on_episode_end(self, episode: int, metrics: Dict[str, Any]) -> None:
        """Called when an episode finishes."""
        pass

    def on_checkpoint(self, step: int, checkpoint_path: str) -> None:
        """Called after a checkpoint is saved."""
        pass

    def on_training_end(self) -> None:
        """Called when training completes."""
        pass


class CallbackList(BaseCallback):
    """Composite container holding multiple callbacks executed sequentially."""

    def __init__(self, callbacks: Optional[List[BaseCallback]] = None) -> None:
        self.callbacks: List[BaseCallback] = callbacks or []

    def add_callback(self, callback: BaseCallback) -> None:
        """Adds a callback to the container."""
        self.callbacks.append(callback)

    def on_training_start(self, config: Any) -> None:
        for cb in self.callbacks:
            cb.on_training_start(config)

    def on_episode_start(self, episode: int) -> None:
        for cb in self.callbacks:
            cb.on_episode_start(episode)

    def on_step(self, step: int, step_data: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_step(step, step_data)

    def on_episode_end(self, episode: int, metrics: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_episode_end(episode, metrics)

    def on_checkpoint(self, step: int, checkpoint_path: str) -> None:
        for cb in self.callbacks:
            cb.on_checkpoint(step, checkpoint_path)

    def on_training_end(self) -> None:
        for cb in self.callbacks:
            cb.on_training_end()
