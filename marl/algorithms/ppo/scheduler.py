"""Learning rate scheduler supporting constant, linear decay, and cosine annealing schedules."""

import math


class PPOLearningRateScheduler:
    """Manages learning rate decay over training timesteps (constant, linear, cosine)."""

    def __init__(
        self,
        initial_lr: float = 0.0003,
        total_timesteps: int = 100000,
        schedule_type: str = "linear",
        min_lr: float = 0.0,
    ) -> None:
        self.initial_lr: float = initial_lr
        self.total_timesteps: int = max(total_timesteps, 1)
        self.schedule_type: str = schedule_type.lower()
        self.min_lr: float = min_lr

    def get_lr(self, current_timestep: int) -> float:
        """Returns computed learning rate for current_timestep progress.

        Args:
            current_timestep: Current step integer count.

        Returns:
            Calculated learning rate float.
        """
        progress = min(max(current_timestep / self.total_timesteps, 0.0), 1.0)

        if self.schedule_type == "constant":
            return self.initial_lr
        elif self.schedule_type == "linear":
            lr = self.initial_lr * (1.0 - progress)
            return max(lr, self.min_lr)
        elif self.schedule_type == "cosine":
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
            return lr
        else:
            return self.initial_lr
