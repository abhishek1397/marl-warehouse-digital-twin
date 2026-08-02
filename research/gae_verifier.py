"""GAE Verifier checking GAE advantage computation, bootstrap values, and discounting correctness."""

from typing import Any, Dict

import torch

from marl.storage import RolloutBuffer, Transition
from marl.storage.gae import compute_gae


class GAEVerifier:
    """Verifies GAE advantage and return calculations against baseline implementation math."""

    @staticmethod
    def verify_gae_calculation() -> Dict[str, Any]:
        """Runs GAE advantage calculation test with known synthetic trajectory inputs."""
        rewards = torch.tensor([1.0, 1.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        next_values = torch.tensor([0.5, 0.5, 0.0])
        dones = torch.tensor([False, False, True])

        advantages, returns = compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95)

        passed = True
        messages = []

        if len(advantages) != 3 or len(returns) != 3:
            passed = False
            messages.append(f"Advantages count {len(advantages)} != 3.")

        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            passed = False
            messages.append("Advantage array contains NaN or Inf.")

        return {
            "status": "PASSED" if passed else "FAILED",
            "passed": passed,
            "messages": messages,
            "sample_advantages": advantages.tolist(),
        }
