"""Buffer Verifier checking agent rollout buffer isolation, resets, and memory integrity."""

from typing import Any, Dict

import torch

from marl.storage import RolloutBuffer, Transition


class BufferVerifier:
    """Verifies RolloutBuffer storage, mini-batch generation, reset functionality, and agent isolation."""

    @staticmethod
    def verify_buffer_isolation() -> Dict[str, Any]:
        """Tests rollout buffer operations for isolation and reset integrity."""
        buf1 = RolloutBuffer(capacity=5, device="cpu")
        buf2 = RolloutBuffer(capacity=5, device="cpu")

        buf1.insert(Transition(observation=torch.ones(2), action=1, reward=1.0, next_observation=torch.ones(2), terminated=False, truncated=False, value_estimate=0.5, log_prob=0.0, agent_id="agent_0"))
        buf2.insert(Transition(observation=torch.zeros(2), action=0, reward=0.0, next_observation=torch.zeros(2), terminated=False, truncated=False, value_estimate=0.0, log_prob=0.0, agent_id="agent_1"))

        passed = True
        messages = []

        if len(buf1) != 1 or len(buf2) != 1:
            passed = False
            messages.append("Buffers must be independently sized.")

        if torch.equal(buf1.transitions[0].observation, buf2.transitions[0].observation):
            passed = False
            messages.append("Buffers must store independent tensor observations.")

        buf1.reset()
        if len(buf1) != 0:
            passed = False
            messages.append("buf1.reset() must clear buffer contents.")

        return {
            "status": "PASSED" if passed else "FAILED",
            "passed": passed,
            "messages": messages,
        }
