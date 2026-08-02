"""MAPPOCheckpointHandler wrapping CheckpointManager for MAPPO shared actor and centralized critic persistence."""

from typing import Any, Dict, Optional

from marl.algorithms.mappo.centralized_critic import CentralizedValueNetwork
from marl.algorithms.mappo.shared_policy import SharedPolicyManager
from marl.trainer.checkpoint_manager import CheckpointManager


class MAPPOCheckpointHandler:
    """Handles serialization, saving, and loading of MAPPO shared actor and centralized critic weights."""

    def __init__(self, checkpoint_manager: CheckpointManager) -> None:
        self.ckpt_manager: CheckpointManager = checkpoint_manager

    def save_checkpoint(
        self,
        policy_manager: SharedPolicyManager,
        critic: CentralizedValueNetwork,
        step: int,
        is_best: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Saves MAPPO actor policies and centralized critic state dicts."""
        state_dict = {"centralized_critic": critic.state_dict()}
        for agent in policy_manager.get_all_agents():
            state_dict[f"policy_{agent.agent_id}"] = agent.policy.state_dict()
            state_dict[f"optimizer_{agent.agent_id}"] = agent.optimizer.optimizer.state_dict()

        return self.ckpt_manager.save_checkpoint(
            state_dict=state_dict,
            step=step,
            is_best=is_best,
            metadata=metadata,
        )

    def load_checkpoint(
        self,
        policy_manager: SharedPolicyManager,
        critic: CentralizedValueNetwork,
        checkpoint_path: str,
    ) -> Dict[str, Any]:
        """Loads MAPPO actor policies and centralized critic state dicts from checkpoint."""
        payload = self.ckpt_manager.load_checkpoint(checkpoint_path)
        state_dict = payload.get("state_dict", {})

        if "centralized_critic" in state_dict:
            critic.load_state_dict(state_dict["centralized_critic"])

        for agent in policy_manager.get_all_agents():
            p_key = f"policy_{agent.agent_id}"
            o_key = f"optimizer_{agent.agent_id}"
            if p_key in state_dict:
                agent.policy.load_state_dict(state_dict[p_key])
            if o_key in state_dict:
                agent.optimizer.optimizer.load_state_dict(state_dict[o_key])

        return payload
