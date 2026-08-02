"""MAPPOBatchBuilder building mini-batches for joint actor-critic updates."""

from typing import Dict, List

import torch

from marl.storage.batch import Batch


class MAPPOBatchBuilder:
    """Helper module constructing joint mini-batch tensors for MAPPO CTDE updates."""

    @staticmethod
    def build_joint_batch(agent_batches: List[Batch]) -> Batch:
        """Concatenates individual agent mini-batches into a joint training batch."""
        if not agent_batches:
            raise ValueError("agent_batches list must not be empty.")

        obs_list = [b.observations for b in agent_batches]
        act_list = [b.actions for b in agent_batches]
        old_lp_list = [b.old_log_probs for b in agent_batches]
        adv_list = [b.advantages for b in agent_batches]
        ret_list = [b.returns for b in agent_batches]
        val_list = [b.values for b in agent_batches]

        # Handle Dict vs Tensor observations
        if isinstance(obs_list[0], dict):
            joint_obs = {}
            for k in obs_list[0].keys():
                joint_obs[k] = torch.cat([b.observations[k] for b in agent_batches], dim=0)
        else:
            joint_obs = torch.cat(obs_list, dim=0)

        joint_actions = torch.cat(act_list, dim=0)
        joint_old_lp = torch.cat(old_lp_list, dim=0)
        joint_adv = torch.cat(adv_list, dim=0)
        joint_ret = torch.cat(ret_list, dim=0)
        joint_val = torch.cat(val_list, dim=0)

        return Batch(
            observations=joint_obs,
            actions=joint_actions,
            advantages=joint_adv,
            returns=joint_ret,
            values=joint_val,
            old_log_probs=joint_old_lp,
        )
