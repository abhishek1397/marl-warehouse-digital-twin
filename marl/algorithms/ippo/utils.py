"""Helper utilities for Independent PPO (IPPO)."""

from typing import Dict, List

import numpy as np


def format_agent_rewards_summary(agent_rewards: Dict[str, float]) -> str:
    """Formats per-agent rewards as a clean single-line summary string."""
    items = [f"{aid}: {rew:.2f}" for aid, rew in agent_rewards.items()]
    return " | ".join(items)
