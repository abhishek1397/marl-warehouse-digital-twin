"""Reward shaping utility functions for PBRS calculations."""


def calculate_shaping_reward(
    phi_current: float,
    phi_next: float,
    gamma: float = 0.99,
    scale: float = 1.0,
) -> float:
    """Computes Ng et al. (1999) shaping reward F(s, a, s') = scale * (gamma * Phi(s') - Phi(s))."""
    return float(scale * (gamma * phi_next - phi_current))


def calculate_goal_progress(dist_current: float, dist_next: float) -> float:
    """Computes distance improvement delta (positive if agent moved closer to goal)."""
    return float(dist_current - dist_next)
