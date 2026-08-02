"""Configuration dataclass for Dynamic Action Masking (DAM)."""

from dataclasses import dataclass


@dataclass
class ActionMaskConfig:
    """Configuration settings for dynamic action masking."""

    enable_action_masking: bool = True
    strict_masking: bool = True
    mask_invalid_moves: bool = True
    mask_invalid_pick: bool = True
    mask_invalid_drop: bool = True
    mask_invalid_charge: bool = True
