"""ActionMaskValidator module verifying invariants on action validity masks."""

from marl.action_masking.action_mask import ActionMask


class ActionMaskValidator:
    """Validates structural invariants of generated action masks."""

    @staticmethod
    def validate_mask(action_mask: ActionMask) -> bool:
        """Verifies that the action mask satisfies validity invariants.

        Invariants:
            1. At least one action must be valid (num_valid >= 1).
            2. Action 4 (Wait) must always remain True.

        Raises:
            ValueError: If an invariant is violated.
        """
        if action_mask.num_valid < 1:
            raise ValueError("Action mask error: No valid actions available in current state!")

        if not action_mask.mask_array[4]:
            raise ValueError("Action mask error: Action 4 (Wait) must always remain valid!")

        return True
