"""
Pipeline and educational step action definitions.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StepForward:
    """Move to the next step in the educational pipeline."""
    pass


@dataclass(frozen=True)
class StepBackward:
    """Move to the previous step in the educational pipeline."""
    pass


@dataclass(frozen=True)
class JumpToStep:
    """Jump to a specific step index."""
    index: int


@dataclass(frozen=True)
class ResetPipeline:
    """Clear all pipeline steps and reset to initial state."""
    pass
