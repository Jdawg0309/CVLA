"""
Pipeline and educational step action definitions.
"""

from dataclasses import dataclass
from typing import Tuple

from state.models.educational_step import EducationalStep


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


@dataclass(frozen=True)
class SetPipeline:
    """Replace the pipeline steps and reset the index."""
    steps: Tuple[EducationalStep, ...]
    index: int = 0
