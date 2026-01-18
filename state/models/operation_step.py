"""
Operation step model for step-by-step execution.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RenderVectorHint:
    """Vector hint to render for a given operation step."""
    coords: Tuple[float, float, float]
    color: Tuple[float, float, float]
    label: str = ""
    visible: bool = True


@dataclass(frozen=True)
class OperationStep:
    """Represents a single step in an operation."""
    title: str
    description: str = ""
    values: Tuple[Tuple[str, float], ...] = ()
    render_vectors: Tuple[RenderVectorHint, ...] = ()
