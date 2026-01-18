"""
Selection state for vector space objects.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SelectionState:
    active_vector_space: Optional[str] = None
    selected_matrix: Optional[str] = None
    selected_vectors: Tuple[str, ...] = ()
