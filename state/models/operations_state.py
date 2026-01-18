"""
Operation execution state.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from state.models.operation_step import OperationStep


@dataclass(frozen=True)
class OperationsState:
    current_operation: Optional[str] = None
    steps: Tuple[OperationStep, ...] = ()
    step_index: int = 0
