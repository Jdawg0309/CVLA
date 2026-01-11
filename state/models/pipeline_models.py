"""
Pipeline and micro-operation models.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
from uuid import uuid4


@dataclass(frozen=True)
class PipelineOp:
    """Immutable pipeline operation definition."""
    id: str
    name: str
    op_type: str
    kernel_name: Optional[str] = None

    @staticmethod
    def create(name: str, op_type: str, kernel_name: Optional[str] = None) -> 'PipelineOp':
        return PipelineOp(
            id=str(uuid4()),
            name=name,
            op_type=op_type,
            kernel_name=kernel_name,
        )


@dataclass(frozen=True)
class MicroOp:
    """Immutable micro-operation snapshot."""
    op_type: str
    stage: str
    step_index: int
    step_total: int
    input_coord: Optional[Tuple[int, int]] = None
    output_coord: Optional[Tuple[int, int]] = None
    input_value: Optional[float] = None
    output_value: Optional[float] = None
    kernel_coord: Optional[Tuple[int, int]] = None
    kernel_value: Optional[float] = None
    product: Optional[float] = None
    partial_sum: Optional[float] = None
    note: str = ""
