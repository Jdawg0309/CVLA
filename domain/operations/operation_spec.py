"""
OperationSpec interface for CVLA operations.

All math operations must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Sequence

from state.models.operation_step import OperationStep


class OperationSpec(ABC):
    """Interface for a single math operation."""

    id: str = ""
    inputs: Tuple[str, ...] = ()
    outputs: Tuple[str, ...] = ()
    assumptions: Tuple[str, ...] = ()

    @abstractmethod
    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        """Compute the final result for the operation."""

    @abstractmethod
    def steps(self, inputs: Dict[str, Any], params: Dict[str, Any], result: Any) -> Sequence[OperationStep]:
        """Return step-by-step trace for the operation."""

    def render_hints(self, inputs: Dict[str, Any], params: Dict[str, Any], result: Any) -> Dict[str, Any]:
        """Optional render hints for the operation."""
        return {}
