"""
Operation record model for tracking operation history.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
from uuid import uuid4


@dataclass(frozen=True)
class OperationRecord:
    """
    Immutable record of an operation performed on tensors.

    Used to track operation history and enable operation replay.
    """
    id: str
    operation_name: str
    parameters: Tuple[Tuple[str, str], ...]  # Key-value pairs as tuples
    target_ids: Tuple[str, ...]  # IDs of tensors operated on
    result_ids: Tuple[str, ...]  # IDs of resulting tensors (if new tensors created)
    timestamp: float  # Unix timestamp
    description: str  # Human-readable description

    @staticmethod
    def create(
        operation_name: str,
        parameters: Tuple[Tuple[str, str], ...],
        target_ids: Tuple[str, ...],
        result_ids: Tuple[str, ...] = (),
        description: str = ""
    ) -> 'OperationRecord':
        """Factory method that generates UUID and timestamp."""
        import time
        return OperationRecord(
            id=str(uuid4()),
            operation_name=operation_name,
            parameters=parameters,
            target_ids=target_ids,
            result_ids=result_ids,
            timestamp=time.time(),
            description=description or _generate_description(operation_name, parameters)
        )


def _generate_description(operation_name: str, parameters: Tuple[Tuple[str, str], ...]) -> str:
    """Generate a human-readable description for an operation."""
    param_str = ", ".join(f"{k}={v}" for k, v in parameters) if parameters else ""
    if param_str:
        return f"{operation_name}({param_str})"
    return operation_name
