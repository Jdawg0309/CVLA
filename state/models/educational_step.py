"""
Educational pipeline step model.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
from uuid import uuid4
import numpy as np

from state.models.image_model import ImageData


@dataclass(frozen=True)
class EducationalStep:
    """
    Represents one step in an educational pipeline.
    """
    id: str
    title: str
    explanation: str
    operation: str
    input_data: Optional[ImageData] = None
    output_data: Optional[ImageData] = None
    kernel_name: Optional[str] = None
    kernel_values: Optional[Tuple[Tuple[float, ...], ...]] = None
    transform_matrix: Optional[Tuple[Tuple[float, ...], ...]] = None
    kernel_position: Optional[Tuple[int, int]] = None

    @staticmethod
    def create(title: str, explanation: str, operation: str,
               input_data: Optional[ImageData] = None,
               output_data: Optional[ImageData] = None,
               kernel_name: Optional[str] = None,
               kernel_values: Optional[np.ndarray] = None,
               transform_matrix: Optional[np.ndarray] = None) -> 'EducationalStep':
        """Factory method."""
        kv = None
        if kernel_values is not None:
            kv = tuple(tuple(row) for row in kernel_values)
        tm = None
        if transform_matrix is not None:
            tm = tuple(tuple(row) for row in transform_matrix)

        return EducationalStep(
            id=str(uuid4()),
            title=title,
            explanation=explanation,
            operation=operation,
            input_data=input_data,
            output_data=output_data,
            kernel_name=kernel_name,
            kernel_values=kv,
            transform_matrix=tm
        )
