"""
Immutable Data Models for CVLA.
"""

from state.models.vector_model import VectorData
from state.models.matrix_model import MatrixData
from state.models.image_model import ImageData
from state.models.educational_step import EducationalStep
from state.models.pipeline_models import PipelineOp, MicroOp
from state.models.tensor_model import TensorData, TensorDType
from state.models.operation_record import OperationRecord
from state.models.operation_step import OperationStep, RenderVectorHint
from state.models.operations_state import OperationsState
from state.models.vector_space import VectorSpace, VectorSpaceRelationship, VectorSpaceGraph
from state.models.selection_state import SelectionState
from state.models.render_state import RenderState
from state.models.cache_state import CacheState
from state.models.tensor_compat import (
    vector_to_tensor, tensor_to_vector,
    matrix_to_tensor, tensor_to_matrix,
    image_to_tensor, tensor_to_image,
    vectors_to_tensors, matrices_to_tensors,
    tensors_to_vectors, tensors_to_matrices,
    filter_tensors_by_type,
)

__all__ = [
    # Legacy models
    'VectorData',
    'MatrixData',
    'ImageData',
    'EducationalStep',
    'PipelineOp',
    'MicroOp',
    # New unified tensor model
    'TensorData',
    'TensorDType',
    'OperationRecord',
    'OperationStep',
    'RenderVectorHint',
    'OperationsState',
    'VectorSpace',
    'VectorSpaceRelationship',
    'VectorSpaceGraph',
    'SelectionState',
    'RenderState',
    'CacheState',
    # Compatibility functions
    'vector_to_tensor',
    'tensor_to_vector',
    'matrix_to_tensor',
    'tensor_to_matrix',
    'image_to_tensor',
    'tensor_to_image',
    'vectors_to_tensors',
    'matrices_to_tensors',
    'tensors_to_vectors',
    'tensors_to_matrices',
    'filter_tensors_by_type',
]
