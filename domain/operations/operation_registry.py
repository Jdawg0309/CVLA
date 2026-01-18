"""
Operation registry for CVLA.
"""

from typing import Dict, List, Optional, Any, Sequence, Tuple
import numpy as np

from domain.operations.operation_spec import OperationSpec
from state.models.operation_step import OperationStep, RenderVectorHint


class OperationRegistry:
    """Registry for all operations."""

    def __init__(self):
        self._operations: Dict[str, OperationSpec] = {}

    def register(self, operation: OperationSpec) -> None:
        self._operations[operation.id] = operation

    def get(self, operation_id: str) -> Optional[OperationSpec]:
        return self._operations.get(operation_id)

    def list_all(self) -> List[OperationSpec]:
        return list(self._operations.values())


class DotOperation(OperationSpec):
    id = "dot"
    inputs = ("vector_a", "vector_b")
    outputs = ("scalar",)
    assumptions = ("vectors are same length",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> float:
        vectors = inputs.get("vectors") or ()
        if len(vectors) < 2:
            raise ValueError("Dot requires two vectors.")
        v1, v2 = vectors[0], vectors[1]
        arr1 = np.array(v1.coords, dtype="f4")
        arr2 = np.array(v2.coords, dtype="f4")
        if arr1.shape != arr2.shape:
            raise ValueError("Vector dimensions must match.")
        return float(arr1.dot(arr2))

    def steps(self, inputs: Dict[str, Any], params: Dict[str, Any], result: Any) -> Sequence[OperationStep]:
        vectors = inputs.get("vectors") or ()
        if len(vectors) < 2:
            return ()
        v1, v2 = vectors[0], vectors[1]
        a = np.array(v1.coords, dtype="f4")
        b = np.array(v2.coords, dtype="f4")
        if a.shape != b.shape:
            return ()

        steps: List[OperationStep] = []
        running = 0.0
        for idx in range(len(a)):
            product = float(a[idx] * b[idx])
            running += product
            steps.append(OperationStep(
                title=f"Multiply component {idx + 1}",
                description=f"{a[idx]:.4g} × {b[idx]:.4g} = {product:.4g}",
                values=(
                    ("a", float(a[idx])),
                    ("b", float(b[idx])),
                    ("product", product),
                    ("sum", float(running)),
                ),
                render_vectors=(
                    RenderVectorHint(
                        coords=(float(running), 0.0, 0.0),
                        color=(0.95, 0.9, 0.2),
                        label="dot sum",
                    ),
                ),
            ))

        steps.append(OperationStep(
            title="Dot product",
            description=f"Result = {float(result):.4g}",
            values=(("dot", float(result)),),
            render_vectors=(
                RenderVectorHint(
                    coords=(float(result), 0.0, 0.0),
                    color=(0.95, 0.9, 0.2),
                    label="dot result",
                ),
            ),
        ))
        return steps


class MatrixVectorMultiplyOperation(OperationSpec):
    id = "matrix_vector_multiply"
    inputs = ("matrix", "vector")
    outputs = ("vector",)
    assumptions = ("matrix columns match vector length",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Tuple[float, ...]:
        matrix = inputs.get("matrix")
        vector = inputs.get("vector")
        if matrix is None or vector is None:
            raise ValueError("Matrix-vector multiply requires matrix and vector.")
        mat_np = matrix.to_numpy()
        vec_np = vector.to_numpy()
        if mat_np.shape[1] != vec_np.shape[0]:
            raise ValueError("Matrix columns must match vector length.")
        result = mat_np @ vec_np
        return tuple(float(x) for x in result)

    def steps(self, inputs: Dict[str, Any], params: Dict[str, Any], result: Any) -> Sequence[OperationStep]:
        matrix = inputs.get("matrix")
        vector = inputs.get("vector")
        if matrix is None or vector is None:
            return ()
        mat_np = matrix.to_numpy()
        vec_np = vector.to_numpy()
        if mat_np.shape[1] != vec_np.shape[0]:
            return ()

        row_color = getattr(matrix, "color", (0.6, 0.8, 1.0))
        input_color = getattr(vector, "color", (0.8, 0.8, 0.8))
        output_color = (0.95, 0.9, 0.2)

        def to_vec3(values):
            vals = list(values)
            if len(vals) < 3:
                vals = vals + [0.0] * (3 - len(vals))
            return tuple(float(v) for v in vals[:3])

        steps: List[OperationStep] = []
        running = [0.0 for _ in range(mat_np.shape[0])]
        for idx in range(mat_np.shape[0]):
            row = mat_np[idx]
            dot_val = float(np.dot(row, vec_np))
            running[idx] = dot_val
            partial = running.copy()
            steps.append(OperationStep(
                title=f"Row {idx + 1} · input",
                description=f"Row {idx + 1} dot input = {dot_val:.4g}",
                values=(("row_dot", dot_val),),
                render_vectors=(
                    RenderVectorHint(
                        coords=to_vec3(row),
                        color=row_color,
                        label=f"row {idx + 1}",
                    ),
                    RenderVectorHint(
                        coords=to_vec3(vec_np),
                        color=input_color,
                        label="input v",
                    ),
                    RenderVectorHint(
                        coords=to_vec3(partial),
                        color=output_color,
                        label="partial output",
                    ),
                ),
            ))

        return steps


class GaussianEliminationOperation(OperationSpec):
    id = "gaussian_elimination"
    inputs = ("matrix",)
    outputs = ("steps",)
    assumptions = ("augmented matrix [A|b]",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        return inputs.get("matrix")

    def steps(self, inputs: Dict[str, Any], params: Dict[str, Any], result: Any) -> Sequence[OperationStep]:
        return (
            OperationStep(
                title="Gaussian elimination (stub)",
                description="Step trace will be implemented in the registry.",
                values=(),
            ),
        )


registry = OperationRegistry()
registry.register(DotOperation())
registry.register(MatrixVectorMultiplyOperation())
registry.register(GaussianEliminationOperation())
