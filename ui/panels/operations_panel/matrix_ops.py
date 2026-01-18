"""
Matrix operations widget for the operations panel.

Provides UI for matrix-specific operations.
"""

import imgui
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.app_state import AppState
    from state.models.tensor_model import TensorData

from state.actions.tensor_actions import ApplyOperation, DuplicateTensor
from state.actions.matrix_actions import ToggleMatrixPlot
from state.selectors import get_matrices, get_vectors


class MatrixOpsWidget:
    """Widget for matrix operations."""

    UNARY_OPS = [
        ("transpose", "Transpose", "Swap rows and columns"),
        ("inverse", "Inverse", "Compute matrix inverse"),
        ("determinant", "Determinant", "Compute determinant"),
        ("trace", "Trace", "Sum of diagonal elements"),
    ]

    DECOMPOSITION_OPS = [
        ("eigen", "Eigendecomp", "Eigenvalue decomposition"),
        ("svd", "SVD", "Singular value decomposition"),
        ("qr", "QR", "QR decomposition"),
        ("lu", "LU", "LU decomposition"),
    ]

    def __init__(self):
        self._scale_factor = 1.0
        self._selected_other_idx = 0
        self._selected_vector_idx = 0

    def render(self, tensor: "TensorData", state: "AppState", dispatch, width: float):
        """Render matrix operations UI."""
        if tensor is None or not tensor.is_matrix:
            return

        imgui.text("MATRIX OPERATIONS")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Visualization toggles
        imgui.text("Visualization:")
        imgui.spacing()
        plot_enabled = getattr(state, "matrix_plot_enabled", False)
        changed, plot_enabled = imgui.checkbox("3D Matrix Plot", plot_enabled)
        if changed:
            dispatch(ToggleMatrixPlot())
        if imgui.is_item_hovered():
            imgui.set_tooltip("Toggle 3D value plot (off shows basis transform)")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Basic operations
        imgui.text("Basic Operations:")
        imgui.spacing()

        button_width = (width - 30) / 2
        for i, (op_id, op_name, op_desc) in enumerate(self.UNARY_OPS):
            if i > 0 and i % 2 == 0:
                pass  # New line
            elif i > 0:
                imgui.same_line()

            # Check if operation is valid
            is_square = tensor.rows == tensor.cols
            needs_square = op_id in ("inverse", "determinant", "trace")

            if needs_square and not is_square:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)

            if imgui.button(f"{op_name}##mat", button_width, 25):
                if not needs_square or is_square:
                    dispatch(ApplyOperation(
                        operation_name=op_id,
                        parameters=(),
                        target_ids=(tensor.id,),
                        create_new=True
                    ))

            if needs_square and not is_square:
                imgui.pop_style_var()

            if imgui.is_item_hovered():
                if needs_square and not is_square:
                    imgui.set_tooltip(f"{op_desc} (requires square matrix)")
                else:
                    imgui.set_tooltip(op_desc)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Scale operation
        imgui.text("Scale:")
        imgui.spacing()

        imgui.push_item_width(width - 100)
        _, self._scale_factor = imgui.input_float(
            "Factor##mat",
            self._scale_factor,
            0.1, 1.0,
            "%.2f"
        )
        imgui.pop_item_width()

        imgui.same_line()
        if imgui.button("Apply##scale_mat", 60, 0):
            dispatch(ApplyOperation(
                operation_name="scale",
                parameters=(("factor", str(self._scale_factor)),),
                target_ids=(tensor.id,),
                create_new=True
            ))

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Decompositions
        if tensor.rows == tensor.cols:
            imgui.text("Decompositions:")
            imgui.spacing()

            for i, (op_id, op_name, op_desc) in enumerate(self.DECOMPOSITION_OPS):
                if i > 0 and i % 2 == 0:
                    pass
                elif i > 0:
                    imgui.same_line()

                if imgui.button(f"{op_name}##decomp", button_width, 25):
                    dispatch(ApplyOperation(
                        operation_name=op_id,
                        parameters=(),
                        target_ids=(tensor.id,),
                        create_new=True
                    ))

                if imgui.is_item_hovered():
                    imgui.set_tooltip(op_desc)

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

        # Matrix-vector multiplication
        imgui.text("Apply to Vector:")
        imgui.spacing()

        vectors = list(get_vectors(state))
        if not vectors:
            imgui.text_colored("No vectors available", 0.5, 0.5, 0.5, 1.0)
        else:
            # Filter vectors with compatible dimensions
            compatible = [v for v in vectors if len(v.coords) == tensor.cols]

            if not compatible:
                imgui.text_colored(
                    f"No vectors with {tensor.cols} dimensions",
                    0.5, 0.5, 0.5, 1.0
                )
            else:
                vector_names = [v.label for v in compatible]
                if self._selected_vector_idx >= len(vector_names):
                    self._selected_vector_idx = 0

                imgui.push_item_width(width - 20)
                _, self._selected_vector_idx = imgui.combo(
                    "##target_vector",
                    self._selected_vector_idx,
                    vector_names
                )
                imgui.pop_item_width()

                imgui.spacing()

                if imgui.button("Transform Vector", width - 20, 25):
                    target_vec = compatible[self._selected_vector_idx]
                    dispatch(ApplyOperation(
                        operation_name="matrix_multiply",
                        parameters=(),
                        target_ids=(tensor.id, target_vec.id),
                        create_new=True
                    ))

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Matrix-matrix multiplication
        imgui.text("Matrix Multiply:")
        imgui.spacing()

        matrices = [m for m in get_matrices(state) if m.id != tensor.id]
        if not matrices:
            imgui.text_colored("No other matrices available", 0.5, 0.5, 0.5, 1.0)
        else:
            # Filter compatible matrices (cols == other.rows)
            compatible = [m for m in matrices if tensor.cols == m.rows]

            if not compatible:
                imgui.text_colored("No compatible matrices", 0.5, 0.5, 0.5, 1.0)
                imgui.text_colored(
                    f"(need {tensor.cols} rows)",
                    0.4, 0.4, 0.4, 1.0
                )
            else:
                matrix_names = [m.label for m in compatible]
                if self._selected_other_idx >= len(matrix_names):
                    self._selected_other_idx = 0

                imgui.push_item_width(width - 20)
                _, self._selected_other_idx = imgui.combo(
                    "##other_matrix",
                    self._selected_other_idx,
                    matrix_names
                )
                imgui.pop_item_width()

                imgui.spacing()

                other = compatible[self._selected_other_idx]
                result_shape = f"Result: {tensor.rows} x {other.cols}"
                imgui.text_colored(result_shape, 0.5, 0.5, 0.5, 1.0)

                if imgui.button("Multiply Matrices", width - 20, 25):
                    dispatch(ApplyOperation(
                        operation_name="matrix_multiply",
                        parameters=(),
                        target_ids=(tensor.id, other.id),
                        create_new=True
                    ))

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Quick actions
        if imgui.button("Duplicate", width - 20, 25):
            dispatch(DuplicateTensor(id=tensor.id))
