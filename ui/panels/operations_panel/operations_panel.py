"""
Main operations panel for the right side of the CVLA interface.

Features:
- Calculator view showing tensor math visually (T1 × T2 = Result)
- Operation tree with 30+ linear algebra operations
- CLI reflection of operations
"""

import imgui
import numpy as np
from typing import TYPE_CHECKING, Callable, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from state.app_state import AppState
    from state.models.tensor_model import TensorData

from ui.utils import set_next_window_position, set_next_window_size
from domain.vectors.vector_ops import gaussian_elimination_steps
from state.actions import (
    SetPipeline,
    SetViewPreset,
    SetViewUpAxis,
    ToggleViewGrid,
    ToggleViewLabels,
    ToggleView2D,
    SetViewGridSize,
    SetViewMajorTick,
    SetViewMinorTick,
    ToggleViewAutoRotate,
    SetViewRotationSpeed,
    ToggleViewCubeFaces,
    ToggleViewCubeCorners,
    ToggleViewTensorFaces,
    SetViewCubicGridDensity,
    SetViewCubeFaceOpacity,
)
from state.actions.matrix_actions import ToggleMatrixPlot
from state.actions.tensor_actions import (
    ApplyOperation,
    ClearBinaryOperation,
    DuplicateTensor,
    PreviewOperation,
    SetBinaryOperation,
    UpdateTensor,
    CancelPreview,
    ConfirmPreview,
)
from state.models import EducationalStep
from state.models.operation_record import OperationRecord
from state.models.tensor_model import TensorDType
from state.selectors import (
    get_matrices,
    get_vectors,
    get_tensor_stats,
    get_tensor_magnitude,
    get_tensor_norm,
)
from state.selectors.tensor_selectors import get_selected_tensor


_SET_WINDOW_POS_FIRST = getattr(imgui, "SET_WINDOW_POS_FIRST_USE_EVER", 0)
_SET_WINDOW_SIZE_FIRST = getattr(imgui, "SET_WINDOW_SIZE_FIRST_USE_EVER", 0)
_WINDOW_RESIZABLE = getattr(imgui, "WINDOW_RESIZABLE", 1)
_WINDOW_NO_COLLAPSE = getattr(imgui, "WINDOW_NO_COLLAPSE", 0)
_WINDOW_ALWAYS_VERTICAL_SCROLLBAR = getattr(
    imgui, "WINDOW_ALWAYS_VERTICAL_SCROLLBAR", 0
)


# ============================================================================
# OPERATION DEFINITIONS - 30+ Linear Algebra Operations
# ============================================================================

@dataclass
class OperationDef:
    """Definition of an operation button."""
    id: str
    name: str
    symbol: str  # Math symbol for calculator view
    description: str
    requires_square: bool = False
    requires_second_tensor: bool = False
    category: str = "basic"
    output_type: str = "tensor"  # tensor, scalar, tuple


# Operation categories with their operations
OPERATION_CATEGORIES: Dict[str, List[OperationDef]] = {
    "Basic Arithmetic": [
        OperationDef("add", "Add", "+", "Add two tensors element-wise", requires_second_tensor=True),
        OperationDef("subtract", "Subtract", "-", "Subtract tensors element-wise", requires_second_tensor=True),
        OperationDef("scale", "Scale", "×k", "Multiply by scalar"),
        OperationDef("negate", "Negate", "-A", "Negate all elements"),
        OperationDef("hadamard", "Hadamard", "⊙", "Element-wise multiplication", requires_second_tensor=True),
    ],
    "Matrix Multiplication": [
        OperationDef("matrix_multiply", "Matrix Multiply", "×", "Matrix multiplication A×B", requires_second_tensor=True),
        OperationDef("outer_product", "Outer Product", "⊗", "Outer product of vectors", requires_second_tensor=True),
        OperationDef("kronecker", "Kronecker", "⊗", "Kronecker product", requires_second_tensor=True),
    ],
    "Matrix Properties": [
        OperationDef("determinant", "Determinant", "det", "Compute determinant", requires_square=True, output_type="scalar"),
        OperationDef("trace", "Trace", "tr", "Sum of diagonal elements", requires_square=True, output_type="scalar"),
        OperationDef("rank", "Rank", "rank", "Matrix rank", output_type="scalar"),
        OperationDef("condition_number", "Condition #", "κ", "Condition number", output_type="scalar"),
        OperationDef("nullity", "Nullity", "null", "Dimension of null space", output_type="scalar"),
    ],
    "Transformations": [
        OperationDef("transpose", "Transpose", "Aᵀ", "Swap rows and columns"),
        OperationDef("inverse", "Inverse", "A⁻¹", "Matrix inverse", requires_square=True),
        OperationDef("pseudoinverse", "Pseudoinverse", "A⁺", "Moore-Penrose pseudoinverse"),
        OperationDef("adjoint", "Adjoint", "A*", "Conjugate transpose"),
        OperationDef("cofactor", "Cofactor", "cof", "Cofactor matrix", requires_square=True),
        OperationDef("adjugate", "Adjugate", "adj", "Adjugate matrix", requires_square=True),
    ],
    "Decompositions": [
        OperationDef("eigen", "Eigendecomp", "λ,v", "Eigenvalue decomposition", requires_square=True, output_type="tuple"),
        OperationDef("svd", "SVD", "UΣVᵀ", "Singular Value Decomposition", output_type="tuple"),
        OperationDef("qr", "QR", "QR", "QR decomposition", output_type="tuple"),
        OperationDef("lu", "LU", "LU", "LU decomposition", requires_square=True, output_type="tuple"),
        OperationDef("cholesky", "Cholesky", "LLᵀ", "Cholesky decomposition", requires_square=True),
        OperationDef("schur", "Schur", "QTQ*", "Schur decomposition", requires_square=True, output_type="tuple"),
    ],
    "Eigenvalues & Vectors": [
        OperationDef("eigenvalues", "Eigenvalues", "λ", "Compute eigenvalues", requires_square=True, output_type="tuple"),
        OperationDef("eigenvectors", "Eigenvectors", "v", "Compute eigenvectors", requires_square=True, output_type="tuple"),
        OperationDef("spectral_radius", "Spectral Radius", "ρ", "Largest eigenvalue magnitude", requires_square=True, output_type="scalar"),
        OperationDef("power_iteration", "Power Iteration", "Aⁿv", "Dominant eigenvector via iteration", requires_square=True),
    ],
    "Change of Basis": [
        OperationDef("change_basis", "Change Basis", "P⁻¹AP", "Change basis transformation", requires_second_tensor=True),
        OperationDef("similarity_transform", "Similarity", "S⁻¹AS", "Similarity transformation", requires_second_tensor=True),
        OperationDef("orthogonalize", "Orthogonalize", "orth", "Gram-Schmidt orthogonalization"),
        OperationDef("project_subspace", "Project Subspace", "proj", "Project onto column space", requires_second_tensor=True),
    ],
    "Norms & Distances": [
        OperationDef("frobenius_norm", "Frobenius", "‖A‖F", "Frobenius norm", output_type="scalar"),
        OperationDef("l1_norm", "L1 Norm", "‖A‖₁", "L1 norm (max column sum)", output_type="scalar"),
        OperationDef("l2_norm", "L2 Norm", "‖A‖₂", "L2 norm (spectral norm)", output_type="scalar"),
        OperationDef("inf_norm", "Infinity", "‖A‖∞", "Infinity norm (max row sum)", output_type="scalar"),
        OperationDef("nuclear_norm", "Nuclear", "‖A‖*", "Nuclear norm (sum of singular values)", output_type="scalar"),
    ],
    "Vector Operations": [
        OperationDef("normalize", "Normalize", "v̂", "Scale to unit length"),
        OperationDef("dot", "Dot Product", "·", "Compute dot product", requires_second_tensor=True, output_type="scalar"),
        OperationDef("cross", "Cross Product", "×", "Cross product (3D)", requires_second_tensor=True),
        OperationDef("project", "Project", "proj", "Project onto vector", requires_second_tensor=True),
        OperationDef("reject", "Reject", "rej", "Rejection from vector", requires_second_tensor=True),
        OperationDef("angle_between", "Angle", "θ", "Angle between vectors", requires_second_tensor=True, output_type="scalar"),
    ],
    "Linear Systems": [
        OperationDef("gaussian_elimination", "Gaussian Elim", "→REF", "Row echelon form"),
        OperationDef("rref", "RREF", "→RREF", "Reduced row echelon form"),
        OperationDef("back_substitution", "Back Sub", "←", "Back substitution"),
        OperationDef("solve_linear", "Solve Ax=b", "x", "Solve linear system", requires_second_tensor=True),
        OperationDef("least_squares", "Least Squares", "x̂", "Least squares solution", requires_second_tensor=True),
    ],
    "Special Matrices": [
        OperationDef("symmetrize", "Symmetrize", "sym", "Make symmetric: (A+Aᵀ)/2"),
        OperationDef("skew_symmetrize", "Skew-Sym", "skew", "Make skew-symmetric: (A-Aᵀ)/2"),
        OperationDef("diagonalize", "Diagonalize", "D", "Extract/form diagonal", requires_square=True),
        OperationDef("triangular_upper", "Upper Tri", "U", "Upper triangular part"),
        OperationDef("triangular_lower", "Lower Tri", "L", "Lower triangular part"),
    ],
}


# CLI Operation Log - stores recent operations for display
class CLIOperationLog:
    """Singleton to track operations for CLI display."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._log: List[Dict[str, Any]] = []
            cls._instance._max_entries = 50
        return cls._instance

    def log_operation(self, op_name: str, inputs: List[str], output: str,
                      math_repr: str, result_preview: str = ""):
        """Log an operation for CLI display."""
        entry = {
            "operation": op_name,
            "inputs": inputs,
            "output": output,
            "math": math_repr,
            "result": result_preview,
        }
        self._log.append(entry)
        if len(self._log) > self._max_entries:
            self._log.pop(0)
        # Print to CLI
        print(f"[CVLA] {math_repr}")
        if result_preview:
            print(f"       → {result_preview}")

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent operations."""
        return self._log[-n:]

    def clear(self):
        """Clear the log."""
        self._log.clear()


# Global CLI log instance
cli_log = CLIOperationLog()


# ============================================================================
# Tensor Inspector Widget
# ============================================================================


class TensorInspectorWidget:
    """Elevated inspector that surfaces tensor metadata attractively."""

    def __init__(self):
        self._panel_height = 320

    def render(self, tensor: Optional["TensorData"], width: float):
        imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.06, 0.06, 0.12, 1.0)
        imgui.push_style_color(imgui.COLOR_BORDER, 0.3, 0.5, 0.9, 0.8)
        if imgui.begin_child("##tensor_inspector", width - 10, self._panel_height, border=True):
            if tensor is None:
                imgui.spacing()
                imgui.text_colored("Select a tensor to inspect its properties.", 0.7, 0.55, 0.45, 1.0)
                imgui.text_colored("The inspector will summarize its order, shape, and values.", 0.5, 0.7, 0.8, 1.0)
            else:
                self._render_content(tensor, width)
        imgui.end_child()
        imgui.pop_style_color(2)

    def _render_content(self, tensor: "TensorData", width: float):
        imgui.spacing()
        imgui.text_colored("Tensor Inspector", 0.7, 0.8, 1.0, 1.0)
        imgui.text_colored(tensor.label, 0.8, 0.9, 0.6, 1.0)
        imgui.separator()
        rows = [
            ("Order", str(tensor.rank)),
            ("Shape", self._format_shape(tensor)),
            ("Type", self._format_type(tensor)),
        ]
        for title, value in rows:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.6, 0.8, 0.95, 1.0)
            imgui.text(f"{title}:")
            imgui.pop_style_color()
            imgui.same_line()
            imgui.text_colored(value, 0.95, 0.95, 0.75, 1.0)
        imgui.spacing()
        self._render_values_block(tensor, width)

    def _format_type(self, tensor: "TensorData") -> str:
        if tensor.rank == 0:
            return "Scalar"
        if tensor.rank == 1:
            rows, cols = self._vector_orientation(tensor)
            return f"Vector ({rows}×{cols})"
        if tensor.rank == 2:
            return "Matrix"
        return "Higher-order tensor"

    def _vector_orientation(self, tensor: "TensorData") -> Tuple[int, int]:
        if len(tensor.shape) == 2:
            return tensor.shape
        return (tensor.shape[0], 1)

    def _format_shape(self, tensor: "TensorData") -> str:
        if tensor.rank == 0:
            return "()"
        if tensor.rank == 1:
            if len(tensor.shape) == 2:
                rows, cols = tensor.shape
                orientation = "column" if cols == 1 else "row" if rows == 1 else "vector"
                return f"{rows}×{cols} ({orientation})"
            return f"({tensor.shape[0]},)"
        if tensor.rank == 2:
            return f"{tensor.shape[0]}×{tensor.shape[1]}"
        if tensor.shape:
            return "×".join(str(dim) for dim in tensor.shape)
        return "—"

    def _render_values_block(self, tensor: "TensorData", width: float):
        text = self._matrix_view_text(tensor)
        if not text:
            return
        block_width = max(width - 40, 160)
        imgui.text_colored("Values", 0.6, 0.8, 1.0, 1.0)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, 0.1, 0.1, 0.14, 1.0)
        imgui.push_style_color(imgui.COLOR_TEXT, 0.9, 0.9, 0.9, 1.0)
        _, _ = imgui.input_text_multiline(
            "##tensor_values",
            text,
            block_width,
            140,
            imgui.INPUT_TEXT_READ_ONLY | imgui.INPUT_TEXT_NO_HORIZONTAL_SCROLL,
        )
        imgui.pop_style_color(2)

    def _matrix_view_text(self, tensor: "TensorData") -> str:
        arr = tensor.to_numpy()
        if arr.ndim == 0:
            return str(float(arr))
        if arr.ndim == 1:
            separator = "\n" if self._vector_orientation(tensor)[1] == 1 else " "
            return separator.join(str(float(x)) for x in arr)
        lines = []
        for row in arr:
            lines.append("  ".join(str(float(v)) for v in row))
        return "\n".join(lines)



# ============================================================================
# OPERATION TREE WIDGET
# ============================================================================

class OperationTreeWidget:
    """
    Tree view of all available operations organized by category.
    Each operation is a clickable button.
    """

    def __init__(self):
        self._expanded_categories: Dict[str, bool] = {}
        self._selected_op: Optional[OperationDef] = None
        self._scale_value: float = 1.0

    def render(self, tensor: "TensorData", state: "AppState", dispatch, width: float):
        """Render the operation tree."""
        total_ops = sum(len(ops) for ops in OPERATION_CATEGORIES.values())
        imgui.text_colored(f"OPERATIONS ({total_ops})", 0.6, 0.8, 1.0, 1.0)
        imgui.separator()
        imgui.spacing()

        if tensor is None:
            imgui.text_colored("Select a tensor first", 0.5, 0.5, 0.5, 1.0)
            imgui.spacing()
            imgui.text_colored("Create tensors in the Input panel,", 0.4, 0.4, 0.4, 1.0)
            imgui.text_colored("then click one to see operations.", 0.4, 0.4, 0.4, 1.0)
            return

        # Render each category
        for category, operations in OPERATION_CATEGORIES.items():
            self._render_category(category, operations, tensor, state, dispatch,
                                  width)

    def _render_category(self, category: str, operations: List[OperationDef],
                         tensor: "TensorData", state: "AppState", dispatch,
                         width: float):
        """Render a category with its operations."""
        if category not in self._expanded_categories:
            self._expanded_categories[category] = True

        expanded = imgui.tree_node(f"{category}##{category}")

        if expanded:
            imgui.spacing()

            # Render operations as buttons in a 2-column grid
            button_width = max((width - 30) / 2, 100)
            col = 0

            for op in operations:
                is_selected = self._selected_op and self._selected_op.id == op.id

                if col > 0:
                    imgui.same_line()

                if is_selected:
                    imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.5, 0.3, 1.0)

                if imgui.button(f"{op.name}##{op.id}", button_width, 26):
                    self._handle_operation_click(op, tensor, state, dispatch)

                if is_selected:
                    imgui.pop_style_color(1)

                # Tooltip
                if imgui.is_item_hovered():
                    tooltip = f"{op.description}\nSymbol: {op.symbol}"
                    imgui.set_tooltip(tooltip)

                col = (col + 1) % 2

            imgui.tree_pop()
            imgui.spacing()

    def _handle_operation_click(self, op: OperationDef, tensor: "TensorData",
                                 state: "AppState", dispatch):
        """Handle clicking an operation button."""
        self._selected_op = op

        if op.requires_second_tensor:
            # Set binary operation mode - user will click second tensor in input panel
            dispatch(SetBinaryOperation(
                operation_name=op.id,
                first_tensor_id=tensor.id
            ))
        else:
            # Execute immediately for unary operations
            self._execute_operation(op, tensor, None, state, dispatch)

    def render_popups(self, tensor: "TensorData", state: "AppState", dispatch):
        """Render any open popups for operation configuration."""
        if tensor is None:
            return

        if self._selected_op and self._selected_op.requires_second_tensor:
            self._render_second_tensor_popup(tensor, state, dispatch)

        # Scale value popup
        if self._selected_op and self._selected_op.id == "scale":
            self._render_scale_popup(tensor, state, dispatch)

    def _render_second_tensor_popup(self, tensor: "TensorData", state: "AppState",
                                     dispatch):
        """Render popup for selecting second tensor."""
        op = self._selected_op
        if not op or tensor is None:
            return

        # Get compatible tensors
        compatible = self._get_compatible_tensors(op, tensor, state)

        if imgui.begin_popup(f"##select_second_{op.id}"):
            imgui.text_colored(f"Select second tensor for {op.name}", 0.6, 0.8, 1.0, 1.0)
            imgui.separator()
            imgui.spacing()

            if not compatible:
                imgui.text_colored("No compatible tensors available", 0.8, 0.4, 0.4, 1.0)
            else:
                for t in compatible:
                    if imgui.selectable(f"{t.label} ({self._format_shape(t)})")[0]:
                        self._execute_operation(op, tensor, t, state, dispatch)
                        imgui.close_current_popup()

            imgui.spacing()
            if imgui.button("Cancel", 100, 24):
                self._selected_op = None
                imgui.close_current_popup()

            imgui.end_popup()

    def _render_scale_popup(self, tensor: "TensorData", state: "AppState",
                            dispatch):
        """Render popup for scale factor input."""
        if tensor is None:
            return
        if imgui.begin_popup("##scale_input"):
            imgui.text("Scale Factor:")
            _, self._scale_value = imgui.input_float("##scale_val", self._scale_value)

            if imgui.button("Apply", 80, 24):
                dispatch(ApplyOperation(
                    operation_name="scale",
                    parameters=(("factor", str(self._scale_value)),),
                    target_ids=(tensor.id,),
                    create_new=True
                ))
                # Log to CLI
                cli_log.log_operation(
                    "scale",
                    [tensor.label],
                    f"{tensor.label}_scaled",
                    f"{self._scale_value} × {tensor.label}"
                )
                imgui.close_current_popup()

            imgui.same_line()
            if imgui.button("Cancel", 80, 24):
                imgui.close_current_popup()

            imgui.end_popup()

    def _get_compatible_tensors(self, op: OperationDef, tensor: "TensorData",
                                 state: "AppState") -> List["TensorData"]:
        """Get tensors compatible with the operation."""
        if state is None or tensor is None:
            return []

        compatible = []
        for t in state.tensors:
            if t.id == tensor.id:
                continue

            # Check compatibility based on operation
            if op.id in ("add", "subtract", "hadamard"):
                # Same shape required
                if t.shape == tensor.shape:
                    compatible.append(t)
            elif op.id == "matrix_multiply":
                # A.cols == B.rows
                if tensor.rank == 2 and t.rank == 2:
                    if tensor.cols == t.rows:
                        compatible.append(t)
                elif tensor.rank == 2 and t.rank == 1:
                    if tensor.cols == len(t.coords):
                        compatible.append(t)
            elif op.id in ("dot", "cross", "project", "reject", "angle_between"):
                # Same dimension vectors
                if tensor.rank == 1 and t.rank == 1:
                    if len(tensor.coords) == len(t.coords):
                        compatible.append(t)
            elif op.id == "outer_product":
                # Any two vectors
                if tensor.rank == 1 and t.rank == 1:
                    compatible.append(t)
            elif op.id in ("change_basis", "similarity_transform"):
                # Square matrix of same size
                if tensor.rank == 2 and t.rank == 2:
                    if tensor.rows == t.rows == t.cols:
                        compatible.append(t)
            else:
                # Default: same rank
                if t.rank == tensor.rank:
                    compatible.append(t)

        return compatible

    def _format_shape(self, tensor: "TensorData") -> str:
        """Format tensor shape for display."""
        return "×".join(str(d) for d in tensor.shape)

    def _execute_operation(self, op: OperationDef, tensor: "TensorData",
                           second_tensor: Optional["TensorData"], state: "AppState",
                           dispatch):
        """Execute the operation and update state."""
        if tensor is None:
            return
        # Build parameters
        params = ()
        target_ids = (tensor.id,)

        if second_tensor:
            params = (("other_id", second_tensor.id),)
            target_ids = (tensor.id, second_tensor.id)

        # Dispatch the operation
        dispatch(ApplyOperation(
            operation_name=op.id,
            parameters=params,
            target_ids=target_ids,
            create_new=True
        ))

        # Log to CLI
        if second_tensor:
            math_repr = f"{tensor.label} {op.symbol} {second_tensor.label}"
            inputs = [tensor.label, second_tensor.label]
        else:
            math_repr = f"{op.symbol}({tensor.label})"
            inputs = [tensor.label]

        cli_log.log_operation(op.id, inputs, f"result", math_repr)

        # Clear selection
        self._selected_op = None


# ============================================================================
# CLI LOG WIDGET
# ============================================================================

class CLILogWidget:
    """Widget showing recent operations in CLI-style format."""

    def __init__(self):
        self._show_full_log = False

    def render(self, width: float):
        """Render the CLI log."""
        imgui.text_colored("CLI LOG", 0.6, 0.8, 1.0, 1.0)
        imgui.same_line()
        if imgui.small_button("Clear"):
            cli_log.clear()
        imgui.separator()

        # Dark terminal-style background
        imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.05, 0.05, 0.08, 1.0)

        log_height = 120 if self._show_full_log else 80
        if imgui.begin_child("##cli_log", width - 20, log_height, border=True):
            recent = cli_log.get_recent(10 if self._show_full_log else 5)

            if not recent:
                imgui.text_colored("> Ready for operations...", 0.4, 0.4, 0.4, 1.0)
            else:
                for entry in recent:
                    imgui.text_colored(f"> {entry['math']}", 0.3, 0.8, 0.3, 1.0)
                    if entry.get('result'):
                        imgui.text_colored(f"  → {entry['result']}", 0.6, 0.6, 0.6, 1.0)

        imgui.end_child()
        imgui.pop_style_color(1)

        _, self._show_full_log = imgui.checkbox("Show full log", self._show_full_log)


# ============================================================================
# TENSOR INFO WIDGET (Simplified for new layout)
# ============================================================================

class TensorInfoWidget:
    """Compact widget displaying selected tensor details."""

    def __init__(self):
        self._label_buffer = ""
        self._editing_label = False

    def render(self, tensor: "TensorData", state: "AppState", dispatch, width: float):
        """Render tensor information."""
        if tensor is None:
            imgui.text_colored("No tensor selected", 0.5, 0.5, 0.5, 1.0)
            return

        # Compact header
        type_colors = {
            'r1': (0.4, 0.7, 1.0),
            'r2': (0.4, 1.0, 0.7),
            'r3': (0.8, 0.8, 0.8),
        }
        if tensor.rank == 1:
            rank_key = "r1"
            rank_label = "VEC"
        elif tensor.rank == 2:
            rank_key = "r2"
            rank_label = "MAT"
        else:
            rank_key = "r3"
            rank_label = f"R{tensor.rank}"
        color = type_colors.get(rank_key, (0.8, 0.8, 0.8))

        imgui.text_colored(rank_label, *color, 1.0)
        imgui.same_line()
        imgui.text(tensor.label)
        imgui.same_line()
        imgui.text_colored(f"({self._format_shape(tensor)})", 0.5, 0.5, 0.5, 1.0)

        # Edit button
        imgui.same_line()
        if imgui.small_button("Edit"):
            self._label_buffer = tensor.label
            self._editing_label = True
            imgui.open_popup("##edit_label")

        # Label edit popup
        if imgui.begin_popup("##edit_label"):
            changed, new_label = imgui.input_text("Label", self._label_buffer, 64)
            if changed:
                self._label_buffer = new_label
            if imgui.button("Save", 60, 0):
                if self._label_buffer.strip():
                    dispatch(UpdateTensor(id=tensor.id, label=self._label_buffer.strip()))
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", 60, 0):
                imgui.close_current_popup()
            imgui.end_popup()

        # Quick stats
        if tensor.rank == 1:
            mag = get_tensor_magnitude(tensor)
            imgui.text_colored(f"‖v‖ = {mag:.4f}", 0.6, 0.6, 0.6, 1.0)
        elif tensor.rank == 2:
            norm = get_tensor_norm(tensor)
            imgui.text_colored(f"‖A‖F = {norm:.4f}", 0.6, 0.6, 0.6, 1.0)
            if tensor.rows == tensor.cols:
                imgui.same_line()
                imgui.text_colored("(square)", 0.5, 0.7, 0.5, 1.0)

    def _format_shape(self, tensor: "TensorData") -> str:
        return "×".join(str(d) for d in tensor.shape)


# ============================================================================
# VIEW SETTINGS WIDGET
# ============================================================================

class ViewSettingsWidget:
    """Widget for visualization settings."""

    PRESETS = [
        ("cube", "Cube", "3D cubic grid view"),
        ("xy", "XY Plane", "Top-down 2D view"),
        ("xz", "XZ Plane", "Front 2D view"),
        ("yz", "YZ Plane", "Side 2D view"),
    ]

    UP_AXES = [
        ("z", "Z (Standard)", "Z-up coordinate system"),
        ("y", "Y (Blender/Unity)", "Y-up coordinate system"),
        ("x", "X", "X-up coordinate system"),
    ]

    def __init__(self):
        pass

    def render(self, state: "AppState", dispatch, width: float):
        """Render view settings UI."""
        if state is None:
            imgui.text_disabled("No state available")
            return

        imgui.text("VIEW SETTINGS")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # View preset
        imgui.text("View Preset:")
        imgui.spacing()

        button_width = (width - 30) / 2
        for i, (preset_id, preset_name, tooltip) in enumerate(self.PRESETS):
            if i > 0 and i % 2 == 0:
                pass
            elif i > 0:
                imgui.same_line()

            is_active = (
                (state.view_grid_mode == "cube" and preset_id == "cube") or
                (state.view_grid_mode == "plane" and state.view_grid_plane == preset_id)
            )

            if is_active:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.5, 0.7, 1.0)

            if imgui.button(preset_name, button_width, 25):
                dispatch(SetViewPreset(preset=preset_id))

            if is_active:
                imgui.pop_style_color(1)

            if imgui.is_item_hovered():
                imgui.set_tooltip(tooltip)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Display toggles
        imgui.text("Display:")
        imgui.spacing()

        changed, _ = imgui.checkbox("Show Grid", state.view_show_grid)
        if changed:
            dispatch(ToggleViewGrid())

        imgui.text_colored(
            "Axes stay visible and inherit the current theme colors.",
            0.6, 0.7, 0.9, 1.0
        )

        changed, _ = imgui.checkbox("Render Tensor Faces", state.view_show_tensor_faces)
        if changed:
            dispatch(ToggleViewTensorFaces())
        if imgui.is_item_hovered():
            imgui.set_tooltip("Rank-2 tensors get translucent faces with black outlines.")

        changed, _ = imgui.checkbox("Show Labels", state.view_show_labels)
        if changed:
            dispatch(ToggleViewLabels())

        changed, _ = imgui.checkbox("2D Mode", state.view_mode_2d)
        if changed:
            dispatch(ToggleView2D())


# ============================================================================
# MAIN OPERATIONS PANEL
# ============================================================================

class OperationsPanel:
    """
    Right-side panel for tensor operations and visualization.

    Layout:
    1. Calculator View - shows tensor math visually
    2. Tensor Info - selected tensor details
    3. Operation Tree - all operations as clickable buttons
    4. CLI Log - recent operations in terminal style
    """

    def __init__(self):
        self.inspector = TensorInspectorWidget()
        self.tensor_info = TensorInfoWidget()
        self.operation_tree = OperationTreeWidget()
        self.cli_log = CLILogWidget()
        self.view_settings = ViewSettingsWidget()

    def render(
        self,
        rect: tuple,
        state: "AppState",
        dispatch: Callable,
    ):
        """
        Render the operations panel.

        Args:
            rect: (x, y, width, height) for panel position
            state: Current AppState
            dispatch: Function to dispatch actions
        """
        x, y, width, height = rect

        set_next_window_position(x, y, cond=_SET_WINDOW_POS_FIRST)
        set_next_window_size((width, height), cond=_SET_WINDOW_SIZE_FIRST)
        imgui.set_next_window_size_constraints(
            (350, 400),
            (width + 100, height + 200),
        )

        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 4.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (12, 10))

        flags = _WINDOW_RESIZABLE | _WINDOW_NO_COLLAPSE

        if imgui.begin("Operations", flags=flags):
            selected = get_selected_tensor(state) if state else None
            active_mode = state.active_mode if state else "vectors"

            content_width = imgui.get_content_region_available_width()

            # Different content based on mode
            if active_mode == "visualize":
                self._render_view_mode(state, dispatch, content_width)
            else:
                self._render_operations_mode(selected, state, dispatch, content_width)

        imgui.end()
        imgui.pop_style_var(2)

    def _render_operations_mode(self, selected: "TensorData", state: "AppState",
                                 dispatch, width: float):
        """Render the main operations mode."""
        self.inspector.render(selected, width)
        imgui.spacing()

        # 2. Tensor Info (compact)
        self.tensor_info.render(selected, state, dispatch, width)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # 3. Operation Tree (scrollable, fills remaining space)
        if imgui.begin_child("##op_tree", width - 10, 0, border=False,
                            flags=_WINDOW_ALWAYS_VERTICAL_SCROLLBAR):
            self.operation_tree.render(selected, state, dispatch, width - 25)
            # Popups must be rendered in same ID stack where they were opened
            self.operation_tree.render_popups(selected, state, dispatch)
        imgui.end_child()

    def _render_view_mode(self, state: "AppState", dispatch, width: float):
        """Render view settings mode."""
        if imgui.begin_child("##view_content", 0, 0, border=False,
                            flags=_WINDOW_ALWAYS_VERTICAL_SCROLLBAR):
            self.view_settings.render(state, dispatch, width - 30)
        imgui.end_child()
