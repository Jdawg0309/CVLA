"""
Main operations panel for the right side of the CVLA interface.

Orchestrates tensor info, type-specific operations, and preview.
"""

import imgui
import numpy as np
from typing import TYPE_CHECKING, Callable, List, Tuple, Optional, Dict, Any

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
    ToggleViewAxes,
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
    DuplicateTensor,
    PreviewOperation,
    UpdateTensor,
    CancelPreview,
    ConfirmPreview,
)
from state.models import EducationalStep
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


# Available kernels with display names and descriptions
KERNELS: List[Tuple[str, str, str]] = [
    ("identity", "Identity", "No change"),
    ("box_blur", "Box Blur", "Simple averaging blur"),
    ("gaussian_blur", "Gaussian Blur", "Weighted 3x3 blur"),
    ("gaussian_blur_5x5", "Gaussian 5x5", "Stronger blur"),
    ("sharpen", "Sharpen", "Enhance edges and details"),
    ("sobel_x", "Sobel X", "Detect vertical edges"),
    ("sobel_y", "Sobel Y", "Detect horizontal edges"),
    ("laplacian", "Laplacian", "All-direction edges"),
    ("edge_detect", "Edge Detect", "Strong edge detection"),
    ("prewitt_x", "Prewitt X", "Simpler vertical edges"),
    ("prewitt_y", "Prewitt Y", "Simpler horizontal edges"),
    ("emboss", "Emboss", "3D embossed effect"),
    ("ridge_detect", "Ridge", "Horizontal line detection"),
]


class TensorInfoWidget:
    """Widget displaying selected tensor details."""

    def __init__(self):
        self._label_buffer = ""
        self._editing_label = False

    def render(self, tensor: "TensorData", state: "AppState", dispatch, width: float):
        """Render tensor information."""
        if tensor is None:
            imgui.text_colored("No tensor selected", 0.5, 0.5, 0.5, 1.0)
            imgui.spacing()
            imgui.text_colored("Select a tensor from the list", 0.5, 0.5, 0.5, 1.0)
            imgui.text_colored("to view details and operations.", 0.5, 0.5, 0.5, 1.0)
            return

        # Header with type icon
        type_colors = {
            'r1': (0.4, 0.7, 1.0),
            'r2': (0.4, 1.0, 0.7),
            'r3': (0.8, 0.8, 0.8),
        }
        if tensor.rank == 1:
            rank_key = "r1"
            rank_label = "RANK-1"
        elif tensor.rank == 2:
            rank_key = "r2"
            rank_label = "RANK-2"
        else:
            rank_key = "r3"
            rank_label = f"RANK-{tensor.rank}"
        color = type_colors.get(rank_key, (0.8, 0.8, 0.8))

        imgui.text_colored(rank_label, *color, 1.0)
        imgui.same_line()
        if tensor.dtype in (TensorDType.IMAGE_RGB, TensorDType.IMAGE_GRAYSCALE):
            imgui.text_disabled(f"[{tensor.dtype.value}]")
            imgui.same_line()

        # Editable label
        if self._editing_label:
            imgui.push_item_width(width - 100)
            changed, new_label = imgui.input_text(
                "##edit_label",
                self._label_buffer,
                256,
                imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
            )
            imgui.pop_item_width()
            if changed or imgui.is_key_pressed(imgui.KEY_ESCAPE):
                if changed and self._label_buffer.strip():
                    dispatch(UpdateTensor(id=tensor.id, label=self._label_buffer.strip()))
                self._editing_label = False
        else:
            imgui.text(tensor.label)
            imgui.same_line()
            if imgui.small_button("Edit"):
                self._label_buffer = tensor.label
                self._editing_label = True

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Basic properties
        imgui.text("Shape:")
        imgui.same_line(80)
        imgui.text(self._format_shape(tensor))

        imgui.text("DType:")
        imgui.same_line(80)
        imgui.text(str(tensor.dtype.value))

        imgui.text("Visible:")
        imgui.same_line(80)
        _, new_visible = imgui.checkbox("##visible", tensor.visible)
        if _ != tensor.visible:
            dispatch(UpdateTensor(id=tensor.id, visible=new_visible))

        # Color picker (for vectors/matrices)
        if tensor.rank in (1, 2):
            imgui.text("Color:")
            imgui.same_line(80)
            _, new_color = imgui.color_edit3(
                "##color",
                tensor.color[0], tensor.color[1], tensor.color[2]
            )
            if new_color != tensor.color:
                dispatch(UpdateTensor(id=tensor.id, color=new_color))

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Type-specific info (dtype takes precedence over rank)
        if tensor.dtype in (TensorDType.IMAGE_RGB, TensorDType.IMAGE_GRAYSCALE):
            self._render_image_info(tensor)
        elif tensor.rank == 1:
            self._render_vector_info(tensor)
        elif tensor.rank == 2:
            self._render_matrix_info(tensor)

    def _format_shape(self, tensor: "TensorData") -> str:
        """Format tensor shape for display."""
        return " x ".join(str(d) for d in tensor.shape)

    def _render_vector_info(self, tensor: "TensorData"):
        """Render vector-specific information."""
        imgui.text("Coordinates:")
        coords = tensor.coords
        for i, c in enumerate(coords):
            axis = ['x', 'y', 'z', 'w'][i] if i < 4 else f"[{i}]"
            imgui.text(f"  {axis}: {c:.4f}")

        imgui.spacing()
        magnitude = get_tensor_magnitude(tensor)
        imgui.text(f"Magnitude: {magnitude:.4f}")

    def _render_matrix_info(self, tensor: "TensorData"):
        """Render matrix-specific information."""
        imgui.text("Values:")

        # Show matrix values in a compact grid
        values = tensor.values
        rows = min(len(values), 5)  # Show at most 5 rows
        cols = min(len(values[0]) if values else 0, 5)  # Show at most 5 cols

        for r in range(rows):
            row_str = "  "
            for c in range(cols):
                row_str += f"{values[r][c]:8.3f}"
            if cols < len(values[0]):
                row_str += " ..."
            imgui.text(row_str)

        if rows < len(values):
            imgui.text("  ...")

        imgui.spacing()
        norm = get_tensor_norm(tensor)
        imgui.text(f"Frobenius Norm: {norm:.4f}")

    def _render_image_info(self, tensor: "TensorData"):
        """Render image-specific information."""
        imgui.text(f"Dimensions: {tensor.height} x {tensor.width}")
        imgui.text(f"Channels: {tensor.channels}")
        imgui.text(f"Grayscale: {'Yes' if tensor.is_grayscale else 'No'}")

        imgui.spacing()

        # Statistics
        stats = get_tensor_stats(tensor)
        imgui.text("Statistics:")
        imgui.text(f"  Min: {stats[0]:.4f}")
        imgui.text(f"  Max: {stats[1]:.4f}")
        imgui.text(f"  Mean: {stats[2]:.4f}")
        imgui.text(f"  Std: {stats[3]:.4f}")


class VectorOpsWidget:
    """Widget for vector operations."""

    UNARY_OPS = [
        ("normalize", "Normalize", "Scale to unit length"),
        ("negate", "Negate", "Reverse direction"),
    ]

    BINARY_OPS = [
        ("add", "Add", "Add two vectors"),
        ("subtract", "Subtract", "Subtract vectors"),
        ("dot", "Dot Product", "Compute dot product"),
        ("cross", "Cross Product", "Compute cross product (3D)"),
        ("project", "Project", "Project onto another vector"),
    ]

    SCALAR_OPS = [
        ("scale", "Scale", "Multiply by scalar"),
    ]

    def __init__(self):
        self._scale_factor = 1.0
        self._selected_other_idx = 0

    def render(self, tensor: "TensorData", state: "AppState", dispatch, width: float):
        """Render vector operations UI."""
        if tensor is None or tensor.rank != 1:
            return

        imgui.text("RANK-1 OPERATIONS")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Unary operations
        imgui.text("Transform:")
        imgui.spacing()

        for op_id, op_name, op_desc in self.UNARY_OPS:
            if imgui.button(op_name, width - 20, 25):
                dispatch(ApplyOperation(
                    operation_name=op_id,
                    parameters=(),
                    target_ids=(tensor.id,),
                    create_new=True
                ))
            if imgui.is_item_hovered():
                imgui.set_tooltip(op_desc)
            imgui.spacing()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Scalar operations
        imgui.text("Scale:")
        imgui.spacing()

        imgui.push_item_width(width - 100)
        _, self._scale_factor = imgui.input_float(
            "Factor",
            self._scale_factor,
            0.1, 1.0,
            "%.2f"
        )
        imgui.pop_item_width()

        imgui.same_line()
        if imgui.button("Apply##scale", 60, 0):
            dispatch(ApplyOperation(
                operation_name="scale",
                parameters=(("factor", str(self._scale_factor)),),
                target_ids=(tensor.id,),
                create_new=True
            ))

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Binary operations (need another rank-1 tensor)
        imgui.text("With Another Rank-1 Tensor:")
        imgui.spacing()

        # Get other vectors
        other_vectors = [v for v in get_vectors(state) if v.id != tensor.id]

        if not other_vectors:
            imgui.text_colored("No other rank-1 tensors available", 0.5, 0.5, 0.5, 1.0)
        else:
            # Vector selector
            vector_names = [v.label for v in other_vectors]
            if self._selected_other_idx >= len(vector_names):
                self._selected_other_idx = 0

            imgui.push_item_width(width - 20)
            _, self._selected_other_idx = imgui.combo(
                "##other_vector",
                self._selected_other_idx,
                vector_names
            )
            imgui.pop_item_width()

            imgui.spacing()

            other = other_vectors[self._selected_other_idx]

            # Binary operation buttons
            button_width = (width - 30) / 2
            for i, (op_id, op_name, op_desc) in enumerate(self.BINARY_OPS):
                if i > 0 and i % 2 == 0:
                    pass  # New line
                elif i > 0:
                    imgui.same_line()

                # Special case: cross product only for 3D
                if op_id == "cross" and len(tensor.coords) != 3:
                    imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)

                if imgui.button(f"{op_name}##bin", button_width, 25):
                    if op_id != "cross" or len(tensor.coords) == 3:
                        dispatch(ApplyOperation(
                            operation_name=op_id,
                            parameters=(("other_id", other.id),),
                            target_ids=(tensor.id, other.id),
                            create_new=True
                        ))

                if op_id == "cross" and len(tensor.coords) != 3:
                    imgui.pop_style_var()

                if imgui.is_item_hovered():
                    imgui.set_tooltip(op_desc)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Quick actions
        imgui.text("Quick Actions:")
        imgui.spacing()

        half_width = (width - 30) / 2

        if imgui.button("Duplicate", half_width, 25):
            dispatch(DuplicateTensor(id=tensor.id))

        imgui.same_line()

        if imgui.button("To Origin", half_width, 25):
            dispatch(ApplyOperation(
                operation_name="to_origin",
                parameters=(),
                target_ids=(tensor.id,),
                create_new=False
            ))


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
        if tensor is None or tensor.rank != 2:
            return

        imgui.text("RANK-2 OPERATIONS")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Visualization toggles
        imgui.text("Visualization:")
        imgui.spacing()
        plot_enabled = getattr(state, "matrix_plot_enabled", False)
        changed, plot_enabled = imgui.checkbox("3D Rank-2 Plot", plot_enabled)
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
        imgui.text("Multiply by Rank-1 Tensor:")
        imgui.spacing()

        vectors = list(get_vectors(state))
        if not vectors:
            imgui.text_colored("No rank-1 tensors available", 0.5, 0.5, 0.5, 1.0)
        else:
            # Filter vectors with compatible dimensions
            compatible = [v for v in vectors if len(v.coords) == tensor.cols]

            if not compatible:
                imgui.text_colored(
                    f"No rank-1 tensors with {tensor.cols} dimensions",
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

                if imgui.button("Apply Rank-2 x Rank-1", width - 20, 25):
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
        imgui.text("Rank-2 Multiply:")
        imgui.spacing()

        matrices = [m for m in get_matrices(state) if m.id != tensor.id]
        if not matrices:
            imgui.text_colored("No other rank-2 tensors available", 0.5, 0.5, 0.5, 1.0)
        else:
            # Filter compatible matrices (cols == other.rows)
            compatible = [m for m in matrices if tensor.cols == m.rows]

            if not compatible:
                imgui.text_colored("No compatible rank-2 tensors", 0.5, 0.5, 0.5, 1.0)
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

                if imgui.button("Multiply Rank-2 Tensors", width - 20, 25):
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


class ImageOpsWidget:
    """Widget for image operations."""

    def __init__(self):
        self._selected_kernel_idx = 0
        self._rotation_angle = 0.0
        self._scale_factor = 1.0
        self._normalize_mean = 0.0
        self._normalize_std = 1.0
        self._show_kernel_preview = False

    def render(self, tensor: "TensorData", state: "AppState", dispatch, width: float):
        """Render image operations UI."""
        if tensor is None or tensor.dtype not in (TensorDType.IMAGE_RGB, TensorDType.IMAGE_GRAYSCALE):
            return

        imgui.text("IMAGE DTYPE OPS")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Image info
        h, w = tensor.shape[0], tensor.shape[1]
        channels = tensor.shape[2] if len(tensor.shape) > 2 else 1
        imgui.text_colored(f"Size: {w}x{h}", 0.7, 0.7, 0.7, 1.0)
        imgui.same_line()
        ch_str = "RGB" if channels == 3 else ("RGBA" if channels == 4 else "Grayscale")
        imgui.text_colored(f"Channels: {ch_str}", 0.7, 0.7, 0.7, 1.0)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Convolution section
        self._render_convolution_section(tensor, dispatch, width)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Transform section
        self._render_transform_section(tensor, dispatch, width)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Preprocessing section
        self._render_preprocessing_section(tensor, dispatch, width)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Quick actions
        self._render_quick_actions(tensor, dispatch, width)

    def _render_convolution_section(self, tensor, dispatch, width):
        """Render convolution/kernel operations."""
        imgui.text("Convolution:")
        imgui.spacing()

        # Kernel selector
        kernel_names = [k[1] for k in KERNELS]
        imgui.push_item_width(width - 20)
        changed, self._selected_kernel_idx = imgui.combo(
            "##kernel_select",
            self._selected_kernel_idx,
            kernel_names
        )
        imgui.pop_item_width()

        # Show kernel description
        kernel_id, kernel_name, kernel_desc = KERNELS[self._selected_kernel_idx]
        imgui.text_colored(kernel_desc, 0.6, 0.6, 0.6, 1.0)

        imgui.spacing()

        # Preview toggle
        _, self._show_kernel_preview = imgui.checkbox(
            "Show preview",
            self._show_kernel_preview
        )

        if self._show_kernel_preview and changed:
            dispatch(PreviewOperation(
                operation_name="apply_kernel",
                parameters=(("kernel", kernel_id),),
                target_id=tensor.id
            ))

        imgui.spacing()

        # Apply button
        if imgui.button("Apply Kernel", width - 20, 28):
            dispatch(ApplyOperation(
                operation_name="apply_kernel",
                parameters=(("kernel", kernel_id),),
                target_ids=(tensor.id,),
                create_new=False
            ))

        imgui.spacing()

        # Quick kernel buttons (edge detection)
        imgui.text_disabled("Quick filters:")
        half_width = (width - 30) / 2

        if imgui.button("Sobel", half_width, 22):
            dispatch(ApplyOperation(
                operation_name="apply_kernel",
                parameters=(("kernel", "sobel_x"),),
                target_ids=(tensor.id,),
                create_new=False
            ))
        imgui.same_line()
        if imgui.button("Blur", half_width, 22):
            dispatch(ApplyOperation(
                operation_name="apply_kernel",
                parameters=(("kernel", "gaussian_blur"),),
                target_ids=(tensor.id,),
                create_new=False
            ))

        if imgui.button("Sharpen", half_width, 22):
            dispatch(ApplyOperation(
                operation_name="apply_kernel",
                parameters=(("kernel", "sharpen"),),
                target_ids=(tensor.id,),
                create_new=False
            ))
        imgui.same_line()
        if imgui.button("Edges", half_width, 22):
            dispatch(ApplyOperation(
                operation_name="apply_kernel",
                parameters=(("kernel", "edge_detect"),),
                target_ids=(tensor.id,),
                create_new=False
            ))

    def _render_transform_section(self, tensor, dispatch, width):
        """Render affine transform operations."""
        imgui.text("Transform:")
        imgui.spacing()

        # Rotation
        imgui.text_disabled("Rotation (degrees):")
        imgui.push_item_width(width - 80)
        _, self._rotation_angle = imgui.slider_float(
            "##rotation",
            self._rotation_angle,
            -180.0, 180.0,
            "%.1f"
        )
        imgui.pop_item_width()
        imgui.same_line()
        if imgui.button("R##apply_rot", 50, 0):
            dispatch(ApplyOperation(
                operation_name="rotate",
                parameters=(("angle", str(self._rotation_angle)),),
                target_ids=(tensor.id,),
                create_new=False
            ))

        imgui.spacing()

        # Scale
        imgui.text_disabled("Scale factor:")
        imgui.push_item_width(width - 80)
        _, self._scale_factor = imgui.slider_float(
            "##scale_img",
            self._scale_factor,
            0.25, 4.0,
            "%.2fx"
        )
        imgui.pop_item_width()
        imgui.same_line()
        if imgui.button("S##apply_scale", 50, 0):
            dispatch(ApplyOperation(
                operation_name="scale_image",
                parameters=(("factor", str(self._scale_factor)),),
                target_ids=(tensor.id,),
                create_new=False
            ))

        imgui.spacing()

        # Flip buttons
        half_width = (width - 30) / 2
        if imgui.button("Flip H", half_width, 22):
            dispatch(ApplyOperation(
                operation_name="flip_horizontal",
                parameters=(),
                target_ids=(tensor.id,),
                create_new=False
            ))
        imgui.same_line()
        if imgui.button("Flip V", half_width, 22):
            dispatch(ApplyOperation(
                operation_name="flip_vertical",
                parameters=(),
                target_ids=(tensor.id,),
                create_new=False
            ))

    def _render_preprocessing_section(self, tensor, dispatch, width):
        """Render preprocessing operations."""
        imgui.text("Preprocessing:")
        imgui.spacing()

        # Normalize
        imgui.text_disabled("Normalize (mean, std):")
        imgui.push_item_width((width - 30) / 2)
        _, self._normalize_mean = imgui.input_float(
            "##norm_mean",
            self._normalize_mean,
            0.01, 0.1,
            "%.3f"
        )
        imgui.same_line()
        _, self._normalize_std = imgui.input_float(
            "##norm_std",
            self._normalize_std,
            0.01, 0.1,
            "%.3f"
        )
        imgui.pop_item_width()

        imgui.spacing()

        if imgui.button("Normalize", width - 20, 25):
            dispatch(ApplyOperation(
                operation_name="normalize",
                parameters=(
                    ("mean", str(self._normalize_mean)),
                    ("std", str(self._normalize_std)),
                ),
                target_ids=(tensor.id,),
                create_new=False
            ))

        imgui.spacing()

        # Color mode conversion
        half_width = (width - 30) / 2
        if imgui.button("To Grayscale", half_width, 22):
            dispatch(ApplyOperation(
                operation_name="to_grayscale",
                parameters=(),
                target_ids=(tensor.id,),
                create_new=False
            ))
        imgui.same_line()
        if imgui.button("Invert", half_width, 22):
            dispatch(ApplyOperation(
                operation_name="invert",
                parameters=(),
                target_ids=(tensor.id,),
                create_new=False
            ))

    def _render_quick_actions(self, tensor, dispatch, width):
        """Render quick action buttons."""
        imgui.text("Quick Actions:")
        imgui.spacing()

        half_width = (width - 30) / 2

        if imgui.button("Duplicate", half_width, 25):
            dispatch(DuplicateTensor(id=tensor.id))

        imgui.same_line()

        if imgui.button("Reset", half_width, 25):
            dispatch(ApplyOperation(
                operation_name="reset_image",
                parameters=(),
                target_ids=(tensor.id,),
                create_new=False
            ))

        imgui.spacing()

        if imgui.button("Export as Rank-2", width - 20, 25):
            dispatch(ApplyOperation(
                operation_name="to_matrix",
                parameters=(),
                target_ids=(tensor.id,),
                create_new=False
            ))


class LinearSystemsWidget:
    """Widget for Gaussian elimination over Ax = b."""

    def __init__(self):
        self._steps: Optional[List[Dict[str, Any]]] = None
        self._status: Optional[str] = None
        self._solution: Optional[np.ndarray] = None

    def render(self, state: "AppState", dispatch, width: float, selected: Optional["TensorData"]):
        if state is None:
            return

        imgui.text("LINEAR SYSTEMS")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if selected is None or selected.rank != 2:
            imgui.text_colored(
                "Select a rank-2 tensor from the left panel.",
                0.6, 0.6, 0.6, 1.0
            )
            imgui.spacing()
            imgui.text_wrapped(
                "Gaussian elimination runs on the selected tensor. "
                "Use an augmented matrix [A | b] with shape n x (n+1)."
            )
            return

        rows = selected.rows
        cols = selected.cols
        if cols != rows + 1:
            imgui.text_colored(
                f"Selected tensor shape is {rows} x {cols}.",
                1.0, 0.6, 0.2, 1.0
            )
            imgui.text_wrapped(
                "Expected an augmented matrix with shape n x (n+1) "
                "to represent [A | b]."
            )
            return

        imgui.text_disabled(f"Using selected tensor: {selected.label} ({rows} x {cols})")
        imgui.spacing()
        if imgui.button("Gaussian Elimination", width - 20, 26):
            self._run_elimination(selected, dispatch)

        if self._status is not None:
            imgui.spacing()
            if self._status == "unique" and self._solution is not None:
                imgui.text_colored("Solution:", 0.2, 0.8, 0.2, 1.0)
                for i, val in enumerate(self._solution):
                    imgui.text(f"x{i + 1} = {val:.4g}")
            elif self._status == "infinite":
                imgui.text_colored("Infinite solutions (free variables).", 1.0, 0.7, 0.2, 1.0)
            elif self._status == "inconsistent":
                imgui.text_colored("No solution (inconsistent system).", 1.0, 0.4, 0.4, 1.0)

        if self._steps:
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            step_idx = min(state.pipeline_step_index, len(self._steps) - 1)
            step = self._steps[step_idx]

            imgui.text(f"Step {step_idx + 1} / {len(self._steps)}: {step['title']}")
            if step.get("description"):
                imgui.text_wrapped(step["description"])

            imgui.spacing()
            matrix = step.get("matrix")
            if matrix:
                self._render_matrix(matrix)

    def _run_elimination(self, tensor: "TensorData", dispatch):
        values = np.array(tensor.values, dtype=np.float32)
        n = values.shape[0]
        A = values[:, :n]
        b = values[:, n]

        steps, solution, status = gaussian_elimination_steps(A, b)
        self._steps = steps
        self._solution = solution
        self._status = status

        pipeline_steps = tuple(
            EducationalStep.create(
                title=step["title"],
                explanation=step["description"],
                operation="gaussian_elimination",
            )
            for step in steps
        )
        if dispatch:
            dispatch(SetPipeline(steps=pipeline_steps, index=0))

    def _render_matrix(self, matrix):
        rows = len(matrix)
        cols = len(matrix[0]) if rows else 0

        table_flags = 0
        try:
            table_flags = imgui.TABLE_BORDERS_INNER_H | imgui.TABLE_BORDERS_OUTER
        except Exception:
            pass

        if imgui.begin_table("##ls_matrix", cols, table_flags):
            for i in range(rows):
                imgui.table_next_row()
                for j in range(cols):
                    imgui.table_next_column()
                    imgui.text(f"{matrix[i][j]:.2f}")
            imgui.end_table()


class OperationPreviewWidget:
    """Widget showing operation preview with before/after views."""

    def __init__(self):
        self._show_diff = False
        self._split_view = True

    def render(self, tensor: "TensorData", state: "AppState", dispatch, width: float):
        """Render operation preview UI."""
        pending_op = getattr(state, 'pending_operation', None)
        preview_result = getattr(state, 'operation_preview_tensor', None)

        if not pending_op:
            imgui.text_colored("No pending operation", 0.5, 0.5, 0.5, 1.0)
            return

        imgui.text("OPERATION PREVIEW")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Operation name and parameters
        imgui.text(f"Operation: ")
        imgui.same_line()
        imgui.text_colored(pending_op, 0.4, 0.8, 0.4, 1.0)

        params = getattr(state, 'pending_operation_params', ())
        if params:
            imgui.spacing()
            imgui.text_disabled("Parameters:")
            for key, value in params:
                imgui.text(f"  {key}: {value}")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # View options
        _, self._split_view = imgui.checkbox("Split view", self._split_view)
        imgui.same_line()
        _, self._show_diff = imgui.checkbox("Show diff", self._show_diff)

        imgui.spacing()

        # Preview visualization
        self._render_preview_visualization(
            tensor, preview_result, state, width
        )

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Action buttons
        half_width = (width - 30) / 2

        imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.5, 0.2, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.6, 0.3, 1.0)
        if imgui.button("Apply", half_width, 30):
            dispatch(ConfirmPreview())
        imgui.pop_style_color(2)

        imgui.same_line()

        imgui.push_style_color(imgui.COLOR_BUTTON, 0.5, 0.2, 0.2, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.6, 0.3, 0.3, 1.0)
        if imgui.button("Cancel", half_width, 30):
            dispatch(CancelPreview())
        imgui.pop_style_color(2)

    def _render_preview_visualization(
        self,
        original: "TensorData",
        preview: Optional["TensorData"],
        state: "AppState",
        width: float
    ):
        """Render the before/after visualization."""
        preview_height = 150

        if original is None:
            imgui.text_colored("No tensor selected", 0.5, 0.5, 0.5, 1.0)
            return

        if preview is None:
            imgui.text_colored("Computing preview...", 0.7, 0.7, 0.3, 1.0)
            return

        if self._split_view:
            # Side by side view
            half = (width - 30) / 2

            imgui.begin_child("##preview_before", half, preview_height, border=True)
            imgui.text_colored("BEFORE", 0.7, 0.7, 0.7, 1.0)
            imgui.separator()
            self._render_tensor_preview(original, half - 10)
            imgui.end_child()

            imgui.same_line()

            imgui.begin_child("##preview_after", half, preview_height, border=True)
            imgui.text_colored("AFTER", 0.4, 0.8, 0.4, 1.0)
            imgui.separator()
            self._render_tensor_preview(preview, half - 10)
            imgui.end_child()
        else:
            # Single view (after only)
            imgui.begin_child("##preview_result", width - 20, preview_height, border=True)
            imgui.text_colored("RESULT", 0.4, 0.8, 0.4, 1.0)
            imgui.separator()
            self._render_tensor_preview(preview, width - 30)
            imgui.end_child()

        if self._show_diff and original is not None and preview is not None:
            imgui.spacing()
            self._render_diff_info(original, preview)

    def _render_tensor_preview(self, tensor: "TensorData", width: float):
        """Render a compact preview of a tensor."""
        if tensor is None:
            imgui.text_disabled("(no data)")
            return

        if tensor.dtype in (TensorDType.IMAGE_RGB, TensorDType.IMAGE_GRAYSCALE):
            # Show image dimensions and stats
            h, w = tensor.shape[0], tensor.shape[1]
            channels = tensor.shape[2] if len(tensor.shape) > 2 else 1
            imgui.text(f"Size: {w}x{h}")
            ch_str = "RGB" if channels == 3 else ("RGBA" if channels == 4 else "Gray")
            imgui.text(f"Format: {ch_str}")

            # Would show thumbnail here if we had image rendering
            imgui.text_colored("(image preview)", 0.5, 0.5, 0.5, 1.0)

        elif tensor.rank == 1:
            # Show vector coordinates
            coords = tensor.coords
            if len(coords) <= 4:
                coord_str = ", ".join(f"{c:.3f}" for c in coords)
                imgui.text(f"[{coord_str}]")
            else:
                coord_str = ", ".join(f"{c:.2f}" for c in coords[:3])
                imgui.text(f"[{coord_str}, ...]")

            # Show magnitude
            magnitude = sum(c * c for c in coords) ** 0.5
            imgui.text_colored(f"Magnitude: {magnitude:.4f}", 0.6, 0.6, 0.6, 1.0)

        elif tensor.rank == 2:
            # Show matrix dimensions and sample values
            rows, cols = tensor.shape[0], tensor.shape[1]
            imgui.text(f"Shape: {rows}x{cols}")

            # Show first few values
            if rows > 0 and cols > 0:
                data = tensor.data
                for i in range(min(3, rows)):
                    row = data[i] if i < len(data) else ()
                    vals = ", ".join(f"{v:.2f}" for v in row[:min(4, cols)])
                    if cols > 4:
                        vals += ", ..."
                    imgui.text_colored(f"[{vals}]", 0.7, 0.7, 0.7, 1.0)
                if rows > 3:
                    imgui.text_colored("...", 0.5, 0.5, 0.5, 1.0)

    def _render_diff_info(self, original: "TensorData", preview: "TensorData"):
        """Show difference statistics between original and preview."""
        imgui.text("Changes:")

        # Shape change
        if original.shape != preview.shape:
            imgui.text_colored(
                f"Shape: {original.shape} -> {preview.shape}",
                0.8, 0.6, 0.2, 1.0
            )

        # For numeric tensors, compute difference stats
        if original.rank == 1 and preview.rank == 1:
            orig_coords = original.coords
            prev_coords = preview.coords
            if len(orig_coords) == len(prev_coords):
                diff = [abs(a - b) for a, b in zip(orig_coords, prev_coords)]
                max_diff = max(diff) if diff else 0
                avg_diff = sum(diff) / len(diff) if diff else 0
                imgui.text(f"Max change: {max_diff:.4f}")
                imgui.text(f"Avg change: {avg_diff:.4f}")


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
                pass  # New row
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

        # Up axis
        imgui.text("Up Axis:")
        imgui.same_line(80)
        imgui.push_item_width(width - 100)

        current_axis = state.view_up_axis
        axis_names = [a[1] for a in self.UP_AXES]
        current_idx = next(
            (i for i, a in enumerate(self.UP_AXES) if a[0] == current_axis),
            0
        )

        changed, new_idx = imgui.combo("##up_axis", current_idx, axis_names)
        if changed:
            dispatch(SetViewUpAxis(axis=self.UP_AXES[new_idx][0]))
        imgui.pop_item_width()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Display toggles
        imgui.text("Display:")
        imgui.spacing()

        changed, _ = imgui.checkbox("Show Grid", state.view_show_grid)
        if changed:
            dispatch(ToggleViewGrid())

        changed, _ = imgui.checkbox("Show Axes", state.view_show_axes)
        if changed:
            dispatch(ToggleViewAxes())

        changed, _ = imgui.checkbox("Show Labels", state.view_show_labels)
        if changed:
            dispatch(ToggleViewLabels())

        changed, _ = imgui.checkbox("Show Tensor Faces", state.view_show_tensor_faces)
        if changed:
            dispatch(ToggleViewTensorFaces())

        changed, _ = imgui.checkbox("2D Mode", state.view_mode_2d)
        if changed:
            dispatch(ToggleView2D())

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Grid settings
        if state.view_show_grid:
            imgui.text("Grid Settings:")
            imgui.spacing()

            imgui.push_item_width(width - 100)

            changed, new_size = imgui.slider_int(
                "Grid Size", state.view_grid_size, 5, 50
            )
            if changed:
                dispatch(SetViewGridSize(size=new_size))

            changed, new_major = imgui.slider_int(
                "Major Ticks", state.view_major_tick, 1, 10
            )
            if changed:
                dispatch(SetViewMajorTick(value=new_major))

            changed, new_minor = imgui.slider_int(
                "Minor Ticks", state.view_minor_tick, 1, 5
            )
            if changed:
                dispatch(SetViewMinorTick(value=new_minor))

            imgui.pop_item_width()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

        # Cubic view settings
        if state.view_grid_mode == "cube":
            imgui.text("Cubic View:")
            imgui.spacing()

            changed, _ = imgui.checkbox("Auto-rotate", state.view_auto_rotate)
            if changed:
                dispatch(ToggleViewAutoRotate())

            imgui.push_item_width(width - 100)

            if state.view_auto_rotate:
                changed, new_speed = imgui.slider_float(
                    "Rotation Speed", state.view_rotation_speed, 0.1, 2.0
                )
                if changed:
                    dispatch(SetViewRotationSpeed(speed=new_speed))

            changed, _ = imgui.checkbox("Show Cube Faces", state.view_show_cube_faces)
            if changed:
                dispatch(ToggleViewCubeFaces())

            changed, _ = imgui.checkbox("Corner Indicators", state.view_show_cube_corners)
            if changed:
                dispatch(ToggleViewCubeCorners())

            changed, new_density = imgui.slider_float(
                "Grid Density", state.view_cubic_grid_density, 0.5, 3.0
            )
            if changed:
                dispatch(SetViewCubicGridDensity(density=new_density))

            if state.view_show_cube_faces:
                changed, new_opacity = imgui.slider_float(
                    "Face Opacity", state.view_cube_face_opacity, 0.01, 0.3
                )
                if changed:
                    dispatch(SetViewCubeFaceOpacity(opacity=new_opacity))

            imgui.pop_item_width()

class OperationsPanel:
    """
    Right-side panel for tensor operations and visualization.

    Shows:
    - Selected tensor info
    - Type-specific operations (vector/matrix/image)
    - Operation preview (before/after)
    """

    def __init__(self):
        self.tensor_info = TensorInfoWidget()
        self.vector_ops = VectorOpsWidget()
        self.matrix_ops = MatrixOpsWidget()
        self.image_ops = ImageOpsWidget()
        self.linear_systems = LinearSystemsWidget()
        self.preview = OperationPreviewWidget()
        self.view_settings = ViewSettingsWidget()

        # Panel state
        self._show_preview = True

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
            (280, 300),
            (width + 100, height + 200),
        )

        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 4.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (12, 10))

        flags = _WINDOW_RESIZABLE | _WINDOW_NO_COLLAPSE

        if imgui.begin("Operations", flags=flags):
            # Get selected tensor and active mode
            selected = get_selected_tensor(state) if state else None
            active_mode = state.active_mode if state else "vectors"

            # Panel header
            self._render_header(selected, width)

            imgui.separator()
            imgui.spacing()

            # Render different content based on active mode
            if active_mode == "visualize":
                # View mode - show visualization settings
                self._render_view_mode(state, dispatch, width)
            elif active_mode == "settings":
                # Settings mode - show app settings
                self._render_settings_mode(state, dispatch, width)
            else:
                # Tensor mode (rank/shape-driven)
                self._render_tensor_mode(selected, state, dispatch, width)

        imgui.end()
        imgui.pop_style_var(2)

    def _render_header(self, selected, width: float):
        """Render the panel header."""
        imgui.text("Operations")
        imgui.same_line()

        if selected:
            type_colors = {
                "r1": (0.2, 0.6, 0.8, 1.0),
                "r2": (0.8, 0.6, 0.2, 1.0),
                "r3": (0.7, 0.7, 0.7, 1.0),
            }
            if selected.rank == 1:
                rank_label = "RANK-1"
                color = type_colors["r1"]
            elif selected.rank == 2:
                rank_label = "RANK-2"
                color = type_colors["r2"]
            else:
                rank_label = f"RANK-{selected.rank}"
                color = type_colors["r3"]
            imgui.text_colored(f"[{rank_label}]", *color)
            imgui.same_line()
            imgui.text_disabled(f"({selected.label})")
            if selected.dtype in (TensorDType.IMAGE_RGB, TensorDType.IMAGE_GRAYSCALE):
                imgui.same_line()
                imgui.text_disabled(f"[{selected.dtype.value}]")
        else:
            imgui.text_disabled("(No selection)")

        # Options menu
        imgui.same_line(width - 60)
        if imgui.small_button("Options"):
            imgui.open_popup("##ops_options")

        if imgui.begin_popup("##ops_options"):
            _, self._show_preview = imgui.checkbox(
                "Show Preview", self._show_preview
            )
            imgui.separator()
            if imgui.menu_item("Reset Layout")[0]:
                self._show_preview = True
            imgui.end_popup()

    def _render_no_selection(self, width: float):
        """Render content when nothing is selected."""
        imgui.spacing()
        imgui.spacing()

        # Center the message
        text = "Select a tensor to see operations"
        text_width = imgui.calc_text_size(text)[0]
        imgui.set_cursor_pos_x((width - text_width) / 2)
        imgui.text_colored(text, 0.5, 0.5, 0.5, 1.0)

        imgui.spacing()
        imgui.spacing()

        # Hint text
        hint_width = width - 40
        imgui.set_cursor_pos_x(20)
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + hint_width)
        imgui.text_colored(
            "Create tensors using the Input panel on the left, "
            "then select one to apply operations.",
            0.4, 0.4, 0.4, 1.0
        )
        imgui.pop_text_wrap_pos()

        imgui.spacing()
        imgui.spacing()

    def _render_type_ops(self, tensor, state, dispatch, width: float):
        """Render type-specific operations based on tensor type."""
        if tensor and tensor.dtype in (TensorDType.IMAGE_RGB, TensorDType.IMAGE_GRAYSCALE):
            self.image_ops.render(tensor, state, dispatch, width)
        elif tensor and tensor.rank == 1:
            self.vector_ops.render(tensor, state, dispatch, width)
        elif tensor and tensor.rank == 2:
            self.matrix_ops.render(tensor, state, dispatch, width)
        else:
            imgui.text_colored(
                "Unknown tensor type",
                0.8, 0.4, 0.4, 1.0
            )

    def _render_view_mode(self, state, dispatch, width: float):
        """Render View mode content - visualization settings."""
        if imgui.begin_child(
            "##view_settings_content",
            0, 0,
            border=False,
            flags=_WINDOW_ALWAYS_VERTICAL_SCROLLBAR
        ):
            self.view_settings.render(state, dispatch, width - 30)
        imgui.end_child()

    def _render_settings_mode(self, state, dispatch, width: float):
        """Render Settings mode content - application settings."""
        if imgui.begin_child(
            "##settings_content",
            0, 0,
            border=False,
            flags=_WINDOW_ALWAYS_VERTICAL_SCROLLBAR
        ):
            imgui.spacing()

            # Rendering Settings
            expanded, _ = imgui.collapsing_header(
                "Rendering",
                imgui.TREE_NODE_DEFAULT_OPEN
            )
            if expanded:
                imgui.spacing()
                imgui.text_colored("Post-Processing", 0.7, 0.8, 0.9, 1.0)
                imgui.spacing()

                # Post-process toggle (if app reference available)
                imgui.text("HDR rendering with bloom and tonemapping")
                imgui.text_colored("is enabled by default.", 0.5, 0.5, 0.5, 1.0)
                imgui.spacing()

                imgui.text_colored("Infinite Grid", 0.7, 0.8, 0.9, 1.0)
                imgui.spacing()
                imgui.text("GPU-procedural infinite grid with")
                imgui.text("anti-aliased lines and distance fade.")
                imgui.spacing()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Performance Settings
            expanded, _ = imgui.collapsing_header(
                "Performance",
                imgui.TREE_NODE_DEFAULT_OPEN
            )
            if expanded:
                imgui.spacing()
                imgui.text("VSync: Enabled")
                imgui.text("MSAA: 8x")
                imgui.spacing()
                imgui.text_colored(
                    "Performance settings are optimized for",
                    0.5, 0.5, 0.5, 1.0
                )
                imgui.text_colored(
                    "visual quality and smooth rendering.",
                    0.5, 0.5, 0.5, 1.0
                )
                imgui.spacing()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Keyboard Shortcuts
            expanded, _ = imgui.collapsing_header("Keyboard Shortcuts", 0)
            if expanded:
                imgui.spacing()
                shortcuts = [
                    ("R", "Toggle auto-rotation"),
                    ("Space", "Reset camera"),
                    ("C", "Toggle grid mode"),
                    ("F", "Toggle cube faces"),
                    ("V", "Toggle vector components"),
                    ("1-4", "Switch view presets"),
                ]
                for key, desc in shortcuts:
                    imgui.text_colored(f"  {key}", 0.6, 0.8, 1.0, 1.0)
                    imgui.same_line(80)
                    imgui.text(desc)
                imgui.spacing()

            imgui.spacing()

        imgui.end_child()

    def _render_tensor_mode(self, selected, state, dispatch, width: float):
        """Render tensor operations using rank/shape logic."""
        if imgui.begin_child(
            "##tensor_ops_content",
            0, 0,
            border=False,
            flags=_WINDOW_ALWAYS_VERTICAL_SCROLLBAR
        ):
            if selected is None:
                self._render_no_selection(width)
                imgui.end_child()
                return

            self.tensor_info.render(selected, state, dispatch, width - 30)
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if selected.rank == 1:
                self.vector_ops.render(selected, state, dispatch, width - 30)
            elif selected.rank == 2:
                self.matrix_ops.render(selected, state, dispatch, width - 30)
                imgui.spacing()
                imgui.separator()
                imgui.spacing()
                self.linear_systems.render(state, dispatch, width - 30, selected)
            else:
                imgui.text_colored(
                    f"No rank-{selected.rank} ops available yet.",
                    0.6, 0.6, 0.6, 1.0
                )

            if selected.dtype in (TensorDType.IMAGE_RGB, TensorDType.IMAGE_GRAYSCALE):
                imgui.spacing()
                imgui.separator()
                imgui.spacing()
                self.image_ops.render(selected, state, dispatch, width - 30)

            if self._show_preview:
                imgui.spacing()
                imgui.separator()
                imgui.spacing()
                self.preview.render(selected, state, dispatch, width - 30)

        imgui.end_child()

    def _render_image_hint(self, width: float):
        """Render hint for Vision mode when no image is selected."""
        imgui.spacing()
        imgui.spacing()

        text = "Select a tensor with image dtype to see image ops"
        text_width = imgui.calc_text_size(text)[0]
        imgui.set_cursor_pos_x((width - text_width) / 2)
        imgui.text_colored(text, 0.5, 0.5, 0.5, 1.0)

        imgui.spacing()
        imgui.spacing()

        hint_width = width - 40
        imgui.set_cursor_pos_x(20)
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + hint_width)
        imgui.text_colored(
            "Use the Input panel to load an image file.",
            0.4, 0.4, 0.4, 1.0
        )
        imgui.pop_text_wrap_pos()
