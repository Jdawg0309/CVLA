"""
Rank-based tensor operations widgets for the operations panel.

Consolidates vector, matrix, and image operation UIs without behavior changes.
"""

import imgui
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from state.app_state import AppState
    from state.models.tensor_model import TensorData

from state.actions.tensor_actions import ApplyOperation, DuplicateTensor, PreviewOperation
from state.actions.matrix_actions import ToggleMatrixPlot
from state.models.tensor_model import TensorDType
from state.selectors import get_matrices, get_vectors


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
