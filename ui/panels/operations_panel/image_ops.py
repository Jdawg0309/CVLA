"""
Image operations widget for the operations panel.

Provides UI for image-specific operations (convolution, transforms, etc.).
"""

import imgui
from typing import TYPE_CHECKING, Optional, List, Tuple

if TYPE_CHECKING:
    from state.app_state import AppState
    from state.models.tensor_model import TensorData

from state.actions.tensor_actions import ApplyOperation, PreviewOperation


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
        if tensor is None or not tensor.is_image:
            return

        imgui.text("IMAGE OPERATIONS")
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
        self._render_convolution_section(tensor, state, dispatch, width)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Transform section
        self._render_transform_section(tensor, state, dispatch, width)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Preprocessing section
        self._render_preprocessing_section(tensor, state, dispatch, width)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Quick actions
        self._render_quick_actions(tensor, state, dispatch, width)

    def _render_convolution_section(self, tensor, state, dispatch, width):
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
                create_new=True
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
                create_new=True
            ))
        imgui.same_line()
        if imgui.button("Blur", half_width, 22):
            dispatch(ApplyOperation(
                operation_name="apply_kernel",
                parameters=(("kernel", "gaussian_blur"),),
                target_ids=(tensor.id,),
                create_new=True
            ))

        if imgui.button("Sharpen", half_width, 22):
            dispatch(ApplyOperation(
                operation_name="apply_kernel",
                parameters=(("kernel", "sharpen"),),
                target_ids=(tensor.id,),
                create_new=True
            ))
        imgui.same_line()
        if imgui.button("Edges", half_width, 22):
            dispatch(ApplyOperation(
                operation_name="apply_kernel",
                parameters=(("kernel", "edge_detect"),),
                target_ids=(tensor.id,),
                create_new=True
            ))

    def _render_transform_section(self, tensor, state, dispatch, width):
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
                create_new=True
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
                create_new=True
            ))

        imgui.spacing()

        # Flip buttons
        half_width = (width - 30) / 2
        if imgui.button("Flip H", half_width, 22):
            dispatch(ApplyOperation(
                operation_name="flip_horizontal",
                parameters=(),
                target_ids=(tensor.id,),
                create_new=True
            ))
        imgui.same_line()
        if imgui.button("Flip V", half_width, 22):
            dispatch(ApplyOperation(
                operation_name="flip_vertical",
                parameters=(),
                target_ids=(tensor.id,),
                create_new=True
            ))

    def _render_preprocessing_section(self, tensor, state, dispatch, width):
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
                create_new=True
            ))

        imgui.spacing()

        # Color mode conversion
        half_width = (width - 30) / 2
        if imgui.button("To Grayscale", half_width, 22):
            dispatch(ApplyOperation(
                operation_name="to_grayscale",
                parameters=(),
                target_ids=(tensor.id,),
                create_new=True
            ))
        imgui.same_line()
        if imgui.button("Invert", half_width, 22):
            dispatch(ApplyOperation(
                operation_name="invert",
                parameters=(),
                target_ids=(tensor.id,),
                create_new=True
            ))

    def _render_quick_actions(self, tensor, state, dispatch, width):
        """Render quick action buttons."""
        imgui.text("Quick Actions:")
        imgui.spacing()

        half_width = (width - 30) / 2

        if imgui.button("Duplicate", half_width, 25):
            from state.actions.tensor_actions import DuplicateTensor
            dispatch(DuplicateTensor(id=tensor.id))

        imgui.same_line()

        if imgui.button("Reset", half_width, 25):
            # Reset to original (undo all operations)
            pass  # TODO: implement reset from history

        imgui.spacing()

        if imgui.button("Export as Matrix", width - 20, 25):
            dispatch(ApplyOperation(
                operation_name="to_matrix",
                parameters=(),
                target_ids=(tensor.id,),
                create_new=True
            ))
