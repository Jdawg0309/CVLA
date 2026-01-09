"""
Images Tab UI Component - State-Driven Architecture

This component demonstrates the correct UI pattern:
- READS from AppState (never modifies directly)
- DISPATCHES actions to request changes
- Uses CONTROLLED inputs (state owns values)

RULES:
1. All values displayed come from state
2. All changes dispatch actions
3. No local mutable state
"""

import imgui
import numpy as np
from typing import Callable

from state import (
    AppState,
    Action,
    # Image actions
    CreateSampleImage, ApplyKernel, ApplyTransform,
    FlipImageHorizontal, UseResultAsInput, ClearImage, LoadImage,
    # Input actions
    SetImagePath, SetSamplePattern, SetSampleSize,
    SetTransformRotation, SetTransformScale, SetSelectedKernel,
    ToggleMatrixValues,
    # Pipeline actions
    StepForward, StepBackward, JumpToStep, ResetPipeline,
    # Queries
    get_current_step,
)


# Available kernels (static, not state)
KERNEL_OPTIONS = [
    ('sobel_x', 'Sobel X - Vertical edges'),
    ('sobel_y', 'Sobel Y - Horizontal edges'),
    ('laplacian', 'Laplacian - All edges'),
    ('gaussian_blur', 'Gaussian Blur - Smoothing'),
    ('sharpen', 'Sharpen - Enhance edges'),
    ('edge_detect', 'Edge Detect - Strong edges'),
    ('box_blur', 'Box Blur - Simple average'),
    ('emboss', 'Emboss - 3D effect'),
]

PATTERN_OPTIONS = ['gradient', 'checkerboard', 'circle', 'edges', 'noise', 'rgb_gradient']


def render_images_tab(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """
    Render the Images tab.

    Args:
        state: Current AppState (read-only)
        dispatch: Function to dispatch actions
    """
    # Check if vision module is available
    try:
        from vision import list_kernels, get_kernel_by_name
        vision_available = True
    except ImportError:
        vision_available = False

    if not vision_available:
        imgui.text_colored("Vision module not available", 0.8, 0.4, 0.4, 1.0)
        imgui.text_disabled("Install Pillow: pip install Pillow")
        return

    # =========================================================================
    # SECTION 1: IMAGE SOURCE
    # =========================================================================
    _render_image_source_section(state, dispatch)

    # =========================================================================
    # SECTION 2: CURRENT IMAGE INFO (if loaded)
    # =========================================================================
    if state.current_image is not None:
        _render_image_info_section(state, dispatch)

    # =========================================================================
    # SECTION 3: CONVOLUTION (if image loaded)
    # =========================================================================
    if state.current_image is not None:
        _render_convolution_section(state, dispatch)

    # =========================================================================
    # SECTION 4: TRANSFORMS (if image loaded)
    # =========================================================================
    if state.current_image is not None:
        _render_transform_section(state, dispatch)

    # =========================================================================
    # SECTION 5: RESULT (if processed)
    # =========================================================================
    if state.processed_image is not None:
        _render_result_section(state, dispatch)

    # =========================================================================
    # SECTION 6: PIPELINE STEPS
    # =========================================================================
    if state.pipeline_steps:
        _render_pipeline_section(state, dispatch)

    # =========================================================================
    # SECTION 7: EDUCATIONAL INFO
    # =========================================================================
    _render_educational_section(state)


def _render_image_source_section(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render image loading/creation controls."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.15, 0.15, 0.18, 0.8)

    if imgui.collapsing_header("Image Source", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        imgui.indent(10)

        # --- Sample Image Creation ---
        imgui.text("Create Sample Image:")
        imgui.spacing()

        # Pattern dropdown (controlled by state)
        imgui.push_item_width(150)
        if imgui.begin_combo("##pattern", state.input_sample_pattern):
            for pattern in PATTERN_OPTIONS:
                is_selected = (pattern == state.input_sample_pattern)
                if imgui.selectable(pattern, is_selected)[0]:
                    dispatch(SetSamplePattern(pattern=pattern))
                if is_selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        imgui.pop_item_width()

        imgui.same_line()

        # Size slider (controlled by state)
        imgui.push_item_width(100)
        changed, new_size = imgui.slider_int("Size##sample", state.input_sample_size, 8, 128)
        if changed:
            dispatch(SetSampleSize(size=new_size))
        imgui.pop_item_width()

        # Create button
        if imgui.button("Create Sample", width=260):
            dispatch(CreateSampleImage(
                pattern=state.input_sample_pattern,
                size=state.input_sample_size
            ))

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # --- Load from File ---
        imgui.text("Load from File:")

        imgui.push_item_width(260)
        changed, new_path = imgui.input_text_with_hint(
            "##imgpath",
            "Path to image (PNG/JPG)...",
            state.input_image_path,
            256
        )
        if changed:
            dispatch(SetImagePath(path=new_path))
        imgui.pop_item_width()

        # Load button (disabled if no path)
        can_load = len(state.input_image_path.strip()) > 0
        if not can_load:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
        if imgui.button("Load Image", width=260) and can_load:
            dispatch(LoadImage(path=state.input_image_path))
        if not can_load:
            imgui.pop_style_var()

        imgui.unindent(10)
        imgui.spacing()

    imgui.pop_style_color()


def _render_image_info_section(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render current image information."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.15, 0.18, 0.15, 0.8)

    if imgui.collapsing_header("Image as Matrix", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        imgui.indent(10)

        img = state.current_image

        # Image name and shape
        imgui.text_colored(img.name, 0.4, 0.8, 1.0, 1.0)
        imgui.text(f"Shape: {img.height} x {img.width}")
        imgui.text(f"Type: {'Grayscale' if img.is_grayscale else 'RGB'}")

        # Statistics
        matrix = img.as_matrix()
        imgui.text(f"Mean: {np.mean(matrix):.3f}  Std: {np.std(matrix):.3f}")
        imgui.text(f"Min: {np.min(matrix):.3f}  Max: {np.max(matrix):.3f}")

        imgui.spacing()

        # Toggle matrix values view
        if imgui.checkbox("Show Matrix Values", state.show_matrix_values)[1] != state.show_matrix_values:
            dispatch(ToggleMatrixValues())

        if state.show_matrix_values:
            imgui.begin_child("##matrix_view", 0, 120, border=True)
            # Show 8x8 preview
            rows = min(8, matrix.shape[0])
            cols = min(8, matrix.shape[1])
            imgui.text_disabled(f"Preview ({rows}x{cols}):")
            for i in range(rows):
                row_str = " ".join(f"{matrix[i, j]:.2f}" for j in range(cols))
                imgui.text(row_str)
            if matrix.shape[0] > 8 or matrix.shape[1] > 8:
                imgui.text_disabled("...")
            imgui.end_child()

        # Clear button
        imgui.spacing()
        if imgui.button("Clear Image", width=120):
            dispatch(ClearImage())

        imgui.unindent(10)
        imgui.spacing()

    imgui.pop_style_color()


def _render_convolution_section(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render convolution controls."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.18, 0.15, 0.15, 0.8)

    if imgui.collapsing_header("Convolution (CNN Core)", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        imgui.indent(10)

        imgui.text_colored("Kernels detect features", 0.8, 0.8, 0.4, 1.0)
        imgui.text_disabled("Select a kernel and apply:")
        imgui.spacing()

        # Kernel dropdown (controlled by state)
        imgui.push_item_width(260)
        if imgui.begin_combo("##kernel", state.selected_kernel):
            for name, desc in KERNEL_OPTIONS:
                is_selected = (name == state.selected_kernel)
                label = f"{name}: {desc}"
                if imgui.selectable(label, is_selected)[0]:
                    dispatch(SetSelectedKernel(kernel_name=name))
                if is_selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        imgui.pop_item_width()

        imgui.spacing()

        # Show kernel matrix
        try:
            from vision import get_kernel_by_name
            kernel = get_kernel_by_name(state.selected_kernel)
            imgui.text("Kernel Matrix:")
            imgui.begin_child("##kernel_view", 0, 70, border=True)
            for row in kernel:
                row_str = "  ".join(f"{v:>6.2f}" for v in row)
                imgui.text(row_str)
            imgui.end_child()
        except Exception:
            pass

        imgui.spacing()

        # Apply button
        if imgui.button("Apply Convolution", width=260):
            dispatch(ApplyKernel(kernel_name=state.selected_kernel))

        # Quick buttons
        imgui.spacing()
        imgui.text("Quick Apply:")
        imgui.columns(2, "##quick", border=False)

        quick_kernels = ['sobel_x', 'sobel_y', 'gaussian_blur', 'sharpen']
        for i, k in enumerate(quick_kernels):
            if imgui.button(k.replace('_', ' ').title(), width=125):
                dispatch(SetSelectedKernel(kernel_name=k))
                dispatch(ApplyKernel(kernel_name=k))
            if i % 2 == 0:
                imgui.next_column()

        imgui.columns(1)

        imgui.unindent(10)
        imgui.spacing()

    imgui.pop_style_color()


def _render_transform_section(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render affine transform controls."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.15, 0.15, 0.18, 0.8)

    if imgui.collapsing_header("Affine Transforms")[0]:
        imgui.indent(10)

        imgui.text_colored("Linear transforms on coordinates", 0.4, 0.8, 0.4, 1.0)
        imgui.spacing()

        # Rotation slider (controlled by state)
        imgui.push_item_width(200)
        changed, new_rot = imgui.slider_float(
            "Rotation##transform",
            state.input_transform_rotation,
            -180, 180, "%.1f deg"
        )
        if changed:
            dispatch(SetTransformRotation(rotation=new_rot))
        imgui.pop_item_width()

        # Scale slider (controlled by state)
        imgui.push_item_width(200)
        changed, new_scale = imgui.slider_float(
            "Scale##transform",
            state.input_transform_scale,
            0.5, 2.0, "%.2fx"
        )
        if changed:
            dispatch(SetTransformScale(scale=new_scale))
        imgui.pop_item_width()

        imgui.spacing()

        # Apply button
        if imgui.button("Apply Transform", width=260):
            dispatch(ApplyTransform(
                rotation=state.input_transform_rotation,
                scale=state.input_transform_scale
            ))

        imgui.spacing()

        # Flip button
        if imgui.button("Flip Horizontal", width=260):
            dispatch(FlipImageHorizontal())

        imgui.unindent(10)
        imgui.spacing()

    imgui.pop_style_color()


def _render_result_section(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render result/output section."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.15, 0.18, 0.15, 0.8)

    if imgui.collapsing_header("Result", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        imgui.indent(10)

        result = state.processed_image

        imgui.text_colored(f"Output: {result.name}", 0.4, 1.0, 0.4, 1.0)
        imgui.text(f"Shape: {result.height} x {result.width}")

        # Show history
        if result.history:
            imgui.text("Operations:")
            for op in result.history:
                imgui.bullet_text(op)

        imgui.spacing()

        # Use as input button
        if imgui.button("Use as Input", width=260):
            dispatch(UseResultAsInput())

        imgui.unindent(10)
        imgui.spacing()

    imgui.pop_style_color()


def _render_pipeline_section(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render educational pipeline steps."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.18, 0.15, 0.18, 0.8)

    if imgui.collapsing_header("Pipeline Steps", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        imgui.indent(10)

        # Step navigation
        total_steps = len(state.pipeline_steps)
        current_idx = state.pipeline_step_index

        imgui.text(f"Step {current_idx + 1} of {total_steps}")

        # Navigation buttons
        can_back = current_idx > 0
        can_forward = current_idx < total_steps - 1

        if not can_back:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
        if imgui.button("<< Prev") and can_back:
            dispatch(StepBackward())
        if not can_back:
            imgui.pop_style_var()

        imgui.same_line()

        if not can_forward:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
        if imgui.button("Next >>") and can_forward:
            dispatch(StepForward())
        if not can_forward:
            imgui.pop_style_var()

        imgui.same_line()
        if imgui.button("Reset"):
            dispatch(ResetPipeline())

        imgui.spacing()

        # Step list
        imgui.begin_child("##steps", 0, 120, border=True)
        for i, step in enumerate(state.pipeline_steps):
            is_current = (i == current_idx)

            if is_current:
                imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 0.4, 1.0)

            if imgui.selectable(f"{i+1}. {step.title}", is_current)[0]:
                dispatch(JumpToStep(index=i))

            if is_current:
                imgui.pop_style_color()

        imgui.end_child()

        # Current step explanation
        current_step = get_current_step(state)
        if current_step:
            imgui.spacing()
            imgui.text_wrapped(current_step.explanation)

        imgui.unindent(10)
        imgui.spacing()

    imgui.pop_style_color()


def _render_educational_section(state: AppState) -> None:
    """Render educational info panel."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.12, 0.12, 0.15, 0.8)

    if imgui.collapsing_header("ML/CV Info")[0]:
        imgui.indent(10)

        imgui.text_wrapped(
            "Images are matrices. Each pixel is a number (0-1). "
            "Convolution kernels slide over the image computing "
            "weighted sums. This is how CNNs detect edges, "
            "textures, and patterns. The kernels here are "
            "what neural networks learn from data."
        )
        imgui.spacing()
        imgui.text_disabled("Pipeline: Image -> Matrix -> Transform -> Result")

        imgui.unindent(10)

    imgui.pop_style_color()
