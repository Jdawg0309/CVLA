"""
Images Tab UI Component - State-Driven Architecture.
"""

import imgui
from typing import Callable

from state import (
    AppState,
    Action,
    ApplyKernel,
    ApplyTransform,
    ClearImage,
    CreateSampleImage,
    FlipImageHorizontal,
    LoadImage,
    NormalizeImage,
    SetActiveImageTab,
    SetImageColorMode,
    SetImageNormalizeMean,
    SetImageNormalizeStd,
    SetImagePath,
    SetImagePreviewResolution,
    SetImageRenderMode,
    SetImageRenderScale,
    SetSamplePattern,
    SetSampleSize,
    SetSelectedKernel,
    SetTransformRotation,
    SetTransformScale,
    ToggleImageDownsample,
    ToggleImageGridOverlay,
    ToggleImageOnGrid,
    ToggleMatrixValues,
    UseResultAsInput,
)

try:
    from domain.images import list_kernels, get_kernel_by_name  # noqa: F401
    _VISION_AVAILABLE = True
except Exception:
    _VISION_AVAILABLE = False

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
    """
    if not _VISION_AVAILABLE:
        imgui.text_colored("Vision module not available", 0.8, 0.4, 0.4, 1.0)
        imgui.text_disabled("Install Pillow: pip install Pillow")
        return

    if state.image_status:
        if state.image_status_level == "error":
            color = (0.9, 0.3, 0.3, 1.0)
        else:
            color = (0.6, 0.8, 0.6, 1.0)
        imgui.text_colored(state.image_status, *color)
        imgui.spacing()

    _render_image_source_section(state, dispatch)

    if state.current_image is None:
        return

    _render_preprocess_tab(state, dispatch)
    _render_educational_section(state)


def _render_image_source_section(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render image loading/creation controls."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.15, 0.15, 0.18, 0.8)

    if imgui.collapsing_header("Image Source", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        imgui.indent(10)

        imgui.text("Create Sample Image:")
        imgui.spacing()

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

        imgui.push_item_width(100)
        changed, new_size = imgui.slider_int("Size##sample", state.input_sample_size, 8, 128)
        if changed:
            dispatch(SetSampleSize(size=new_size))
        imgui.pop_item_width()

        if imgui.button("Create Sample", width=260):
            dispatch(CreateSampleImage(
                pattern=state.input_sample_pattern,
                size=state.input_sample_size
            ))

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

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


def _render_raw_tab(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render the raw image tab with color mode controls."""
    _render_color_mode_selector(state, dispatch)
    _render_image_render_options(state, dispatch)
    _render_image_info_section(state, dispatch)


def _render_image_tab_selector(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render the button group that switches between Raw / Preprocess."""
    tabs = [("Raw Image", "raw"), ("Preprocess", "preprocess")]
    imgui.begin_group()
    for idx, (label, tab_id) in enumerate(tabs):
        active = (state.active_image_tab == tab_id)
        if active:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.45, 0.75, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.28, 0.55, 0.85, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.2, 0.4, 0.7, 1.0)
        if imgui.button(label, 140, 28):
            dispatch(SetActiveImageTab(tab=tab_id))
        if active:
            imgui.pop_style_color(3)
        if idx < len(tabs) - 1:
            imgui.same_line()
    imgui.end_group()
    imgui.separator()


def _render_preprocess_tab(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render preprocessing controls (normalize, convolution, etc.)."""
    _render_normalization_section(state, dispatch)

    _render_convolution_section(state, dispatch)
    _render_transform_section(state, dispatch)

    if state.processed_image is not None:
        _render_result_section(state, dispatch)


def _render_color_mode_selector(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render controls to switch between RGB, grayscale, and heatmap mode."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.16, 0.16, 0.19, 0.75)
    if imgui.collapsing_header("Image Color Mode", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        imgui.indent(10)
        imgui.text("Preview processed images as:")
        imgui.spacing()
        modes = [("RGB (default)", "rgb"), ("Grayscale", "grayscale"), ("Heatmap", "heatmap")]
        for idx, (label_text, mode_id) in enumerate(modes):
            selected = (state.image_color_mode == mode_id)
            if imgui.radio_button(label_text, selected):
                dispatch(SetImageColorMode(mode=mode_id))
            if idx < len(modes) - 1:
                imgui.same_line()
        imgui.unindent(10)
        imgui.spacing()
    imgui.pop_style_color()


def _render_normalization_section(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render normalization inputs and action."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.16, 0.16, 0.19, 0.75)
    if imgui.collapsing_header("Normalize Image", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        imgui.indent(10)
        imgui.text("Standardize the raw image before downstream kernels.")

        changed, new_mean = imgui.input_float(
            "Mean",
            state.input_image_normalize_mean,
            format="%.3f"
        )
        if changed:
            dispatch(SetImageNormalizeMean(mean=new_mean))

        imgui.same_line()
        changed, new_std = imgui.input_float(
            "Std Dev",
            state.input_image_normalize_std,
            format="%.3f"
        )
        if changed:
            dispatch(SetImageNormalizeStd(std=new_std))

        imgui.spacing()
        if imgui.button("Normalize Raw Image", width=-1):
            dispatch(NormalizeImage(
                mean=state.input_image_normalize_mean,
                std=state.input_image_normalize_std,
            ))

        imgui.unindent(10)
        imgui.spacing()
    imgui.pop_style_color()


def _render_image_render_options(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render image render options (mode, scale, and overlays)."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.16, 0.16, 0.19, 0.75)
    if imgui.collapsing_header("Render Options", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        imgui.indent(10)

        imgui.text("Render Mode:")
        imgui.spacing()
        modes = [("Plane", "plane"), ("Height Field", "height-field")]
        for idx, (label, mode_id) in enumerate(modes):
            selected = (state.image_render_mode == mode_id)
            if imgui.radio_button(label, selected):
                dispatch(SetImageRenderMode(mode=mode_id))
            if idx < len(modes) - 1:
                imgui.same_line()

        imgui.spacing()
        imgui.text("Render Scale:")
        imgui.push_item_width(180)
        changed, new_scale = imgui.slider_float(
            "##image_render_scale",
            state.image_render_scale,
            0.1,
            5.0,
            format="%.2f",
        )
        imgui.pop_item_width()
        if changed:
            dispatch(SetImageRenderScale(scale=new_scale))

        imgui.spacing()
        changed, show_on_grid = imgui.checkbox("Show Image on Grid", state.show_image_on_grid)
        if changed:
            dispatch(ToggleImageOnGrid())

        changed, overlay = imgui.checkbox("Pixel Grid Overlay", state.show_image_grid_overlay)
        if changed:
            dispatch(ToggleImageGridOverlay())

        imgui.spacing()
        changed, downsample = imgui.checkbox("Downsample on Load", state.image_downsample_enabled)
        if changed:
            dispatch(ToggleImageDownsample())

        imgui.same_line()
        imgui.push_item_width(120)
        preview_changed, preview_size = imgui.slider_int(
            "Preview Size",
            state.image_preview_resolution,
            32,
            512,
        )
        imgui.pop_item_width()
        if preview_changed:
            dispatch(SetImagePreviewResolution(size=preview_size))

        imgui.unindent(10)
        imgui.spacing()
    imgui.pop_style_color()


def _render_image_info_section(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render current image information."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.15, 0.18, 0.15, 0.8)

    if imgui.collapsing_header("Image as Matrix", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        imgui.indent(10)

        img = state.current_image
        if img is None:
            imgui.text("No image loaded")
            imgui.unindent(10)
            imgui.spacing()
            imgui.pop_style_color()
            return

        imgui.text_colored(img.name, 0.4, 0.8, 1.0, 1.0)
        imgui.text(f"Shape: {img.height} x {img.width}")
        imgui.text(f"Type: {'Grayscale' if img.is_grayscale else 'RGB'}")

        stats = state.current_image_stats
        if stats is None:
            imgui.text("Mean: N/A  Std: N/A")
            imgui.text("Min: N/A  Max: N/A")
        else:
            mean, std, min_val, max_val = stats
            imgui.text(f"Mean: {mean:.3f}  Std: {std:.3f}")
            imgui.text(f"Min: {min_val:.3f}  Max: {max_val:.3f}")

        imgui.spacing()
        selected = state.selected_pixel
        if selected is not None:
            row, col = selected
            image_for_pixel = img
            if state.active_image_tab != "raw" and state.processed_image is not None:
                image_for_pixel = state.processed_image
            if 0 <= row < image_for_pixel.height and 0 <= col < image_for_pixel.width:
                pixels = image_for_pixel.pixels
                if image_for_pixel.is_grayscale:
                    if len(pixels.shape) == 2:
                        value = float(pixels[row, col])
                    else:
                        value = float(pixels[row, col, 0])
                    imgui.text(f"Selected Pixel: ({row}, {col}) = {value:.3f}")
                else:
                    rgb = pixels[row, col]
                    imgui.text(
                        f"Selected Pixel: ({row}, {col}) = "
                        f"({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})"
                    )
            else:
                imgui.text("Selected Pixel: (out of bounds)")
        else:
            imgui.text("Selected Pixel: (none)")

        imgui.spacing()

        if imgui.checkbox("Show Matrix Values", state.show_matrix_values)[1] != state.show_matrix_values:
            dispatch(ToggleMatrixValues())

        if state.show_matrix_values:
            imgui.begin_child("##matrix_view", 0, 120, border=True)
            preview = state.current_image_preview
            if preview:
                rows = len(preview)
                cols = len(preview[0]) if rows else 0
                imgui.text_disabled(f"Preview ({rows}x{cols}):")
                for row in preview:
                    row_str = " ".join(f"{value:.2f}" for value in row)
                    imgui.text(row_str)
                if img.height > rows or img.width > cols:
                    imgui.text_disabled("...")
            else:
                imgui.text_disabled("Preview unavailable")
            imgui.end_child()

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

        kernel = state.selected_kernel_matrix
        if kernel:
            imgui.text("Kernel Matrix:")
            imgui.begin_child("##kernel_view", 0, 70, border=True)
            for row in kernel:
                row_str = "  ".join(f"{v:>6.2f}" for v in row)
                imgui.text(row_str)
            imgui.end_child()
        else:
            imgui.text_disabled("Kernel matrix unavailable")

        imgui.spacing()

        if imgui.button("Apply Convolution", width=260):
            dispatch(ApplyKernel(kernel_name=state.selected_kernel))

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

        imgui.push_item_width(200)
        changed, new_rot = imgui.slider_float(
            "Rotation##transform",
            state.input_transform_rotation,
            -180, 180, "%.1f deg"
        )
        if changed:
            dispatch(SetTransformRotation(rotation=new_rot))
        imgui.pop_item_width()

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

        if imgui.button("Apply Transform", width=260):
            dispatch(ApplyTransform(
                rotation=state.input_transform_rotation,
                scale=state.input_transform_scale
            ))

        imgui.spacing()

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
        if result is None:
            imgui.text("No processed image")
            imgui.unindent(10)
            imgui.spacing()
            imgui.pop_style_color()
            return

        imgui.text_colored(f"Output: {result.name}", 0.4, 1.0, 0.4, 1.0)
        imgui.text(f"Shape: {result.height} x {result.width}")

        if result.history:
            imgui.text("Operations:")
            for op in result.history:
                imgui.bullet_text(op)

        imgui.spacing()

        if imgui.button("Use as Input", width=260):
            dispatch(UseResultAsInput())

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
