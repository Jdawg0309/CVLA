"""
Images tab info section.
"""

import imgui
from typing import Callable

from state import (
    AppState,
    Action,
    ToggleMatrixValues,
    ClearImage,
)


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
