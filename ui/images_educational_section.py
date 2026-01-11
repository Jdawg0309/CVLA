"""
Images tab educational info section.
"""

import imgui

from state import AppState


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
