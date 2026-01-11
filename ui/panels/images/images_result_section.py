"""
Images tab result section.
"""

import imgui
from typing import Callable

from state import (
    AppState,
    Action,
    UseResultAsInput,
)


def _render_result_section(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render result/output section."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.15, 0.18, 0.15, 0.8)

    if imgui.collapsing_header("Result", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        imgui.indent(10)

        result = state.processed_image

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
