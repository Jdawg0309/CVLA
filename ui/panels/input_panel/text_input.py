"""
Text input widget for the input panel.

Provides a multiline text area for entering vectors and matrices.
"""

import imgui
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.app_state import AppState

from state.actions.input_panel_actions import (
    SetTextInput, ClearTextInput, CreateTensorFromTextInput
)
from state.selectors import get_next_color
from ui.panels.input_panel.input_parsers import (
    parse_input, get_type_description, get_shape_string
)


class TextInputWidget:
    """Widget for text-based tensor input."""

    def __init__(self):
        self._buffer = ""
        self._label_buffer = ""

    def render(self, state: "AppState", dispatch, width: float):
        """Render the text input widget."""
        imgui.text("Enter vector or matrix:")

        # Text input area
        imgui.push_item_width(width - 20)
        changed, new_text = imgui.input_text_multiline(
            "##text_input",
            state.input_text_content,
            2048,  # buffer size
            width - 20,
            120
        )
        imgui.pop_item_width()

        if changed:
            dispatch(SetTextInput(content=new_text))

        # Show parsed type and shape
        parsed_type = state.input_text_parsed_type
        if parsed_type:
            type_desc = get_type_description(parsed_type)
            parsed = parse_input(state.input_text_content)
            shape_str = get_shape_string(parsed_type, parsed[1])
            imgui.text_colored(
                f"Detected: {type_desc} {shape_str}",
                0.4, 0.8, 0.4, 1.0
            )
        elif state.input_text_content.strip():
            imgui.text_colored(
                "Could not parse input",
                0.8, 0.4, 0.4, 1.0
            )

        imgui.spacing()

        # Label input
        imgui.text("Label:")
        imgui.same_line()
        imgui.push_item_width(width - 80)
        changed, self._label_buffer = imgui.input_text(
            "##tensor_label",
            self._label_buffer,
            256
        )
        imgui.pop_item_width()

        imgui.spacing()

        # Action buttons
        can_create = parsed_type in ("vector", "matrix")

        if not can_create:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)

        if imgui.button("Create Tensor", width - 20, 30):
            if can_create:
                label = self._label_buffer.strip() or self._generate_label(state, parsed_type)
                color, _ = get_next_color(state)
                dispatch(CreateTensorFromTextInput(
                    label=label,
                    color=color
                ))
                self._label_buffer = ""

        if not can_create:
            imgui.pop_style_var()

        imgui.same_line()
        if imgui.button("Clear"):
            dispatch(ClearTextInput())
            self._label_buffer = ""

        # Help text
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        imgui.text_colored("Input formats:", 0.6, 0.6, 0.6, 1.0)
        imgui.text_colored("  Vector: 1, 2, 3", 0.5, 0.5, 0.5, 1.0)
        imgui.text_colored("  Matrix: 1, 2; 3, 4", 0.5, 0.5, 0.5, 1.0)
        imgui.text_colored("  Or multi-line entries", 0.5, 0.5, 0.5, 1.0)

    def _generate_label(self, state: "AppState", tensor_type: str) -> str:
        """Generate an automatic label for the tensor."""
        from state.selectors import get_tensor_count_by_type
        if tensor_type == "vector":
            count = get_tensor_count_by_type(state, "vector")
            return f"v{count + 1}"
        elif tensor_type == "matrix":
            count = get_tensor_count_by_type(state, "matrix")
            return f"M{count + 1}"
        return "tensor"
