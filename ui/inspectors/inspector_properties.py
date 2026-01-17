"""
Inspector properties section.
"""

import imgui
from state.actions import UpdateVector
from state.selectors import get_vector_magnitude


def _render_properties(self, vector, dispatch):
    """Render vector properties."""
    imgui.text_colored("Properties", 0.8, 0.8, 0.2, 1.0)
    imgui.spacing()

    magnitude = get_vector_magnitude(vector)
    imgui.text(f"Magnitude: {magnitude:.6f}")

    imgui.spacing()
    imgui.text("Color:")
    imgui.same_line()

    color_changed, new_color = imgui.color_edit3(
        "##vec_color_edit",
        *vector.color,
        imgui.COLOR_EDIT_NO_INPUTS
    )
    if color_changed and dispatch:
        dispatch(UpdateVector(id=vector.id, color=new_color))

    imgui.spacing()
    imgui.text("Label:")
    imgui.same_line()

    imgui.push_item_width(150)
    label_changed, new_label = imgui.input_text(
        "##vec_label_edit",
        vector.label,
        32
    )
    imgui.pop_item_width()

    if label_changed and dispatch:
        dispatch(UpdateVector(id=vector.id, label=new_label))

    if hasattr(vector, "metadata") and vector.metadata:
        imgui.spacing()
        imgui.text("Metadata:")
        for key, value in vector.metadata.items():
            imgui.text_disabled(f"  {key}: {value}")
