"""
Inspector coordinate editor.
"""

import imgui
import numpy as np
from state.actions import UpdateVector


def _render_coordinate_editor(self, vector, dispatch):
    """Render coordinate editor."""
    imgui.text_colored("Coordinates", 0.8, 0.8, 0.2, 1.0)
    imgui.spacing()

    x, y, z = vector.coords
    imgui.text(f"X: {x:.6f}")
    imgui.same_line(100)
    imgui.text(f"Y: {y:.6f}")
    imgui.same_line(200)
    imgui.text(f"Z: {z:.6f}")

    imgui.spacing()

    imgui.push_item_width(80)

    imgui.text("X:")
    imgui.same_line()
    x_changed, new_x = imgui.input_float("##edit_x", x, format="%.4f")
    if x_changed and dispatch:
        dispatch(UpdateVector(id=vector.id, coords=(new_x, y, z)))

    imgui.same_line(120)

    imgui.text("Y:")
    imgui.same_line()
    y_changed, new_y = imgui.input_float("##edit_y", y, format="%.4f")
    if y_changed and dispatch:
        dispatch(UpdateVector(id=vector.id, coords=(x, new_y, z)))

    imgui.same_line(240)

    imgui.text("Z:")
    imgui.same_line()
    z_changed, new_z = imgui.input_float("##edit_z", z, format="%.4f")
    if z_changed and dispatch:
        dispatch(UpdateVector(id=vector.id, coords=(x, y, new_z)))

    imgui.pop_item_width()

    imgui.spacing()
    imgui.columns(3, "##quick_edit", border=False)

    quick_edits = [
        ("Zero", [0, 0, 0], (0.5, 0.5, 0.5, 1.0)),
        ("Unit X", [1, 0, 0], (1.0, 0.3, 0.3, 1.0)),
        ("Unit Y", [0, 1, 0], (0.3, 1.0, 0.3, 1.0)),
        ("Unit Z", [0, 0, 1], (0.3, 0.5, 1.0, 1.0)),
        ("Normalize", None, (0.3, 0.3, 0.8, 1.0)),
        ("Reset", None, (0.8, 0.5, 0.2, 1.0))
    ]

    for i, (label, coords, color) in enumerate(quick_edits):
        if i > 0 and i % 3 == 0:
            imgui.next_column()

        imgui.push_style_color(imgui.COLOR_BUTTON, *color)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED,
                             color[0]*1.2, color[1]*1.2, color[2]*1.2, 1.0)

        if imgui.button(label, width=-1):
            if not dispatch:
                continue

            if coords is not None:
                dispatch(UpdateVector(id=vector.id, coords=tuple(coords)))
            elif label == "Normalize":
                vec = np.array(vector.coords, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    dispatch(UpdateVector(
                        id=vector.id,
                        coords=tuple((vec / norm).tolist()),
                    ))
            elif label == "Reset":
                dispatch(UpdateVector(id=vector.id, coords=(1.0, 0.0, 0.0)))

        imgui.pop_style_color(2)

        if i % 3 != 2:
            imgui.same_line()

    imgui.columns(1)
