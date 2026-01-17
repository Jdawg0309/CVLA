"""
Inspector coordinate editor.
"""

import imgui
from state.actions import UpdateVector


def _render_coordinate_editor(self, vector, dispatch):
    """Render coordinate editor."""
    imgui.text_colored("Coordinates", 0.8, 0.8, 0.2, 1.0)
    imgui.spacing()

    coords = list(vector.coords)
    for idx, value in enumerate(coords):
        label = f"C{idx+1}" if idx > 2 else ["X", "Y", "Z"][idx]
        imgui.text(f"{label}: {value:.6f}")
        if idx % 3 != 2 and idx != len(coords) - 1:
            imgui.same_line(100 + (idx % 3) * 100)

    imgui.spacing()

    imgui.push_item_width(80)

    for idx, value in enumerate(coords):
        label = f"C{idx+1}" if idx > 2 else ["X", "Y", "Z"][idx]
        imgui.text(f"{label}:")
        imgui.same_line()
        changed, new_val = imgui.input_float(f"##edit_{idx}", value, format="%.4f")
        if changed and dispatch:
            coords[idx] = new_val
            dispatch(UpdateVector(id=vector.id, coords=tuple(coords)))
        if idx % 3 != 2 and idx != len(coords) - 1:
            imgui.same_line(120 + (idx % 3) * 120)

    imgui.pop_item_width()

    imgui.spacing()
    imgui.columns(2, "##dim_controls", border=False)
    if imgui.button("+ Component", width=-1) and dispatch:
        coords.append(0.0)
        dispatch(UpdateVector(id=vector.id, coords=tuple(coords)))
    imgui.next_column()
    if imgui.button("- Component", width=-1) and dispatch:
        if len(coords) > 1:
            coords = coords[:-1]
            dispatch(UpdateVector(id=vector.id, coords=tuple(coords)))
    imgui.columns(1)
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
                norm_sq = sum(val * val for val in vector.coords)
                if norm_sq > 1e-10:
                    norm = norm_sq ** 0.5
                    scaled = tuple(val / norm for val in vector.coords)
                    dispatch(UpdateVector(id=vector.id, coords=scaled))
            elif label == "Reset":
                size = max(1, len(vector.coords))
                reset = [0.0] * size
                reset[0] = 1.0
                dispatch(UpdateVector(id=vector.id, coords=tuple(reset)))

        imgui.pop_style_color(2)

        if i % 3 != 2:
            imgui.same_line()

    imgui.columns(1)
