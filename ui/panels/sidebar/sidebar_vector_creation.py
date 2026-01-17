"""
Sidebar vector creation section.

This module handles the UI for creating new vectors.
Uses dispatch to add vectors to the Redux store.
"""

import imgui
from state.actions import AddVector, SetInputVector


def _render_vector_creation(self):
    """
    Render vector creation section.

    Uses AppState inputs and dispatches AddVector on button click.
    """
    if self._section("Create Vector", "➕"):
        if self._state is None or self._dispatch is None:
            imgui.text_disabled("Vector creation unavailable (no state).")
            self._end_section()
            return

        coords = list(self._state.input_vector_coords)
        label = self._state.input_vector_label
        color = self._state.input_vector_color

        imgui.text("Coordinates:")
        changed, coords = self._input_float_list("vec_coords", coords)
        if changed:
            self._dispatch(SetInputVector(coords=tuple(coords)))

        imgui.spacing()

        imgui.text("Dimension:")
        imgui.same_line()
        imgui.push_item_width(80)
        dim_changed, new_dim = imgui.input_int("##vec_dim", len(coords))
        imgui.pop_item_width()
        if dim_changed:
            target = max(1, min(int(new_dim), 8))
            if target != len(coords):
                if target > len(coords):
                    coords.extend([0.0] * (target - len(coords)))
                else:
                    coords = coords[:target]
                self._dispatch(SetInputVector(coords=tuple(coords)))

        imgui.text("Label:")
        imgui.same_line()
        imgui.push_item_width(150)
        name_changed, label = imgui.input_text("##vec_name", label, 32)
        imgui.pop_item_width()
        if name_changed:
            self._dispatch(SetInputVector(label=label))

        # Show auto-generated label hint
        if not label:
            imgui.same_line()
            imgui.text_disabled("(Auto: v{})".format(self._state.next_vector_id))

        imgui.spacing()

        imgui.text("Color:")
        imgui.same_line()
        color_changed, color = imgui.color_edit3(
            "##vec_color",
            *color,
            imgui.COLOR_EDIT_NO_INPUTS
        )
        if color_changed:
            self._dispatch(SetInputVector(color=color))

        imgui.spacing()
        imgui.spacing()

        if self._styled_button("Create Vector", (0.2, 0.6, 0.2, 1.0), width=-1):
            # Dispatch AddVector action
            self._dispatch(AddVector(
                coords=tuple(coords),
                color=color,
                label=label
            ))

        self._end_section()
