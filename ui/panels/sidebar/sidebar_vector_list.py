"""
Sidebar vector list section.

This module displays the list of vectors and handles selection/deletion.
Reads from AppState.vectors and dispatches actions for state changes.
"""

import imgui
from state.actions import (
    SelectVector,
    DeleteVector,
    DuplicateVector,
    ClearAllVectors,
    DeselectVector,
)


def _render_vector_list(self):
    """
    Render vector list with filtering.

    Reads vectors from AppState and dispatches actions.
    """
    if self._section("Vector List", "📋", default_open=True):
        if self._state is None or self._dispatch is None:
            imgui.text_disabled("Vector list unavailable (no state).")
            self._end_section()
            return

        imgui.columns(2, "##vector_list_header", border=False)
        imgui.text_disabled("Manage vectors")
        imgui.next_column()
        if imgui.button("Options"):
            imgui.open_popup("##vector_list_options")
        imgui.columns(1)

        if imgui.begin_popup("##vector_list_options"):
            if imgui.menu_item("Deselect")[0]:
                self._dispatch(DeselectVector())
            if imgui.menu_item("Clear All")[0]:
                self._dispatch(ClearAllVectors())
            imgui.end_popup()

        imgui.push_item_width(-1)
        filter_changed, self.vector_list_filter = imgui.input_text_with_hint(
            "##vector_filter", "Filter vectors...", self.vector_list_filter, 64
        )
        imgui.pop_item_width()

        imgui.spacing()

        imgui.begin_child("##vector_list", 0, 200, border=True)

        all_vectors = list(self._state.vectors)
        selected_id = self._state.selected_id

        # Apply filter
        filtered_vectors = all_vectors
        if self.vector_list_filter:
            filter_lower = self.vector_list_filter.lower()
            filtered_vectors = [
                v for v in all_vectors
                if filter_lower in v.label.lower() or
                any(filter_lower in str(coord) for coord in v.coords)
            ]

        if not filtered_vectors:
            imgui.text_disabled("No vectors match filter")
        else:
            for i, vector in enumerate(filtered_vectors):
                is_selected = (vector.id == selected_id)

                coords = vector.coords
                if len(coords) <= 3:
                    coords_str = "(" + ", ".join(f"{c:.2f}" for c in coords) + ")"
                else:
                    prefix = ", ".join(f"{c:.2f}" for c in coords[:3])
                    coords_str = f"({prefix}, ...)"
                label_text = f"{vector.label} {coords_str}"

                imgui.push_style_color(imgui.COLOR_TEXT, *vector.color)
                if imgui.selectable(f"##vec_{i}", is_selected)[0]:
                    self._dispatch(SelectVector(id=vector.id))

                draw_list = imgui.get_window_draw_list()
                pos = imgui.get_cursor_screen_pos()
                draw_list.add_circle_filled(
                    pos.x - 20, pos.y - 10,
                    5, imgui.get_color_u32_rgba(*vector.color, 1.0)
                )

                imgui.same_line()
                imgui.text(label_text)
                imgui.pop_style_color(1)

                if imgui.begin_popup_context_item(f"vec_context_{i}"):
                    if imgui.menu_item("Duplicate")[0]:
                        self._dispatch(DuplicateVector(id=vector.id))

                    if imgui.menu_item("Delete")[0]:
                        self._dispatch(DeleteVector(id=vector.id))

                    imgui.end_popup()

        imgui.end_child()

        imgui.spacing()
        imgui.columns(2, "##list_actions", border=False)

        if imgui.button("Clear All", width=-1):
            self._dispatch(ClearAllVectors())

        imgui.next_column()

        if imgui.button("Export...", width=-1):
            self.show_export_dialog = True

        imgui.columns(1)

        self._end_section()
