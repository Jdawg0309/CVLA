"""
Sidebar vector list section.
"""

import imgui


def _render_vector_list(self, scene, selected):
    """Render vector list with filtering."""
    if self._section("Vector List", "📋", default_open=True):
        imgui.push_item_width(-1)
        filter_changed, self.vector_list_filter = imgui.input_text_with_hint(
            "##vector_filter", "Filter vectors...", self.vector_list_filter, 64
        )
        imgui.pop_item_width()

        imgui.spacing()

        imgui.begin_child("##vector_list", 0, 200, border=True)

        filtered_vectors = scene.vectors
        if self.vector_list_filter:
            filter_lower = self.vector_list_filter.lower()
            filtered_vectors = [
                v for v in scene.vectors
                if filter_lower in v.label.lower() or
                any(str(coord) for coord in v.coords if filter_lower in str(coord))
            ]

        if not filtered_vectors:
            imgui.text_disabled("No vectors match filter")
        else:
            for i, vector in enumerate(filtered_vectors):
                is_selected = (vector is selected)

                coords_str = f"({vector.coords[0]:.2f}, {vector.coords[1]:.2f}, {vector.coords[2]:.2f})"
                label_text = f"{vector.label} {coords_str}"

                imgui.push_style_color(imgui.COLOR_TEXT, *vector.color)
                if imgui.selectable(f"##vec_{i}", is_selected)[0]:
                    selected = vector
                    scene.selected_object = vector
                    scene.selection_type = 'vector'

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
                        self._duplicate_vector(scene, vector)

                    if imgui.menu_item("Delete")[0]:
                        if vector is selected:
                            selected = None
                        scene.remove_vector(vector)

                    imgui.end_popup()

        imgui.end_child()

        imgui.spacing()
        imgui.columns(2, "##list_actions", border=False)

        if imgui.button("Clear All", width=-1):
            scene.clear_vectors()
            selected = None

        imgui.next_column()

        if imgui.button("Export...", width=-1):
            self.show_export_dialog = True

        imgui.columns(1)

        self._end_section()

    return selected
