"""
Sidebar vector creation section.
"""

import imgui


def _render_vector_creation(self, scene):
    """Render vector creation section."""
    if self._section("Create Vector", "➕"):
        imgui.text("Coordinates:")
        changed, self.vec_input = self._input_float3("vec_coords", self.vec_input)

        imgui.spacing()

        imgui.text("Label:")
        imgui.same_line()
        imgui.push_item_width(150)
        name_changed, self.vec_name = imgui.input_text("##vec_name", self.vec_name, 32)
        imgui.pop_item_width()

        if not self.vec_name:
            imgui.same_line()
            imgui.text_disabled("(Auto: v{})".format(self.next_vector_id))

        imgui.spacing()

        imgui.text("Color:")
        imgui.same_line()
        color_changed, self.vec_color = imgui.color_edit3("##vec_color",
                                                        *self.vec_color,
                                                        imgui.COLOR_EDIT_NO_INPUTS)

        imgui.spacing()
        imgui.spacing()

        if self._styled_button("Create Vector", (0.2, 0.6, 0.2, 1.0), width=-1):
            self._add_vector(scene)

        self._end_section()
