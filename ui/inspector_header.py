"""
Inspector header rendering.
"""

import imgui


def _render_header(self, vector):
    """Render inspector header."""
    draw_list = imgui.get_window_draw_list()
    pos = imgui.get_cursor_screen_pos()
    draw_list.add_circle_filled(
        pos.x + 15, pos.y + 15,
        10, imgui.get_color_u32_rgba(*vector.color, 1.0)
    )

    imgui.dummy(30, 0)
    imgui.same_line()

    imgui.push_font()
    imgui.text_colored(vector.label, 0.9, 0.9, 1.0, 1.0)
    imgui.pop_font()

    imgui.same_line()
    imgui.text_disabled("(Vector)")

    imgui.text_disabled("3D Position Vector")

    imgui.same_line(200)
    changed, vector.visible = imgui.checkbox("Visible", vector.visible)
