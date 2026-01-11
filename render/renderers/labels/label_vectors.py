"""
Vector label rendering helpers.
"""

import imgui


def draw_vector_labels(self, camera, vectors, width, height, selected_vector=None):
    """Draw labels for vectors."""
    if not self.viewconfig or not self.viewconfig.show_labels:
        return

    dl = imgui.get_background_draw_list()

    for vector in vectors:
        if not vector.visible or not vector.label:
            continue

        x, y, depth = self.world_to_screen(camera, vector.coords, width, height)

        if x is not None and depth > -0.5:
            label = vector.label
            offset_x, offset_y = self.label_offsets['vector']

            if vector is selected_vector:
                r, g, b = vector.color
                color = (min(1.0, r * 1.5), min(1.0, g * 1.5), min(1.0, b * 1.5), 1.0)
            else:
                color = (0.95, 0.95, 0.95, 1.0)

            text_size = imgui.calc_text_size(label)
            padding_x = 6
            padding_y = 4

            bg_color = imgui.get_color_u32_rgba(*self.background_color)
            dl.add_rect_filled(
                x + offset_x - padding_x,
                y + offset_y - padding_y,
                x + offset_x + text_size.x + padding_x,
                y + offset_y + text_size.y + padding_y,
                bg_color, 4.0
            )

            shadow_color = imgui.get_color_u32_rgba(*self.shadow_color)
            dl.add_text(x + offset_x + 2, y + offset_y + 2, shadow_color, label)

            text_color = imgui.get_color_u32_rgba(*color)
            dl.add_text(x + offset_x, y + offset_y, text_color, label)
