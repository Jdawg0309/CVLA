"""
Grid label rendering helpers.
"""

import numpy as np
import imgui


def draw_grid_numbers(self, camera, width, height, viewconfig=None, grid_size=20, major=5):
    """Draw grid coordinate numbers."""
    if not self.viewconfig or not self.viewconfig.show_labels:
        return

    if viewconfig:
        self.viewconfig = viewconfig

    dl = imgui.get_background_draw_list()
    active_planes = _get_active_planes(self)

    camera_dist = max(1.0, camera.radius)
    density_factor = max(1, int(camera_dist / 8))
    step = major * density_factor

    for plane in active_planes:
        for i in range(-grid_size, grid_size + 1, step):
            if i == 0:
                continue

            positions = _get_grid_positions(self, plane, i, grid_size)

            for pos, axis in positions:
                x, y, depth = self.world_to_screen(camera, pos, width, height)

                if x is not None and -0.8 < depth < 0.8:
                    label = str(i)
                    color = self.axis_colors[axis]
                    offset_x, offset_y = self.label_offsets['grid']

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


def _get_active_planes(self):
    if self.viewconfig.grid_mode == "cube":
        return ["xy", "xz", "yz"]
    return [self.viewconfig.grid_plane]


def _get_grid_positions(self, plane, value, grid_size):
    positions = []

    if plane == "xy":
        positions.append((np.array([value, 0, 0], dtype=np.float32), 'x'))
        positions.append((np.array([0, value, 0], dtype=np.float32), 'y'))
    elif plane == "xz":
        positions.append((np.array([value, 0, 0], dtype=np.float32), 'x'))
        positions.append((np.array([0, 0, value], dtype=np.float32), 'z'))
    elif plane == "yz":
        positions.append((np.array([0, value, 0], dtype=np.float32), 'y'))
        positions.append((np.array([0, 0, value], dtype=np.float32), 'z'))

    return positions
