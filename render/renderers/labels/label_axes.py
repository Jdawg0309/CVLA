"""
Axis label rendering helpers.
"""

import numpy as np
import imgui


def draw_axes(self, camera, width, height):
    """Draw axis labels with proper positioning."""
    if not self.viewconfig or not self.viewconfig.show_labels:
        return

    dl = imgui.get_background_draw_list()

    axis_length = 6.0
    endpoints = {
        'x': np.array([axis_length, 0, 0], dtype=np.float32),
        'y': np.array([0, axis_length, 0], dtype=np.float32),
        'z': np.array([0, 0, axis_length], dtype=np.float32)
    }

    if self.viewconfig.up_axis == 'y':
        endpoints = {
            'x': np.array([axis_length, 0, 0], dtype=np.float32),
            'y': np.array([0, 0, axis_length], dtype=np.float32),
            'z': np.array([0, axis_length, 0], dtype=np.float32)
        }
    elif self.viewconfig.up_axis == 'x':
        endpoints = {
            'x': np.array([0, axis_length, 0], dtype=np.float32),
            'y': np.array([0, 0, axis_length], dtype=np.float32),
            'z': np.array([axis_length, 0, 0], dtype=np.float32)
        }

    for axis in ['x', 'y', 'z']:
        endpoint = endpoints[axis]
        x, y, depth = self.world_to_screen(camera, endpoint, width, height)

        if x is not None and depth > -0.9:
            label = _get_axis_label(self, axis).upper()
            color = self.axis_colors[axis]
            offset_x, offset_y = self.label_offsets['axis']

            text_size = imgui.calc_text_size(label)
            padding = 6
            bg_color = imgui.get_color_u32_rgba(*self.background_color)
            dl.add_rect_filled(
                x + offset_x - padding, y + offset_y - padding,
                x + offset_x + text_size.x + padding, y + offset_y + text_size.y + padding,
                bg_color, 6.0
            )

            shadow_color = imgui.get_color_u32_rgba(*self.shadow_color)
            dl.add_text(x + offset_x + 2, y + offset_y + 2, shadow_color, label)

            text_color = imgui.get_color_u32_rgba(*color)
            dl.add_text(x + offset_x, y + offset_y, text_color, label)


def _get_axis_label(self, axis):
    axis_map = {
        'x': 'x',
        'y': 'y',
        'z': 'z'
    }

    if self.viewconfig.up_axis == 'y':
        axis_map = {'x': 'x', 'y': 'z', 'z': 'y'}
    elif self.viewconfig.up_axis == 'x':
        axis_map = {'x': 'y', 'y': 'z', 'z': 'x'}

    return axis_map.get(axis, axis)
