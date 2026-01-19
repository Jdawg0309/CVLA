"""
Enhanced label renderer with improved readability and scaling.
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


class LabelRenderer:
    def __init__(self):
        self.viewconfig = None
        self.font = None
        self.label_cache = {}

        self.axis_colors = {
            'x': (1.0, 0.3, 0.3, 1.0),
            'y': (0.3, 1.0, 0.3, 1.0),
            'z': (0.3, 0.5, 1.0, 1.0)
        }

        self.grid_color = (0.92, 0.92, 0.96, 1.0)
        self.shadow_color = (0.0, 0.0, 0.0, 0.75)
        self.background_color = (0.06, 0.06, 0.07, 0.92)

        self.label_offsets = {
            'axis': (14, -18),
            'grid': (6, -10),
            'vector': (8, 8)
        }

    def update_view(self, viewconfig):
        """Update view configuration."""
        self.viewconfig = viewconfig

    def world_to_screen(self, camera, world_pos, width, height):
        """Convert world coordinates to screen coordinates."""
        screen_pos = camera.world_to_screen(world_pos, width, height)
        if screen_pos is None:
            return None, None, None

        x, y, depth = screen_pos
        return x, y, depth

    draw_axes = draw_axes
    draw_grid_numbers = draw_grid_numbers
    draw_vector_labels = draw_vector_labels
    _get_axis_label = _get_axis_label
    _get_active_planes = _get_active_planes
    _get_grid_positions = _get_grid_positions
