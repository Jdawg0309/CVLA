"""
Enhanced label renderer with improved readability and scaling.
"""

import numpy as np
import imgui

from render.renderers.labels.label_axes import draw_axes, _get_axis_label
from render.renderers.labels.label_grid import draw_grid_numbers, _get_active_planes, _get_grid_positions
from render.renderers.labels.label_vectors import draw_vector_labels


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
