"""
Axis rendering helpers.
"""

import numpy as np


def _render_3d_axes_with_depths(self, vp):
    """Render 3D axes with depth cues."""
    length = 8.0

    axes = [
        {"points": [[0, 0, 0], [length, 0, 0]], "color": (1.0, 0.3, 0.3, 1.0)},
        {"points": [[0, 0, 0], [0, length, 0]], "color": (0.3, 1.0, 0.3, 1.0)},
        {"points": [[0, 0, 0], [0, 0, length]], "color": (0.3, 0.5, 1.0, 1.0)},
    ]

    for axis in axes:
        self.gizmos.draw_lines(
            axis["points"],
            [axis["color"], axis["color"]],
            vp, width=3.0
        )

    size = self.view.grid_size * 1.5
    faint_axes = [
        {"points": [[-size, 0, 0], [size, 0, 0]], "color": (1.0, 0.3, 0.3, 0.15)},
        {"points": [[0, -size, 0], [0, size, 0]], "color": (0.3, 1.0, 0.3, 0.15)},
        {"points": [[0, 0, -size], [0, 0, size]], "color": (0.3, 0.5, 1.0, 0.15)},
    ]

    for axis in faint_axes:
        self.gizmos.draw_lines(
            axis["points"],
            [axis["color"], axis["color"]],
            vp, width=1.0
        )

    self._draw_axis_cones(vp, length)


def _draw_axis_cones(self, vp, length):
    """Draw 3D cones at axis tips."""
    cone_height = length * 0.15
    cone_radius = length * 0.05

    axes = [
        {"tip": [length, 0, 0], "direction": [1, 0, 0], "color": (1.0, 0.3, 0.3, 1.0)},
        {"tip": [0, length, 0], "direction": [0, 1, 0], "color": (0.3, 1.0, 0.3, 1.0)},
        {"tip": [0, 0, length], "direction": [0, 0, 1], "color": (0.3, 0.5, 1.0, 1.0)},
    ]

    for axis in axes:
        tip = np.array(axis["tip"])
        direction = np.array(axis["direction"])
        base = tip - direction * cone_height

        cone_verts = []
        cone_norms = []
        cone_colors = []
        segments = 12

        if abs(direction[0]) < 0.9:
            perp1 = np.array([1.0, 0.0, 0.0])
        else:
            perp1 = np.array([0.0, 1.0, 0.0])

        perp2 = np.cross(direction, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        perp1 = np.cross(perp2, direction)

        for i in range(segments):
            angle1 = 2 * np.pi * i / segments
            angle2 = 2 * np.pi * (i + 1) / segments

            p1 = base + cone_radius * (np.cos(angle1) * perp1 + np.sin(angle1) * perp2)
            p2 = base + cone_radius * (np.cos(angle2) * perp1 + np.sin(angle2) * perp2)

            cone_verts.extend(base.tolist())
            cone_verts.extend(p1.tolist())
            cone_verts.extend(tip.tolist())

            v1 = p1 - base
            v2 = tip - base
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)

            cone_norms.extend([normal.tolist()] * 3)
            cone_colors.extend([axis["color"]] * 3)

        self.gizmos.draw_triangles(cone_verts, cone_norms, cone_colors, vp)
