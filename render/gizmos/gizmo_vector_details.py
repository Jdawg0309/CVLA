"""
Vector detail drawing helpers.
"""

import numpy as np


def draw_vector_with_details(self, vp, vector, selected=False, scale=1.0,
                            show_components=True, show_span=False, depth=True):
    """
    Draw a vector with enhanced visualizations.
    """
    tip = vector.coords * scale
    length = np.linalg.norm(tip)

    if length < 1e-6:
        return

    r, g, b = vector.color
    if selected:
        r = min(1.0, r * 1.5)
        g = min(1.0, g * 1.5)
        b = min(1.0, b * 1.5)
        alpha = 1.0
        shaft_width = 5.0
    else:
        alpha = 1.0
        shaft_width = 3.0

    color = (r, g, b, alpha)

    shaft_verts = [[0, 0, 0], tip.tolist()]
    shaft_colors = [color, color]
    self.draw_lines(shaft_verts, shaft_colors, vp, width=shaft_width, depth=depth)

    self._draw_arrow_head(vp, tip, vector.coords, color, shaft_width, depth)

    if show_components:
        self._draw_vector_components(vp, tip, color, depth)

    self.draw_points([tip.tolist()], [color], vp, size=12.0 if selected else 8.0, depth=depth)


def _draw_arrow_head(self, vp, tip, direction, color, shaft_width, depth=True):
    """Draw a 3D arrow head."""
    length = np.linalg.norm(tip)
    if length < 0.3:
        return

    dir_norm = direction / length
    head_length = min(0.4, length * 0.3)
    head_radius = head_length * 0.4

    head_base = tip - dir_norm * head_length

    if abs(dir_norm[0]) < 0.9:
        perp = np.array([1.0, 0.0, 0.0])
    else:
        perp = np.array([0.0, 1.0, 0.0])
    right = np.cross(dir_norm, perp)
    rnorm = np.linalg.norm(right)
    if rnorm > 1e-8:
        right = right / rnorm
    else:
        right = np.array([0.0, 1.0, 0.0])
    up = np.cross(right, dir_norm)

    head_verts = []
    head_colors = []
    segments = 8

    for i in range(segments):
        angle1 = 2 * np.pi * i / segments
        angle2 = 2 * np.pi * (i + 1) / segments

        p1 = head_base + head_radius * (np.cos(angle1) * right + np.sin(angle1) * up)
        p2 = head_base + head_radius * (np.cos(angle2) * right + np.sin(angle2) * up)

        head_verts.extend(tip.tolist())
        head_verts.extend(p1.tolist())
        head_colors.extend([color, color])

        head_verts.extend(p1.tolist())
        head_verts.extend(p2.tolist())
        head_colors.extend([color, color])

        self.draw_lines(head_verts, head_colors, vp, width=shaft_width, depth=depth)


def _draw_vector_components(self, vp, tip, color, depth=True):
    """Draw projection lines to axes."""
    proj_color = (color[0], color[1], color[2], 0.4)

    xy_proj = [tip[0], tip[1], 0]
    vertices = [tip.tolist(), xy_proj]
    colors = [proj_color, proj_color]
    self.draw_lines(vertices, colors, vp, width=1.0, depth=depth)

    xz_proj = [tip[0], 0, tip[2]]
    vertices = [tip.tolist(), xz_proj]
    self.draw_lines(vertices, colors, vp, width=1.0, depth=depth)

    yz_proj = [0, tip[1], tip[2]]
    vertices = [tip.tolist(), yz_proj]
    self.draw_lines(vertices, colors, vp, width=1.0)
