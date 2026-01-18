"""
Planar grid helpers.
"""

import math


def _is_multiple(value, step, eps=1e-6):
    if step <= 0:
        return False
    scaled = value / step
    return abs(scaled - round(scaled)) <= eps


def _fade(dist, max_dist, power=1.6):
    if max_dist <= 0:
        return 1.0
    t = min(1.0, max(0.0, dist / max_dist))
    return max(0.0, 1.0 - (t ** power))


def draw_grid(self, vp, size=10, step=1, plane='xy',
              color_major=(0.28, 0.30, 0.34, 0.45),
              color_minor=(0.18, 0.20, 0.22, 0.22),
              color_subminor=(0.18, 0.20, 0.22, 0.12),
              major_step=None,
              sub_step=None,
              fade_power=1.6,
              depth=True):
    """Draw a semantic planar grid on specified plane ('xy','xz','yz')."""
    vertices = []
    colors = []
    half = float(size)
    step = float(step) if step > 0 else 1.0
    major_step = float(major_step) if major_step is not None else step * 5.0
    sub_step = float(sub_step) if sub_step is not None else None

    def add_line(i, color):
        fade = _fade(abs(i), half, fade_power)
        if fade <= 0.0:
            return
        c = (color[0], color[1], color[2], color[3] * fade)
        if plane == 'xy':
            vertices.extend([[i, -half, 0], [i, half, 0]])
            colors.extend([c, c])
            vertices.extend([[-half, i, 0], [half, i, 0]])
            colors.extend([c, c])
        elif plane == 'xz':
            vertices.extend([[i, 0, -half], [i, 0, half]])
            colors.extend([c, c])
            vertices.extend([[-half, 0, i], [half, 0, i]])
            colors.extend([c, c])
        elif plane == 'yz':
            vertices.extend([[0, i, -half], [0, i, half]])
            colors.extend([c, c])
            vertices.extend([[0, -half, i], [0, half, i]])
            colors.extend([c, c])

    def add_tier(step_size, color, skip_steps=()):
        if step_size is None or step_size <= 0:
            return
        count = int(math.floor(half / step_size))
        for idx in range(-count, count + 1):
            i = idx * step_size
            if any(_is_multiple(i, s) for s in skip_steps if s):
                continue
            add_line(i, color)

    add_tier(sub_step, color_subminor, skip_steps=(step, major_step))
    add_tier(step, color_minor, skip_steps=(major_step,))
    add_tier(major_step, color_major, skip_steps=())

    self.draw_lines(vertices, colors, vp, width=1.0, depth=depth)


def draw_axes(self, vp, length=6.0, thickness=3.0):
    """Draw basic XYZ axes as lines with endpoints."""
    axes = [
        ([[0, 0, 0], [length, 0, 0]], (1.0, 0.3, 0.3, 1.0)),
        ([[0, 0, 0], [0, length, 0]], (0.3, 1.0, 0.3, 1.0)),
        ([[0, 0, 0], [0, 0, length]], (0.3, 0.5, 1.0, 1.0)),
    ]
    for pts, col in axes:
        self.draw_lines(pts, [col, col], vp, width=thickness)
