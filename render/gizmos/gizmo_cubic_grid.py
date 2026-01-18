"""
Cubic grid helpers.
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


def draw_cubic_grid(self, vp, size=10, major_step=5, minor_step=1,
                   color_major=(0.28, 0.30, 0.34, 0.45),
                   color_minor=(0.18, 0.20, 0.22, 0.22),
                   color_subminor=(0.18, 0.20, 0.22, 0.12),
                   subminor_step=None,
                   fade_power=1.6,
                   depth=True):
    """Draw a semantic 3D cubic grid."""
    vertices = []
    colors = []
    size = float(size)
    major_step = float(major_step) if major_step else 5.0
    minor_step = float(minor_step) if minor_step else 1.0
    subminor_step = float(subminor_step) if subminor_step else None

    def add_line(plane, i, color):
        fade = _fade(abs(i), size, fade_power)
        if fade <= 0.0:
            return
        c = (color[0], color[1], color[2], color[3] * fade)
        if plane == 'xy':
            vertices.extend([[i, -size, 0], [i, size, 0]])
            colors.extend([c, c])
            vertices.extend([[-size, i, 0], [size, i, 0]])
            colors.extend([c, c])
        elif plane == 'xz':
            vertices.extend([[i, 0, -size], [i, 0, size]])
            colors.extend([c, c])
            vertices.extend([[-size, 0, i], [size, 0, i]])
            colors.extend([c, c])
        elif plane == 'yz':
            vertices.extend([[0, i, -size], [0, i, size]])
            colors.extend([c, c])
            vertices.extend([[0, -size, i], [0, size, i]])
            colors.extend([c, c])

    def add_tier(step_size, color, skip_steps=()):
        if step_size is None or step_size <= 0:
            return
        count = int(math.floor(size / step_size))
        for idx in range(-count, count + 1):
            i = idx * step_size
            if any(_is_multiple(i, s) for s in skip_steps if s):
                continue
            for plane in ['xy', 'xz', 'yz']:
                add_line(plane, i, color)

    add_tier(subminor_step, color_subminor, skip_steps=(minor_step, major_step))
    add_tier(minor_step, color_minor, skip_steps=(major_step,))
    add_tier(major_step, color_major, skip_steps=())

    self.draw_lines(vertices, colors, vp, width=1.0, depth=depth)
    self.draw_cube(vp, [-size, -size, -size], [size, size, size],
                  (0.35, 0.35, 0.38, 0.25), width=1.6, depth=depth)


def draw_cube(self, vp, min_corner, max_corner, color, width=2.0, depth=True):
    """Draw a wireframe cube."""
    x0, y0, z0 = min_corner
    x1, y1, z1 = max_corner

    vertices = [
        [x0, y0, z0], [x1, y0, z0],
        [x1, y0, z0], [x1, y1, z0],
        [x1, y1, z0], [x0, y1, z0],
        [x0, y1, z0], [x0, y0, z0],

        [x0, y0, z1], [x1, y0, z1],
        [x1, y0, z1], [x1, y1, z1],
        [x1, y1, z1], [x0, y1, z1],
        [x0, y1, z1], [x0, y0, z1],

        [x0, y0, z0], [x0, y0, z1],
        [x1, y0, z0], [x1, y0, z1],
        [x1, y1, z0], [x1, y1, z1],
        [x0, y1, z0], [x0, y1, z1]
    ]

    colors = [color] * len(vertices)
    self.draw_lines(vertices, colors, vp, width=width, depth=depth)
