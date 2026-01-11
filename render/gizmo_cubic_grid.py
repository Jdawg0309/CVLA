"""
Cubic grid helpers.
"""


def draw_cubic_grid(self, vp, size=10, major_step=5, minor_step=1,
                   color_major=(0.25, 0.27, 0.3, 0.9),
                   color_minor=(0.15, 0.16, 0.18, 0.6)):
    """Draw a beautiful 3D cubic grid."""
    vertices = []
    colors = []

    for plane in ['xy', 'xz', 'yz']:
        for i in range(-size, size + 1, minor_step):
            is_major = (i % major_step == 0)
            color = color_major if is_major else color_minor

            if plane == 'xy':
                vertices.extend([[i, -size, 0], [i, size, 0]])
                colors.extend([color, color])
                vertices.extend([[-size, i, 0], [size, i, 0]])
                colors.extend([color, color])
            elif plane == 'xz':
                vertices.extend([[i, 0, -size], [i, 0, size]])
                colors.extend([color, color])
                vertices.extend([[-size, 0, i], [size, 0, i]])
                colors.extend([color, color])
            elif plane == 'yz':
                vertices.extend([[0, i, -size], [0, i, size]])
                colors.extend([color, color])
                vertices.extend([[0, -size, i], [0, size, i]])
                colors.extend([color, color])

    self.draw_lines(vertices, colors, vp, width=1.0)
    self.draw_cube(vp, [-size, -size, -size], [size, size, size],
                  (0.35, 0.35, 0.38, 0.8), width=2.0)


def draw_cube(self, vp, min_corner, max_corner, color, width=2.0):
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
    self.draw_lines(vertices, colors, vp, width=width)
