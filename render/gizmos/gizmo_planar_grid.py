"""
Planar grid helpers.
"""


def draw_grid(self, vp, size=10, step=1, plane='xy',
              color_major=(0.28, 0.30, 0.34, 0.55),
              color_minor=(0.18, 0.20, 0.22, 0.28)):
    """Draw a simple planar grid on specified plane ('xy','xz','yz')."""
    vertices = []
    colors = []
    half = int(size)
    step = int(step) if step > 0 else 1

    for i in range(-half, half + 1):
        is_major = (i % (step * 5) == 0)
        color = color_major if is_major else color_minor

        if plane == 'xy':
            vertices.extend([[i, -half, 0], [i, half, 0]])
            colors.extend([color, color])
            vertices.extend([[-half, i, 0], [half, i, 0]])
            colors.extend([color, color])
        elif plane == 'xz':
            vertices.extend([[i, 0, -half], [i, 0, half]])
            colors.extend([color, color])
            vertices.extend([[-half, 0, i], [half, 0, i]])
            colors.extend([color, color])
        elif plane == 'yz':
            vertices.extend([[0, i, -half], [0, i, half]])
            colors.extend([color, color])
            vertices.extend([[0, -half, i], [0, half, i]])
            colors.extend([color, color])

    self.draw_lines(vertices, colors, vp, width=1.0)


def draw_axes(self, vp, length=6.0, thickness=3.0):
    """Draw basic XYZ axes as lines with endpoints."""
    axes = [
        ([[0, 0, 0], [length, 0, 0]], (1.0, 0.3, 0.3, 1.0)),
        ([[0, 0, 0], [0, length, 0]], (0.3, 1.0, 0.3, 1.0)),
        ([[0, 0, 0], [0, 0, length]], (0.3, 0.5, 1.0, 1.0)),
    ]
    for pts, col in axes:
        self.draw_lines(pts, [col, col], vp, width=thickness)
