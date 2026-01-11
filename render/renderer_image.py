"""
Image rendering helpers.
"""


def draw_image_plane(self, image_data, vp, scale=1.0, color_mode="grayscale"):
    """Render image pixels as points on the XY plane (Z = 0)."""
    try:
        if image_data.channels >= 3:
            matrix = image_data.data
        else:
            matrix = image_data.as_matrix()
    except Exception:
        return

    h, w = matrix.shape[:2]
    if h == 0 or w == 0:
        return

    points = []
    colors = []
    half_w = w / 2.0
    half_h = h / 2.0
    for y in range(h):
        for x in range(w):
            if matrix.ndim == 3 and matrix.shape[2] >= 3:
                r = max(0.0, min(1.0, float(matrix[y, x, 0])))
                g = max(0.0, min(1.0, float(matrix[y, x, 1])))
                b = max(0.0, min(1.0, float(matrix[y, x, 2])))
                color = (r, g, b, 1.0)
            else:
                intensity = float(matrix[y, x])
                i = max(0.0, min(1.0, intensity))
                color = _image_color(self, i, color_mode)
            px = (x - half_w) * scale
            py = (half_h - y) * scale
            points.append([px, py, 0.0])
            colors.append(color)

    if points:
        self.gizmos.draw_points(points, colors, vp, size=4.0)


def _image_color(self, intensity, color_mode):
    """Map intensity to grayscale or heatmap color."""
    i = max(0.0, min(1.0, float(intensity)))
    if color_mode == "heatmap":
        r = min(1.0, max(0.0, 1.5 * (i - 0.33)))
        g = min(1.0, max(0.0, 1.5 * (1.0 - abs(i - 0.5) * 2.0)))
        b = min(1.0, max(0.0, 1.5 * (0.66 - i)))
        return (r, g, b, 1.0)
    return (i, i, i, 1.0)
