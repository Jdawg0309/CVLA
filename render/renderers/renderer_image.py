"""
Image rendering helpers.
"""


def draw_image_plane(self, image_data, vp, scale=1.0, color_mode="grayscale", color_source=None):
    """Render image pixels as points on the XY plane (Z = 0)."""
    try:
        matrix, channels = _resolve_image_matrix(image_data)
    except Exception:
        return

    alt_matrix = None
    alt_channels = 1
    if color_source is not None:
        try:
            alt_matrix, alt_channels = _resolve_image_matrix(color_source)
        except Exception:
            alt_matrix = None

    h, w = matrix.shape[:2]
    if h == 0 or w == 0:
        return

    points = []
    colors = []
    half_w = w / 2.0
    half_h = h / 2.0

    for y in range(h):
        for x in range(w):
            if _is_rgb_display(color_mode, matrix, channels):
                color = _rgb_color(matrix, channels, y, x)
            elif color_mode == "rgb" and alt_matrix is not None and alt_channels >= 3:
                intensity = _get_intensity(matrix, y, x)
                color = _colorize_with_source(alt_matrix, y, x, intensity)
            else:
                intensity = _get_intensity(matrix, y, x)
                color = _image_color(self, intensity, color_mode)

            px = (x - half_w) * scale
            py = (half_h - y) * scale
            points.append([px, py, 0.0])
            colors.append(color)

    if points:
        self.gizmos.draw_points(points, colors, vp, size=4.0)


def _resolve_image_matrix(image_data):
    """Return the pixel matrix and channel count for the renderer."""
    matrix = None
    if hasattr(image_data, "data"):
        matrix = image_data.data
    elif hasattr(image_data, "pixels"):
        matrix = image_data.pixels

    if matrix is None:
        matrix = image_data.as_matrix()

    channels = getattr(image_data, "channels", matrix.shape[2] if matrix.ndim > 2 else 1)
    return matrix, channels


def _is_rgb_display(color_mode, matrix, channels):
    """Detect whether the matrix is already RGB and the mode allows RGB."""
    return (
        color_mode == "rgb" and
        channels >= 3 and
        matrix.ndim == 3 and
        matrix.shape[2] >= 3
    )


def _rgb_color(matrix, channels, y, x):
    """Read raw RGB color from a 3-channel matrix."""
    r = max(0.0, min(1.0, float(matrix[y, x, 0])))
    g = max(0.0, min(1.0, float(matrix[y, x, 1])))
    b = max(0.0, min(1.0, float(matrix[y, x, 2])))
    return (r, g, b, 1.0)


def _get_intensity(matrix, y, x):
    if matrix.ndim == 3:
        return float(matrix[y, x, 0])
    return float(matrix[y, x])


def _colorize_with_source(alt_matrix, y, x, intensity):
    """Mix the processed intensity with the original RGB channels."""
    base_r = max(0.0, min(1.0, float(alt_matrix[y, x, 0])))
    base_g = max(0.0, min(1.0, float(alt_matrix[y, x, 1])))
    base_b = max(0.0, min(1.0, float(alt_matrix[y, x, 2])))
    factor = 0.4 + 0.6 * intensity
    return (
        max(0.0, min(1.0, base_r * factor)),
        max(0.0, min(1.0, base_g * factor)),
        max(0.0, min(1.0, base_b * factor)),
        1.0,
    )


def _image_color(self, intensity, color_mode):
    """Map intensity to grayscale or heatmap color."""
    i = max(0.0, min(1.0, float(intensity)))
    if color_mode == "heatmap":
        r = min(1.0, max(0.0, 1.5 * (i - 0.33)))
        g = min(1.0, max(0.0, 1.5 * (1.0 - abs(i - 0.5) * 2.0)))
        b = min(1.0, max(0.0, 1.5 * (0.66 - i)))
        return (r, g, b, 1.0)
    return (i, i, i, 1.0)
