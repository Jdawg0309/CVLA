"""
Image rendering helpers.
"""


def draw_image_plane(self, image_data, vp, scale=1.0, color_mode="grayscale", color_source=None):
    """Render image pixels as filled squares on the XY plane (Z = 0)."""
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

    half_w = w / 2.0
    half_h = h / 2.0

    buffer_capacity = getattr(self.gizmos.triangle_vbo, "size", 2 * 1024 * 1024)
    bytes_per_vertex = 10 * 4  # 3 position + 3 normal + 4 color floats.
    bytes_per_pixel = 6 * bytes_per_vertex
    chunk_pixel_limit = max(256, buffer_capacity // bytes_per_pixel)
    chunk_pixel_limit = max(chunk_pixel_limit, 1)

    vertices = []
    normals = []
    colors = []
    normal = [0.0, 0.0, 1.0]

    def flush():
        if not vertices:
            return
        self.gizmos.draw_triangles(vertices, normals, colors, vp, use_lighting=False)
        vertices.clear()
        normals.clear()
        colors.clear()

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

            x0 = (x - half_w) * scale
            x1 = (x - half_w + 1.0) * scale
            y0 = (half_h - y) * scale
            y1 = (half_h - y - 1.0) * scale

            quad = [
                [x0, y0, 0.0],
                [x0, y1, 0.0],
                [x1, y1, 0.0],
                [x0, y0, 0.0],
                [x1, y1, 0.0],
                [x1, y0, 0.0],
            ]

            vertices.extend(quad)
            normals.extend([normal] * len(quad))
            colors.extend([color] * len(quad))

            if len(vertices) >= chunk_pixel_limit * 6:
                flush()

    flush()


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
    value = max(0.0, min(1.0, intensity))
    brightness = 0.3 + 0.7 * value
    return (
        max(0.0, min(1.0, base_r * brightness)),
        max(0.0, min(1.0, base_g * brightness)),
        max(0.0, min(1.0, base_b * brightness)),
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
