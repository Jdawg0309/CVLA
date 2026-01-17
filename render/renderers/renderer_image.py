"""Image rendering helpers."""

import numpy as np


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


def _colorize_with_source(color_matrix, y, x, intensity):
    """Mix the processed intensity with the original RGB channels."""
    base_r = max(0.0, min(1.0, float(color_matrix[y, x, 0])))
    base_g = max(0.0, min(1.0, float(color_matrix[y, x, 1])))
    base_b = max(0.0, min(1.0, float(color_matrix[y, x, 2])))
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


def _cache_key(image_data, color_source, color_mode, scale, render_mode):
    """Create a stable key for caching image batches."""
    image_id = getattr(image_data, "id", id(image_data))
    alt_id = getattr(color_source, "id", id(color_source)) if color_source is not None else None
    return (image_id, alt_id, color_mode, float(scale), render_mode)


def _make_batch(vertices, normals, colors):
    """Convert lists to typed numpy arrays for the triangle batch."""
    verts_np = np.array(vertices, dtype='f4').reshape(-1, 3)
    norms_np = np.array(normals, dtype='f4').reshape(-1, 3)
    colors_np = np.array(colors, dtype='f4').reshape(-1, 4)
    return verts_np, norms_np, colors_np


def _build_image_batches(self, matrix, channels, color_matrix, alt_channels,
                         color_mode, scale, chunk_pixel_limit, half_w, half_h, render_mode):
    """Build vertex, normal, and color batches that fit into the VBO."""
    vertices = []
    normals = []
    colors = []
    normal = [0.0, 0.0, 1.0]
    batches = []

    def flush():
        if not vertices:
            return
        batches.append(_make_batch(vertices, normals, colors))
        vertices.clear()
        normals.clear()
        colors.clear()

    h, w = matrix.shape[:2]
    for y in range(h):
        for x in range(w):
            intensity = None
            if _is_rgb_display(color_mode, matrix, channels):
                color = _rgb_color(matrix, channels, y, x)
            elif color_mode == "rgb" and color_matrix is not None and alt_channels >= 3:
                intensity = _get_intensity(matrix, y, x)
                color = _colorize_with_source(color_matrix, y, x, intensity)
            else:
                intensity = _get_intensity(matrix, y, x)
                color = _image_color(self, intensity, color_mode)

            height = 0.0
            if render_mode == "height-field":
                if intensity is None:
                    intensity = _get_intensity(matrix, y, x)
                height = float(intensity) * scale

            x0 = (x - half_w) * scale
            x1 = (x - half_w + 1.0) * scale
            y0 = (half_h - y) * scale
            y1 = (half_h - y - 1.0) * scale

            quad = [
                [x0, y0, height],
                [x0, y1, height],
                [x1, y1, height],
                [x0, y0, height],
                [x1, y1, height],
                [x1, y0, height],
            ]

            vertices.extend(quad)
            normals.extend([normal] * len(quad))
            colors.extend([color] * len(quad))

            if len(vertices) >= chunk_pixel_limit * 6:
                flush()

    flush()
    return batches


def draw_image_plane(self, image_data, vp, scale=1.0, color_mode="grayscale", color_source=None,
                     render_mode="plane"):
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

    key = _cache_key(image_data, color_source, color_mode, scale, render_mode)
    cache = getattr(self, "_image_plane_cache", None)
    if cache is None or cache["key"] != key:
        batches = _build_image_batches(
            self, matrix, channels,
            alt_matrix if alt_matrix is not None else None,
            alt_channels, color_mode, scale,
            chunk_pixel_limit, half_w, half_h, render_mode
        )
        self._image_plane_cache = {"key": key, "batches": batches}
    else:
        batches = cache["batches"]

    if not batches:
        return

    for verts, norms, colors in batches:
        self.gizmos.draw_triangles(verts, norms, colors, vp, use_lighting=False)
