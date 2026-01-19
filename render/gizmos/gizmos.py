"""
Gizmos: immediate-mode debug drawing.
"""

import ctypes
import math
from contextlib import contextmanager

import moderngl
import numpy as np

from render.shaders.gizmo_programs import (
    _create_line_program,
    _create_triangle_program,
    _create_point_program,
    _create_volume_program,
)
from render.buffers.gizmo_buffers import _init_buffers
from render.shaders.grid_shader import get_grid_shaders


def _get_gl_depth_mask(ctx):
    extra = getattr(ctx, "extra", None)
    if extra is None:
        return True
    gl_obj = extra.get("gl") if isinstance(extra, dict) else getattr(extra, "gl", None)
    if gl_obj is None:
        # assume default writable mask when GL object unavailable
        return True
    buf = (ctypes.c_int * 1)(0)
    gl_obj.glGetIntegerv(moderngl.GL_DEPTH_WRITEMASK, buf)
    return bool(buf[0])


def _set_gl_depth_mask(ctx, mask):
    extra = getattr(ctx, "extra", None)
    if extra is None:
        return
    gl_obj = extra.get("gl") if isinstance(extra, dict) else getattr(extra, "gl", None)
    if gl_obj is None:
        return
    gl_obj.glDepthMask(moderngl.GL_TRUE if mask else moderngl.GL_FALSE)


@contextmanager
def _temporary_depth_mask(ctx, mask=True):
    """Temporarily override the context depth mask."""
    prev_mask = _get_gl_depth_mask(ctx)
    _set_gl_depth_mask(ctx, mask)
    try:
        yield
    finally:
        _set_gl_depth_mask(ctx, prev_mask)


def draw_lines(self, vertices, colors, vp, width=2.0, depth=True, write_depth=True):
    """Draw line segments with per-vertex colors."""
    if not vertices or len(vertices) == 0:
        return

    if depth:
        self.ctx.enable(moderngl.DEPTH_TEST)
    else:
        self.ctx.disable(moderngl.DEPTH_TEST)

    self.ctx.line_width = float(width)

    vertices = np.array(vertices, dtype='f4').reshape(-1, 3)
    colors = np.array(colors, dtype='f4').reshape(-1, 4)

    interleaved = np.zeros((vertices.shape[0], 7), dtype='f4')
    interleaved[:, :3] = vertices
    interleaved[:, 3:7] = colors

    self.line_vbo.write(interleaved.tobytes())
    self.line_program['mvp'].write(vp.astype('f4').tobytes())

    with _temporary_depth_mask(self.ctx, write_depth):
        self.line_vao.render(moderngl.LINES, vertices=vertices.shape[0])


def _ensure_float_array(buffer, components):
    """Ensure the buffer is a numpy array with the given component stride."""
    if isinstance(buffer, np.ndarray):
        array = buffer
    else:
        array = np.array(buffer, dtype='f4')
    return array.reshape(-1, components)


def draw_triangles(self, vertices, normals, colors, vp, model_matrix=None,
                   light_pos=(20, 20, 20), view_pos=(0, 0, 20), use_lighting=True,
                   depth=True, write_depth=True):
    """Draw triangles with normals and colors."""
    if depth:
        self.ctx.enable(moderngl.DEPTH_TEST)
    else:
        self.ctx.disable(moderngl.DEPTH_TEST)
    vertices = _ensure_float_array(vertices, 3)
    if vertices.size == 0:
        return
    normals = _ensure_float_array(normals, 3)
    colors = _ensure_float_array(colors, 4)

    interleaved = np.zeros((vertices.shape[0], 10), dtype='f4')
    interleaved[:, :3] = vertices
    interleaved[:, 3:6] = normals
    interleaved[:, 6:10] = colors

    self.triangle_vbo.write(interleaved.tobytes())

    if model_matrix is None:
        model_matrix = np.eye(4, dtype='f4')

    self.triangle_program['mvp'].write(vp.astype('f4').tobytes())
    self.triangle_program['model'].write(model_matrix.astype('f4').tobytes())
    self.triangle_program['light_pos'].write(np.array(light_pos, dtype='f4').tobytes())
    self.triangle_program['view_pos'].write(np.array(view_pos, dtype='f4').tobytes())
    self.triangle_program['use_lighting'].value = bool(use_lighting)

    transparent = np.any(colors[:, 3] < 0.999)
    if transparent:
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    with _temporary_depth_mask(self.ctx, write_depth):
        self.triangle_vao.render(moderngl.TRIANGLES, vertices=vertices.shape[0])


def draw_points(self, vertices, colors, vp, size=8.0, depth=True):
    """Draw points with per-vertex colors."""
    if not vertices or len(vertices) == 0:
        return

    if depth:
        self.ctx.enable(moderngl.DEPTH_TEST)
    else:
        self.ctx.disable(moderngl.DEPTH_TEST)

    vertices = np.array(vertices, dtype='f4').reshape(-1, 3)
    colors = np.array(colors, dtype='f4').reshape(-1, 4)

    interleaved = np.zeros((vertices.shape[0], 7), dtype='f4')
    interleaved[:, :3] = vertices
    interleaved[:, 3:7] = colors

    self.point_vbo.write(interleaved.tobytes())

    self.point_program['mvp'].write(vp.astype('f4').tobytes())
    self.point_program['point_size'].value = float(size)

    self.point_vao.render(moderngl.POINTS, vertices=vertices.shape[0])


def draw_volume(self, vertices, colors, vp, opacity=0.3, depth=True):
    """Draw volume visualization (transparent cube)."""
    if not vertices or len(vertices) == 0:
        return

    if depth:
        self.ctx.enable(moderngl.DEPTH_TEST)
    else:
        self.ctx.disable(moderngl.DEPTH_TEST)

    vertices = np.array(vertices, dtype='f4').reshape(-1, 3)
    colors = np.array(colors, dtype='f4').reshape(-1, 4)

    interleaved = np.zeros((vertices.shape[0], 7), dtype='f4')
    interleaved[:, :3] = vertices
    interleaved[:, 3:7] = colors

    self.volume_vbo.write(interleaved.tobytes())
    self.volume_program['mvp'].write(vp.astype('f4').tobytes())
    self.volume_program['opacity'].value = float(opacity)

    self.volume_vao.render(moderngl.TRIANGLES, vertices=vertices.shape[0])


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
                   depth=True,
                   write_depth=True):
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

    self.draw_lines(vertices, colors, vp, width=1.0, depth=depth, write_depth=write_depth)
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


def draw_grid(self, vp, size=10, step=1, plane='xy',
              color_major=(0.28, 0.30, 0.34, 0.45),
              color_minor=(0.18, 0.20, 0.22, 0.22),
              color_subminor=(0.18, 0.20, 0.22, 0.12),
              major_step=None,
              sub_step=None,
              fade_power=1.6,
              depth=True,
              write_depth=True):
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

    self.draw_lines(vertices, colors, vp, width=1.0, depth=depth, write_depth=write_depth)


def draw_axes(self, vp, length=6.0, thickness=3.0,
               color_x=(1.0, 0.3, 0.3, 1.0),
               color_y=(0.3, 1.0, 0.3, 1.0),
               color_z=(0.3, 0.5, 1.0, 1.0)):
    """Draw basic XYZ axes as lines with endpoints."""
    axes = [
        ([[0, 0, 0], [length, 0, 0]], color_x),
        ([[0, 0, 0], [0, length, 0]], color_y),
        ([[0, 0, 0], [0, 0, length]], color_z),
    ]
    for pts, col in axes:
        self.draw_lines(pts, [col, col], vp, width=thickness)



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


_SPAN_RADIUS_SCALE = 1.25
_SPAN_LINE_WIDTH = 1.6
_SPAN_BORDER_ALPHA = 0.32
_SPAN_FILL_ALPHA = 0.2
_SUBSPACE_EDGE_WIDTH = 1.4
_SUBSPACE_FACE_ALPHA = 0.12


def _desaturate(color, amount=0.6):
    """Desaturate an RGB/RGBA color by mixing toward luminance."""
    r, g, b = color[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    r = r * (1.0 - amount) + lum * amount
    g = g * (1.0 - amount) + lum * amount
    b = b * (1.0 - amount) + lum * amount
    if len(color) == 4:
        return (r, g, b, color[3])
    return (r, g, b)


def _span_color(vectors, base_alpha):
    """Create a desaturated span color from basis vectors."""
    if not vectors:
        return (0.4, 0.4, 0.5, base_alpha)
    colors = [np.array(v.color, dtype='f4') for v in vectors if hasattr(v, "color")]
    if not colors:
        return (0.4, 0.4, 0.5, base_alpha)
    avg = np.mean(colors, axis=0)
    desat = _desaturate((float(avg[0]), float(avg[1]), float(avg[2])), amount=0.6)
    return (desat[0], desat[1], desat[2], base_alpha)


def _scale_vectors_to_radius(vectors, radius_scale=_SPAN_RADIUS_SCALE):
    """Scale vectors uniformly so their max length matches the semantic radius."""
    lengths = [np.linalg.norm(v) for v in vectors]
    max_len = max(lengths + [0.0])
    if max_len <= 1e-8:
        return vectors
    scale = radius_scale
    return [v * scale for v in vectors]


def draw_vector_span(self, vp, vector1, vector2, color=None, scale=1.0):
    """Draw the span (parallelogram) between two vectors."""
    v1 = vector1.coords * scale
    v2 = vector2.coords * scale
    v1, v2 = _scale_vectors_to_radius([v1, v2], radius_scale=_SPAN_RADIUS_SCALE)

    vertices = [
        [0, 0, 0],
        v1.tolist(),
        (v1 + v2).tolist(),
        v2.tolist()
    ]

    tri_vertices = [
        vertices[0], vertices[1], vertices[2],
        vertices[0], vertices[2], vertices[3]
    ]

    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    normal = normal / norm if norm > 0 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
    normals = [normal.tolist()] * 6

    if color is None:
        color = _span_color([vector1, vector2], _SPAN_FILL_ALPHA)
    colors = [color] * 6

    self.draw_triangles(tri_vertices, normals, colors, vp, use_lighting=True, write_depth=False)

    border_vertices = [
        vertices[0], vertices[1],
        vertices[1], vertices[2],
        vertices[2], vertices[3],
        vertices[3], vertices[0]
    ]
    border_color = _span_color([vector1, vector2], _SPAN_BORDER_ALPHA)
    border_colors = [border_color] * 8
    self.draw_lines(border_vertices, border_colors, vp, width=_SPAN_LINE_WIDTH)


def draw_parallelepiped(self, vp, vectors, color=None, scale=1.0):
    """Draw parallelepiped spanned by three vectors."""
    if len(vectors) < 3:
        return

    v1 = vectors[0].coords * scale
    v2 = vectors[1].coords * scale
    v3 = vectors[2].coords * scale
    v1, v2, v3 = _scale_vectors_to_radius([v1, v2, v3], radius_scale=_SPAN_RADIUS_SCALE)
    if color is None:
        color = _span_color(vectors[:3], _SPAN_BORDER_ALPHA)

    vertices = [
        np.array([0, 0, 0]),
        v1,
        v2,
        v3,
        v1 + v2,
        v1 + v3,
        v2 + v3,
        v1 + v2 + v3
    ]

    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7)
    ]

    edge_vertices = []
    for i, j in edges:
        edge_vertices.append(vertices[i].tolist())
        edge_vertices.append(vertices[j].tolist())

    edge_colors = [(color[0], color[1], color[2], min(_SPAN_BORDER_ALPHA, 0.35))] * len(edge_vertices)
    self.draw_lines(edge_vertices, edge_colors, vp, width=_SUBSPACE_EDGE_WIDTH)

    faces = [
        [vertices[0], vertices[1], vertices[4], vertices[2]],
        [vertices[0], vertices[1], vertices[5], vertices[3]],
        [vertices[0], vertices[2], vertices[6], vertices[3]],
        [vertices[7], vertices[5], vertices[1], vertices[4]],
        [vertices[7], vertices[6], vertices[2], vertices[4]],
        [vertices[7], vertices[6], vertices[3], vertices[5]]
    ]

    for face in faces:
        tri_vertices = [
            face[0].tolist(), face[1].tolist(), face[2].tolist(),
            face[0].tolist(), face[2].tolist(), face[3].tolist()
        ]

        normal = np.cross(face[1] - face[0], face[2] - face[0])
        norm = np.linalg.norm(normal)
        normal = normal / norm if norm > 0 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        normals = [normal.tolist()] * 6

        face_color = (color[0], color[1], color[2], _SUBSPACE_FACE_ALPHA)
        face_colors = [face_color] * 6

        self.draw_triangles(tri_vertices, normals, face_colors, vp, use_lighting=False, write_depth=False)


def draw_basis_transform(self, vp, original_basis, transformed_basis,
                       show_original=True, show_transformed=True):
    """Visualize basis transformation."""
    colors = [
        (1.0, 0.3, 0.3, 0.8),
        (0.3, 1.0, 0.3, 0.8),
        (0.3, 0.5, 1.0, 0.8),
    ]

    if show_original:
        for i, basis in enumerate(original_basis):
            length = np.linalg.norm(basis)
            if length > 0:
                tip = basis * 2.0 / length if length > 2.0 else basis
                vertices = [[0, 0, 0], tip.tolist()]
                line_colors = [colors[i], colors[i]]
                self.draw_lines(vertices, line_colors, vp, width=2.0)

    if show_transformed:
        for i, basis in enumerate(transformed_basis):
            length = np.linalg.norm(basis)
            if length > 0:
                tip = basis * 2.0 / length if length > 2.0 else basis
                vertices = [[0, 0, 0], tip.tolist()]
                bright_color = (
                    min(1.0, colors[i][0] * 1.5),
                    min(1.0, colors[i][1] * 1.5),
                    min(1.0, colors[i][2] * 1.5),
                    1.0
                )
                line_colors = [bright_color, bright_color]
                self.draw_lines(vertices, line_colors, vp, width=3.0)

                if show_original and i < len(original_basis):
                    orig_tip = original_basis[i]
                    orig_tip = orig_tip * 2.0 / np.linalg.norm(orig_tip) if np.linalg.norm(orig_tip) > 2.0 else orig_tip
                    vertices = [orig_tip.tolist(), tip.tolist()]
                    line_colors = [(0.8, 0.8, 0.2, 0.5), (0.8, 0.8, 0.2, 0.5)]
                    self.draw_lines(vertices, line_colors, vp, width=1.0)


def _create_infinite_grid_program(self):
    """Create the infinite grid shader program."""
    vs, fs = get_grid_shaders()
    return self.ctx.program(vertex_shader=vs, fragment_shader=fs)


def _init_infinite_grid(self):
    """Initialize the infinite grid VAO (no vertex buffer needed)."""
    # Empty VAO - we use gl_VertexID in the vertex shader
    self.grid_vao = self.ctx.vertex_array(self.grid_program, [])


def draw_infinite_grid(self, view_matrix, projection_matrix, plane="xy",
                       scale=1.0, major_scale=5.0, fade_distance=50.0,
                       color_minor=(0.22, 0.24, 0.28, 0.25),
                       color_major=(0.35, 0.37, 0.40, 0.45),
                       color_axis_x=(0.95, 0.45, 0.45, 1.0),
                       color_axis_y=(0.45, 0.95, 0.45, 1.0),
                       color_axis_z=(0.55, 0.60, 1.00, 1.0)):
    """
    Draw an infinite procedural grid on the specified plane.

    Args:
        view_matrix: 4x4 view matrix
        projection_matrix: 4x4 projection matrix
        plane: "xy", "xz", or "yz"
        scale: Base grid cell size
        major_scale: Major grid lines every N cells
        fade_distance: Distance at which grid fades out
        color_minor: Minor grid line color (RGBA)
        color_major: Major grid line color (RGBA)
        color_axis_x: X-axis highlight color
        color_axis_y: Y-axis highlight color
        color_axis_z: Z-axis highlight color
    """
    if not hasattr(self, 'grid_program') or self.grid_program is None:
        return

    self.ctx.enable(moderngl.DEPTH_TEST)
    self.ctx.enable(moderngl.BLEND)
    self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    # Map plane string to integer
    plane_map = {"xy": 0, "xz": 1, "yz": 2}
    plane_idx = plane_map.get(plane.lower(), 0)

    # Set uniforms
    self.grid_program['u_view'].write(view_matrix.astype('f4').tobytes())
    self.grid_program['u_projection'].write(projection_matrix.astype('f4').tobytes())
    self.grid_program['u_scale'].value = float(scale)
    self.grid_program['u_major_scale'].value = float(major_scale)
    self.grid_program['u_fade_distance'].value = float(fade_distance)
    self.grid_program['u_plane'].value = plane_idx

    self.grid_program['u_color_minor'].value = color_minor
    self.grid_program['u_color_major'].value = color_major
    self.grid_program['u_color_axis_x'].value = color_axis_x
    self.grid_program['u_color_axis_y'].value = color_axis_y
    self.grid_program['u_color_axis_z'].value = color_axis_z

    # Render the grid (6 vertices for fullscreen quad)
    with _temporary_depth_mask(self.ctx, False):
        self.grid_vao.render(moderngl.TRIANGLES, vertices=6)


class Gizmos:
    def __init__(self, ctx):
        self.ctx = ctx

        self.line_program = self._create_line_program()
        self.triangle_program = self._create_triangle_program()
        self.point_program = self._create_point_program()
        self.volume_program = self._create_volume_program()

        # Infinite grid program
        self.grid_program = self._create_infinite_grid_program()
        self.grid_vao = None

        self.line_vao = None
        self.line_vbo = None
        self.triangle_vao = None
        self.triangle_vbo = None
        self.point_vao = None
        self.point_vbo = None
        self.volume_vao = None
        self.volume_vbo = None

        self._init_buffers()
        self._init_infinite_grid()

    _create_line_program = _create_line_program
    _create_triangle_program = _create_triangle_program
    _create_point_program = _create_point_program
    _create_volume_program = _create_volume_program
    _create_infinite_grid_program = _create_infinite_grid_program
    _init_buffers = _init_buffers
    _init_infinite_grid = _init_infinite_grid
    draw_lines = draw_lines
    draw_triangles = draw_triangles
    draw_points = draw_points
    draw_volume = draw_volume
    draw_cubic_grid = draw_cubic_grid
    draw_cube = draw_cube
    draw_vector_with_details = draw_vector_with_details
    _draw_arrow_head = _draw_arrow_head
    _draw_vector_components = _draw_vector_components
    draw_vector_span = draw_vector_span
    draw_parallelepiped = draw_parallelepiped
    draw_basis_transform = draw_basis_transform
    draw_grid = draw_grid
    draw_axes = draw_axes
    draw_infinite_grid = draw_infinite_grid
