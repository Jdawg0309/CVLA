"""
Main renderer for CVLA.
"""

import math
import time
import moderngl
import numpy as np

from render.viewconfigs.viewconfig import ViewConfig
from render.gizmos.gizmos import Gizmos


def _nice_step(value):
    """Pick a human-friendly step size (1, 2, 5 * 10^n)."""
    if value <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(value))
    f = value / exp
    if f <= 1.0:
        nice = 1.0
    elif f <= 2.0:
        nice = 2.0
    elif f <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return nice * exp


def _compute_grid_steps(size, density=1.0):
    """Compute major/minor/subminor steps based on semantic scale."""
    density = max(0.1, float(density))
    target = max(0.25, (size / 10.0) / density)
    minor = _nice_step(target)
    major = minor * 5.0
    subminor = minor / 2.0 if minor <= 1.0 else None
    if subminor is not None and subminor < 0.2:
        subminor = 0.2
    return minor, major, subminor


def _render_cubic_environment(self, vp, scene):
    """Render a beautiful 3D cubic environment."""
    theme = self.view.theme

    if self.view.show_grid:
        radius = getattr(self.camera, "radius", 15.0)
        base_size = getattr(self.view, "grid_size", 15)
        size = int(max(6, min(base_size, radius * 0.7)))
        minor_step, major_step, subminor_step = _compute_grid_steps(
            size,
            getattr(self.view, "cubic_grid_density", 1.0)
        )

        if self.show_plane_visuals:
            self._render_cube_faces(vp)

        self._render_cube_corner_indicators(vp)

        # Use theme colors for grid with depth pass
        color_major_depth = (theme.grid_color_major[0], theme.grid_color_major[1],
                             theme.grid_color_major[2], theme.grid_color_major[3] * 0.9)
        color_minor_depth = theme.grid_color_minor
        color_subminor_depth = theme.grid_color_subminor

        self.gizmos.draw_cubic_grid(
            vp,
            size=size,
            major_step=major_step,
            minor_step=minor_step,
            subminor_step=subminor_step,
            color_major=color_major_depth,
            color_minor=color_minor_depth,
            color_subminor=color_subminor_depth,
            depth=True,
            write_depth=False,
        )


def _render_planar_environment(self, vp):
    """Render planar grid environment."""
    theme = self.view.theme
    use_infinite_grid = getattr(self.view, 'use_infinite_grid', True)

    if self.view.show_grid:
        if use_infinite_grid and hasattr(self.gizmos, 'draw_infinite_grid'):
            # Use infinite procedural grid
            view_matrix = self.camera.get_view_matrix()
            proj_matrix = self.camera.get_projection_matrix()
            radius = getattr(self.camera, "radius", 15.0)

            self.gizmos.draw_infinite_grid(
                view_matrix,
                proj_matrix,
                plane=self.view.grid_plane,
                scale=1.0,
                major_scale=5.0,
                fade_distance=radius * 3.0,
                color_minor=theme.grid_color_minor,
                color_major=theme.grid_color_major,
                color_axis_x=theme.axis_color_x,
                color_axis_y=theme.axis_color_y,
                color_axis_z=theme.axis_color_z,
            )
        else:
            # Fallback to CPU-generated grid
            radius = getattr(self.camera, "radius", 15.0)
            base_size = getattr(self.view, "grid_size", 15)
            size = int(max(6, min(base_size, radius * 0.7)))
            minor_step, major_step, subminor_step = _compute_grid_steps(
                size,
                getattr(self.view, "cubic_grid_density", 1.0)
            )

            # Use theme colors for grid with depth pass
            color_major_depth = (theme.grid_color_major[0], theme.grid_color_major[1],
                                 theme.grid_color_major[2], theme.grid_color_major[3] * 0.9)
            color_minor_depth = theme.grid_color_minor
            color_subminor_depth = theme.grid_color_subminor

            self.gizmos.draw_grid(
                vp,
                size=size,
                step=minor_step,
                major_step=major_step,
                sub_step=subminor_step,
                plane=self.view.grid_plane,
                color_major=color_major_depth,
                color_minor=color_minor_depth,
                color_subminor=color_subminor_depth,
                depth=True,
                write_depth=False,
            )


def _render_axes_overlay(self, vp):
    """Draw axes overlays before other geometry."""
    axis_len = float(max(8.0, getattr(self.camera, "radius", 10.0) * 0.7))
    self._render_3d_axes_with_depths(vp, length=axis_len)
    theme = self.view.theme
    self.gizmos.draw_points([[0, 0, 0]], [theme.origin_color], vp,
                            size=6.0, depth=False, write_depth=False)


def _render_cube_faces(self, vp):
    """Draw translucent faces of the bounding cube."""
    size = float(self.view.grid_size)

    faces = [
        {"normal": [0, 0, 1], "vertices": [
            [-size, -size, size], [size, -size, size],
            [size, size, size], [-size, size, size]
        ]},
        {"normal": [0, 0, -1], "vertices": [
            [-size, -size, -size], [-size, size, -size],
            [size, size, -size], [size, -size, -size]
        ]},
        {"normal": [0, 1, 0], "vertices": [
            [-size, size, -size], [-size, size, size],
            [size, size, size], [size, size, -size]
        ]},
        {"normal": [0, -1, 0], "vertices": [
            [-size, -size, -size], [size, -size, -size],
            [size, -size, size], [-size, -size, size]
        ]},
        {"normal": [1, 0, 0], "vertices": [
            [size, -size, -size], [size, size, -size],
            [size, size, size], [size, -size, size]
        ]},
        {"normal": [-1, 0, 0], "vertices": [
            [-size, -size, -size], [-size, -size, size],
            [-size, size, size], [-size, size, -size]
        ]}
    ]

    for i, face in enumerate(faces):
        vertices = face["vertices"]
        normal = face["normal"]

        tri_vertices = [
            vertices[0], vertices[1], vertices[2],
            vertices[0], vertices[2], vertices[3]
        ]
        normals = [normal] * 6

        try:
            cfg_colors = getattr(self.view, 'cube_face_colors', None)
            if cfg_colors:
                color = tuple(cfg_colors[i % len(cfg_colors)])
            else:
                color = self.cube_face_colors[i % len(self.cube_face_colors)]
        except Exception:
            color = self.cube_face_colors[i % len(self.cube_face_colors)]

        try:
            opacity = float(getattr(self.view, 'cube_face_opacity', color[3]))
            color = (color[0], color[1], color[2], opacity)
        except Exception:
            pass

        self.gizmos.draw_triangles(
            tri_vertices, normals, [color] * 6,
            vp, use_lighting=False, write_depth=False
        )

        border_vertices = [
            vertices[0], vertices[1],
            vertices[1], vertices[2],
            vertices[2], vertices[3],
            vertices[3], vertices[0]
        ]
        border_color = (color[0] * 1.5, color[1] * 1.5, color[2] * 1.5, 0.8)
        self.gizmos.draw_lines(
            border_vertices, [border_color] * 8,
            vp, width=1.0, write_depth=False
        )


def _render_cube_corner_indicators(self, vp):
    """Draw indicators at cube corners showing axis directions."""
    theme = self.view.theme
    size = float(self.view.grid_size)
    corners = [
        [size, size, size],
        [size, size, -size],
        [size, -size, size],
        [size, -size, -size],
        [-size, size, size],
        [-size, size, -size],
        [-size, -size, size],
        [-size, -size, -size],
    ]

    corner_color = theme.corner_color
    self.gizmos.draw_points(corners, [corner_color] * len(corners), vp,
                            size=4.0, depth=True, write_depth=False)

    # Use theme axis colors with reduced alpha for corner indicators
    axis_x_color = (theme.axis_color_x[0], theme.axis_color_x[1], theme.axis_color_x[2], 0.7)
    axis_y_color = (theme.axis_color_y[0], theme.axis_color_y[1], theme.axis_color_y[2], 0.7)
    axis_z_color = (theme.axis_color_z[0], theme.axis_color_z[1], theme.axis_color_z[2], 0.7)

    axis_length = size * 0.2
    for corner in corners:
        x_end = [corner[0] + axis_length, corner[1], corner[2]]
        self.gizmos.draw_lines(
            [corner, x_end],
            [axis_x_color, axis_x_color],
            vp, width=1.5, write_depth=False
        )

        y_end = [corner[0], corner[1] + axis_length, corner[2]]
        self.gizmos.draw_lines(
            [corner, y_end],
            [axis_y_color, axis_y_color],
            vp, width=1.5, write_depth=False
        )

        z_end = [corner[0], corner[1], corner[2] + axis_length]
        self.gizmos.draw_lines(
            [corner, z_end],
            [axis_z_color, axis_z_color],
            vp, width=1.5, write_depth=False
        )


def _render_3d_axes_with_depths(self, vp, length=None):
    """Render 3D axes with depth cues."""
    theme = self.view.theme

    if length is None:
        length = max(10.0, self.view.grid_size * 0.75)

    def _highlight(color, boost=0.25, min_alpha=0.9):
        r = min(1.0, color[0] + boost)
        g = min(1.0, color[1] + boost)
        b = min(1.0, color[2] + boost)
        a = max(min_alpha, color[3])
        return (r, g, b, a)

    axes = [
        {"points": [[0, 0, 0], [length, 0, 0]], "color": _highlight(theme.axis_color_x, boost=0.3)},
        {"points": [[0, 0, 0], [0, length, 0]], "color": _highlight(theme.axis_color_y, boost=0.3)},
        {"points": [[0, 0, 0], [0, 0, length]], "color": _highlight(theme.axis_color_z, boost=0.3)},
    ]

    for axis in axes:
        self.gizmos.draw_lines(
            axis["points"],
            [axis["color"], axis["color"]],
            vp, width=6.5, depth=False, write_depth=False
        )

    size = max(20.0, self.view.grid_size) * 1.2
    faint_color = _highlight(theme.faint_axis_color, boost=0.1, min_alpha=0.5)
    faint_axes = [
        {"points": [[-size, 0, 0], [size, 0, 0]], "color": faint_color},
        {"points": [[0, -size, 0], [0, size, 0]], "color": faint_color},
        {"points": [[0, 0, -size], [0, 0, size]], "color": faint_color},
    ]

    for axis in faint_axes:
        self.gizmos.draw_lines(
            axis["points"],
            [axis["color"], axis["color"]],
            vp, width=3.0, depth=False, write_depth=False
        )

    self._draw_axis_cones(vp, length)


def _draw_axis_cones(self, vp, length):
    """Draw 3D cones at axis tips."""
    theme = self.view.theme
    cone_height = length * 0.15
    cone_radius = length * 0.05

    axes = [
        {"tip": [length, 0, 0], "direction": [1, 0, 0], "color": theme.axis_color_x},
        {"tip": [0, length, 0], "direction": [0, 1, 0], "color": theme.axis_color_y},
        {"tip": [0, 0, length], "direction": [0, 0, 1], "color": theme.axis_color_z},
    ]

    for axis in axes:
        tip = np.array(axis["tip"])
        direction = np.array(axis["direction"])
        base = tip - direction * cone_height

        cone_verts = []
        cone_norms = []
        cone_colors = []
        segments = 12

        if abs(direction[0]) < 0.9:
            perp1 = np.array([1.0, 0.0, 0.0])
        else:
            perp1 = np.array([0.0, 1.0, 0.0])

        perp2 = np.cross(direction, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        perp1 = np.cross(perp2, direction)

        for i in range(segments):
            angle1 = 2 * np.pi * i / segments
            angle2 = 2 * np.pi * (i + 1) / segments

            p1 = base + cone_radius * (np.cos(angle1) * perp1 + np.sin(angle1) * perp2)
            p2 = base + cone_radius * (np.cos(angle2) * perp1 + np.sin(angle2) * perp2)

            cone_verts.extend(base.tolist())
            cone_verts.extend(p1.tolist())
            cone_verts.extend(tip.tolist())

            v1 = p1 - base
            v2 = tip - base
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)

            cone_norms.extend([normal.tolist()] * 3)
            cone_colors.extend([axis["color"]] * 3)

        self.gizmos.draw_triangles(cone_verts, cone_norms, cone_colors, vp, write_depth=False)


def _render_linear_algebra_visuals(self, scene, vp):
    """Render linear algebra visualizations."""
    try:
        if hasattr(scene, 'preview_matrix') and scene.preview_matrix is not None:
            matrix = scene.preview_matrix
            basis_vectors = [
                np.array([1, 0, 0], dtype='f4'),
                np.array([0, 1, 0], dtype='f4'),
                np.array([0, 0, 1], dtype='f4')
            ]
            transformed_basis = []
            for basis in basis_vectors:
                if matrix.shape == (2, 2):
                    vec2 = matrix @ np.array([basis[0], basis[1]], dtype='f4')
                    transformed = np.array([vec2[0], vec2[1], basis[2]], dtype='f4')
                elif matrix.shape == (3, 3):
                    transformed = matrix @ basis
                elif matrix.shape == (4, 4):
                    point = np.array([basis[0], basis[1], basis[2], 1.0])
                    transformed_h = matrix @ point
                    transformed = transformed_h[:3] / transformed_h[3]
                else:
                    transformed = basis
                transformed_basis.append(transformed)

            self.gizmos.draw_basis_transform(vp, basis_vectors, transformed_basis,
                                             show_original=True, show_transformed=True)
    except Exception:
        pass

    if getattr(scene, 'vector_span', None) is not None:
        v1, v2 = scene.vector_span
        self.gizmos.draw_vector_span(vp, v1, v2, scale=self.vector_scale)
    elif self.show_vector_spans and len(scene.vectors) >= 2:
        self.gizmos.draw_vector_span(vp, scene.vectors[0], scene.vectors[1], scale=self.vector_scale)

    if len(scene.vectors) >= 3:
        self.gizmos.draw_parallelepiped(vp, scene.vectors[:3], scale=self.vector_scale)

    self._render_matrix_3d_plot(scene, vp, force=True)

    if not getattr(scene, 'show_matrix_plot', False):
        for matrix_dict in scene.matrices:
            if not matrix_dict['visible']:
                continue

            matrix = matrix_dict['matrix']
            basis_vectors = [
                np.array([1, 0, 0], dtype='f4'),
                np.array([0, 1, 0], dtype='f4'),
                np.array([0, 0, 1], dtype='f4')
            ]

            transformed_basis = []
            for basis in basis_vectors:
                if matrix.shape == (2, 2):
                    vec2 = matrix @ np.array([basis[0], basis[1]], dtype='f4')
                    transformed = np.array([vec2[0], vec2[1], basis[2]], dtype='f4')
                elif matrix.shape == (3, 3):
                    transformed = matrix @ basis
                elif matrix.shape == (4, 4):
                    point = np.array([basis[0], basis[1], basis[2], 1.0])
                    transformed_h = matrix @ point
                    transformed = transformed_h[:3] / transformed_h[3]
                else:
                    continue
                transformed_basis.append(transformed)

            self.gizmos.draw_basis_transform(
                vp,
                basis_vectors,
                transformed_basis,
                show_original=True,
                show_transformed=True
            )


def _render_matrix_3d_plot(self, scene, vp, force=False, only_nonsquare=False):
    """Render matrix values as a 3D point plot when enabled."""
    if not force and not getattr(scene, 'show_matrix_plot', False):
        return

    for matrix_dict in scene.matrices:
        if not matrix_dict.get('visible', True):
            continue
        matrix = matrix_dict.get('matrix')
        if matrix is None or matrix.ndim != 2:
            continue
        rows, cols = matrix.shape
        if only_nonsquare and rows == cols:
            continue
        if rows == 0 or cols == 0:
            continue

        values = matrix.flatten()
        min_val = float(values.min())
        max_val = float(values.max())
        range_val = max(1e-5, max_val - min_val)
        max_abs = max(1e-5, float(np.max(np.abs(values))))
        spacing = 1.6
        z_scale = 1.2 / max(1.0, max_abs / 3.0)

        points = []
        colors = []
        for i in range(rows):
            for j in range(cols):
                value = float(matrix[i, j])
                norm = (value - min_val) / range_val
                x = (j - (cols - 1) / 2.0) * spacing
                y = (i - (rows - 1) / 2.0) * spacing
                z = value * z_scale
                points.append([x, y, z])
                colors.append(_matrix_point_color(norm))

        if points:
            self.gizmos.draw_points(points, colors, vp, size=18.0, depth=True)


def _matrix_point_color(t):
    """Map normalized value to a warm-to-cool gradient."""
    return (
        min(1.0, max(0.0, 0.6 + t * 0.4)),
        min(1.0, max(0.0, 0.3 + (1.0 - t) * 0.4)),
        min(1.0, max(0.0, 1.0 - t * 0.5)),
        1.0
    )


def _render_tensor_faces(self, scene, vp):
    """Render triangle meshes for rank-2 tensors shaped (N, 3)."""
    faces = getattr(scene, "tensor_faces", None)
    if not faces:
        return

    for mesh in faces:
        vertices = mesh.get("vertices")
        normals = mesh.get("normals")
        colors = mesh.get("colors")
        if vertices is None or normals is None or colors is None:
            continue
        self.gizmos.draw_triangles(vertices, normals, colors, vp, use_lighting=False)

        try:
            outline_color = (0.0, 0.0, 0.0, 0.9)
            outline_lines = []
            outline_colors = []
            tri_vertices = vertices.reshape(-1, 3)
            tri_count = len(tri_vertices) // 3
            for i in range(tri_count):
                v0 = tri_vertices[i * 3]
                v1 = tri_vertices[i * 3 + 1]
                v2 = tri_vertices[i * 3 + 2]
                edges = [(v0, v1), (v1, v2), (v2, v0)]
                for a, b in edges:
                    outline_lines.extend([a.tolist(), b.tolist()])
                    outline_colors.extend([outline_color, outline_color])
            if outline_lines:
                self.gizmos.draw_lines(outline_lines, outline_colors, vp,
                                       width=2.5, depth=True, write_depth=False)
        except Exception:
            pass


def _render_vectors_with_enhancements(self, scene, vp):
    """Render all vectors with enhanced visualizations."""
    self.vector_scale = float(self.view.vector_scale)

    for vector in scene.vectors:
        if vector.visible:
            is_selected = (vector is scene.selected_object and
                           scene.selection_type == 'vector')

            self.gizmos.draw_vector_with_details(
                vp, vector, is_selected, self.vector_scale,
                show_components=self.show_vector_components,
                show_span=False,
                depth=False
            )

            if self.camera.mode_2d:
                self._render_vector_projections(vector, vp)


def _render_vector_projections(self, vector, vp):
    """Render vector projection lines in 2D mode."""
    tip = vector.coords * self.vector_scale

    if self.camera.view_preset == "xy":
        proj_tip = np.array([tip[0], tip[1], 0])
        proj_color = (vector.color[0], vector.color[1], vector.color[2], 0.3)

        line = [[tip[0], tip[1], tip[2]], [proj_tip[0], proj_tip[1], proj_tip[2]]]
        self.gizmos.draw_lines(line, [proj_color, proj_color], vp, width=1.0,
                               depth=False, write_depth=False)
        self.gizmos.draw_points([proj_tip], [proj_color], vp, size=4.0, depth=False, write_depth=False)

    elif self.camera.view_preset == "xz":
        proj_tip = np.array([tip[0], 0, tip[2]])
        proj_color = (vector.color[0], vector.color[1], vector.color[2], 0.3)

        line = [[tip[0], tip[1], tip[2]], [proj_tip[0], proj_tip[1], proj_tip[2]]]
        self.gizmos.draw_lines(line, [proj_color, proj_color], vp, width=1.0,
                               depth=False, write_depth=False)
        self.gizmos.draw_points([proj_tip], [proj_color], vp, size=4.0, depth=False, write_depth=False)

    elif self.camera.view_preset == "yz":
        proj_tip = np.array([0, tip[1], tip[2]])
        proj_color = (vector.color[0], vector.color[1], vector.color[2], 0.3)

        line = [[tip[0], tip[1], tip[2]], [proj_tip[0], proj_tip[1], proj_tip[2]]]
        self.gizmos.draw_lines(line, [proj_color, proj_color], vp, width=1.0,
                               depth=False, write_depth=False)
        self.gizmos.draw_points([proj_tip], [proj_color], vp, size=4.0, depth=False, write_depth=False)


def _render_selection_highlight(self, vector, vp):
    """Render special highlight for selected vector."""
    tip = vector.coords * self.vector_scale

    pulse = (np.sin(time.time() * 5) * 0.2 + 0.8)

    circle_verts = []
    circle_colors = []
    segments = 24
    radius = 0.4 * pulse

    for i in range(segments):
        angle1 = 2 * np.pi * i / segments
        angle2 = 2 * np.pi * (i + 1) / segments

        p1 = tip + radius * np.array([np.cos(angle1), np.sin(angle1), 0])
        p2 = tip + radius * np.array([np.cos(angle2), np.sin(angle2), 0])

        circle_verts.extend(p1.tolist())
        circle_verts.extend(p2.tolist())

        pulse_color = (1.0, 1.0, 0.2, 0.7 * pulse)
        circle_colors.extend([pulse_color, pulse_color])

    self.gizmos.draw_lines(circle_verts, circle_colors, vp, width=2.0,
                          depth=False, write_depth=False)

    origin_line = [[0, 0, 0], tip.tolist()]
    highlight_color = (1.0, 1.0, 0.2, 0.8)
    self.gizmos.draw_lines(origin_line, [highlight_color, highlight_color], vp, width=2.0,
                          depth=False, write_depth=False)

    sphere_color = (1.0, 1.0, 0.2, 0.9)
    self.gizmos.draw_points([tip.tolist()], [sphere_color], vp, size=14.0,
                            depth=False, write_depth=False)


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
    if image_data is None:
        return

    try:
        matrix, channels = _resolve_image_matrix(image_data)
    except Exception as e:
        print(f"[CVLA] Image resolve error: {e}")
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
        self.gizmos.draw_triangles(verts, norms, colors, vp, use_lighting=False, depth=False,
                                   write_depth=False)


class Renderer:
    def __init__(self, ctx, camera, view=None):
        self.ctx = ctx
        self.camera = camera
        self.view = view or ViewConfig()

        self.gizmos = Gizmos(ctx)

        self.vector_scale = 3.0
        self.show_vector_labels = True
        self.show_plane_visuals = True
        self.show_vector_components = True
        self.show_vector_spans = False

        self._image_plane_cache = {"key": None, "batches": []}

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self._vp_cache = None
        self._vp_cache_dirty = True

        self.cube_face_colors = [
            (0.3, 0.3, 0.8, 0.05),
            (0.8, 0.3, 0.3, 0.05),
            (0.3, 0.8, 0.3, 0.05),
            (0.8, 0.8, 0.3, 0.05),
            (0.8, 0.3, 0.8, 0.05),
            (0.3, 0.8, 0.8, 0.05),
        ]

    def _get_view_projection(self):
        """Get cached view-projection matrix."""
        if self._vp_cache_dirty or self._vp_cache is None:
            self._vp_cache = self.camera.vp()
            self._vp_cache_dirty = False
        return self._vp_cache

    def update_view(self, view_config):
        """Update view configuration."""
        self.view = view_config
        self._vp_cache_dirty = True

        if hasattr(view_config, 'vector_scale'):
            self.vector_scale = view_config.vector_scale

        if hasattr(view_config, 'show_plane_visuals'):
            self.show_plane_visuals = view_config.show_plane_visuals

        try:
            if hasattr(view_config, 'cube_face_colors'):
                self.cube_face_colors = [tuple(c) for c in view_config.cube_face_colors]
        except Exception:
            pass

    def render(self, scene, image_data=None, show_image_on_grid=False, image_render_scale=1.0,
               image_color_mode="grayscale", image_color_source=None,
               image_render_mode="plane", show_image_grid_overlay=False):
        """Main rendering method."""
        self._clear_with_gradient()

        vp = self._get_view_projection()
        self._render_axes_overlay(vp)

        self._render_linear_algebra_visuals(scene, vp)
        self._render_tensor_faces(scene, vp)
        self._render_vectors_with_enhancements(scene, vp)

        if scene.selected_object and scene.selection_type == 'vector':
            self._render_selection_highlight(scene.selected_object, vp)

        overlay_grid_size = None
        overlay_matrix = None
        if image_data is not None and show_image_on_grid:
            try:
                overlay_matrix, _ = _resolve_image_matrix(image_data)
            except Exception:
                overlay_matrix = None

            self.draw_image_plane(
                image_data,
                vp,
                scale=image_render_scale,
                color_mode=image_color_mode,
                color_source=image_color_source,
                render_mode=image_render_mode,
            )

            if show_image_grid_overlay and overlay_matrix is not None:
                height, width = overlay_matrix.shape[:2]
                overlay_grid_size = int(
                    max(1.0, max(width, height) * image_render_scale * 0.5)
                )

        if self.view.grid_mode == "cube":
            self._render_cubic_environment(vp, scene)
        else:
            self._render_planar_environment(vp)

        if overlay_grid_size is not None:
            try:
                self.gizmos.draw_grid(
                    vp,
                    size=overlay_grid_size,
                    step=1,
                    plane="xy",
                    color_major=(0.35, 0.35, 0.4, 0.6),
                    color_minor=(0.22, 0.22, 0.25, 0.4),
                    depth=False,
                    write_depth=False,
                )
            except Exception:
                pass

    def _clear_with_gradient(self):
        """Clear with a subtle gradient background."""
        theme = self.view.theme
        if self.view.grid_mode == "cube":
            self.ctx.clear(
                color=theme.background_color_cube,
                depth=1.0
            )
        else:
            self.ctx.clear(
                color=theme.background_color,
                depth=1.0
            )

    draw_image_plane = draw_image_plane
    _image_color = _image_color
    _render_cubic_environment = _render_cubic_environment
    _render_planar_environment = _render_planar_environment
    _render_cube_faces = _render_cube_faces
    _render_cube_corner_indicators = _render_cube_corner_indicators
    _render_3d_axes_with_depths = _render_3d_axes_with_depths
    _draw_axis_cones = _draw_axis_cones
    _render_linear_algebra_visuals = _render_linear_algebra_visuals
    _render_tensor_faces = _render_tensor_faces
    _render_matrix_3d_plot = _render_matrix_3d_plot
    _render_vectors_with_enhancements = _render_vectors_with_enhancements
    _render_vector_projections = _render_vector_projections
    _render_axes_overlay = _render_axes_overlay
    _render_selection_highlight = _render_selection_highlight
