"""
Cubic face and corner rendering helpers.
"""


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
            vp, use_lighting=False
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
            vp, width=1.0
        )


def _render_cube_corner_indicators(self, vp):
    """Draw indicators at cube corners showing axis directions."""
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

    corner_color = (0.8, 0.8, 0.9, 0.6)
    self.gizmos.draw_points(corners, [corner_color] * len(corners), vp, size=4.0)

    axis_length = size * 0.2
    for corner in corners:
        x_end = [corner[0] + axis_length, corner[1], corner[2]]
        self.gizmos.draw_lines(
            [corner, x_end],
            [(1.0, 0.3, 0.3, 0.7), (1.0, 0.3, 0.3, 0.7)],
            vp, width=1.5
        )

        y_end = [corner[0], corner[1] + axis_length, corner[2]]
        self.gizmos.draw_lines(
            [corner, y_end],
            [(0.3, 1.0, 0.3, 0.7), (0.3, 1.0, 0.3, 0.7)],
            vp, width=1.5
        )

        z_end = [corner[0], corner[1], corner[2] + axis_length]
        self.gizmos.draw_lines(
            [corner, z_end],
            [(0.3, 0.5, 1.0, 0.7), (0.3, 0.5, 1.0, 0.7)],
            vp, width=1.5
        )
