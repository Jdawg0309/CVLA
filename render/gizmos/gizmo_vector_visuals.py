"""
Vector visualization helpers.
"""

import numpy as np


def draw_vector_span(self, vp, vector1, vector2, color=(0.2, 0.4, 0.8, 0.3)):
    """Draw the span (parallelogram) between two vectors."""
    v1 = vector1.coords
    v2 = vector2.coords

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

    colors = []
    for i in range(6):
        if i < 3:
            colors.append((color[0], color[1], color[2], color[3] * 0.8))
        else:
            colors.append((color[0], color[1], color[2], color[3] * 0.6))

    self.draw_triangles(tri_vertices, normals, colors, vp, use_lighting=True)

    border_vertices = [
        vertices[0], vertices[1],
        vertices[1], vertices[2],
        vertices[2], vertices[3],
        vertices[3], vertices[0]
    ]
    border_colors = [(color[0] * 0.7, color[1] * 0.7, color[2] * 0.7, 1.0)] * 8
    self.draw_lines(border_vertices, border_colors, vp, width=1.5)


def draw_parallelepiped(self, vp, vectors, color=(0.3, 0.6, 0.9, 0.2)):
    """Draw parallelepiped spanned by three vectors."""
    if len(vectors) < 3:
        return

    v1, v2, v3 = vectors[0].coords, vectors[1].coords, vectors[2].coords

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

    edge_colors = [(color[0] * 0.8, color[1] * 0.8, color[2] * 0.8, 1.0)] * len(edge_vertices)
    self.draw_lines(edge_vertices, edge_colors, vp, width=2.0)

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

        face_color = (color[0], color[1], color[2], color[3] * 0.15)
        face_colors = [face_color] * 6

        self.draw_triangles(tri_vertices, normals, face_colors, vp, use_lighting=False)


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
