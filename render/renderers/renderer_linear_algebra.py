"""
Linear algebra visualization helpers.
"""

import numpy as np


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
                if matrix.shape == (3, 3):
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

    if self.show_vector_spans and len(scene.vectors) >= 2:
        self.gizmos.draw_vector_span(vp, scene.vectors[0], scene.vectors[1])

    if len(scene.vectors) >= 3:
        self.gizmos.draw_parallelepiped(vp, scene.vectors[:3])

    self._render_matrix_3d_plot(scene, vp)

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
                if matrix.shape == (3, 3):
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


def _render_matrix_3d_plot(self, scene, vp):
    """Render matrix values as a 3D point plot when enabled."""
    if not getattr(scene, 'show_matrix_plot', False):
        return

    for matrix_dict in scene.matrices:
        if not matrix_dict.get('visible', True):
            continue
        matrix = matrix_dict.get('matrix')
        if matrix is None or matrix.ndim != 2:
            continue
        rows, cols = matrix.shape
        if rows == 0 or cols == 0:
            continue

        values = matrix.flatten()
        min_val = float(values.min())
        max_val = float(values.max())
        range_val = max(1e-5, max_val - min_val)
        spacing = 1.25

        points = []
        colors = []
        for i in range(rows):
            for j in range(cols):
                value = float(matrix[i, j])
                norm = (value - min_val) / range_val
                x = (j - (cols - 1) / 2.0) * spacing
                y = (i - (rows - 1) / 2.0) * spacing
                z = value * 0.5
                points.append([x, y, z])
                colors.append(_matrix_point_color(norm))

        if points:
            self.gizmos.draw_points(points, colors, vp, size=10.0, depth=True)


def _matrix_point_color(t):
    """Map normalized value to a warm-to-cool gradient."""
    return (
        min(1.0, max(0.0, 0.6 + t * 0.4)),
        min(1.0, max(0.0, 0.3 + (1.0 - t) * 0.4)),
        min(1.0, max(0.0, 1.0 - t * 0.5)),
        1.0
    )
