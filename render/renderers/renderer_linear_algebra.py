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
