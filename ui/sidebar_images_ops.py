"""
Sidebar image operation helpers.
"""

import numpy as np
from state.actions import AddVector


def _add_image_as_vectors(self, image_matrix):
    """Add image matrix rows as 3D vectors for visualization."""
    matrix = image_matrix.as_matrix()
    h, w = matrix.shape[:2]

    if h > 8 or w > 8:
        return

    for i in range(min(h, 8)):
        row = matrix[i, :min(w, 3)]
        if len(row) < 3:
            row = np.pad(row, (0, 3 - len(row)))

        coords = np.array(row, dtype=np.float32) * 5
        color = self._get_next_color()
        label = f"img_row{i}"
        if self._dispatch:
            self._dispatch(AddVector(
                coords=tuple(coords.tolist()),
                color=color,
                label=label,
            ))
