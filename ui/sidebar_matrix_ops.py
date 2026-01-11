"""
Sidebar matrix operation helpers.
"""

import numpy as np
from core.vector import Vector3D


def _resize_matrix(self):
    """Resize matrix input based on selected size."""
    new_size = self.matrix_size

    new_matrix = []
    for i in range(new_size):
        row = []
        for j in range(new_size):
            if i < len(self.matrix_input) and j < len(self.matrix_input[0]):
                row.append(self.matrix_input[i][j])
            else:
                row.append(1.0 if i == j else 0.0)
        new_matrix.append(row)

    self.matrix_input = new_matrix


def _add_matrix(self, scene):
    """Add matrix to scene."""
    matrix = np.array(self.matrix_input, dtype=np.float32)
    mat_dict = scene.add_matrix(matrix, label=self.matrix_name)

    scene.selected_object = mat_dict
    scene.selection_type = 'matrix'

    self.operation_result = {
        'type': 'add_matrix',
        'label': self.matrix_name,
        'shape': mat_dict['matrix'].shape
    }

    self.show_matrix_editor = False


def _apply_matrix_to_selected(self, scene):
    """Apply matrix to selected vector."""
    if scene.selected_object and scene.selection_type == 'vector':
        matrix = np.array(self.matrix_input, dtype=np.float32)
        scene.apply_matrix_to_selected(matrix)


def _compute_null_space(self, scene, matrix):
    """Compute null space for `matrix` and add basis vectors to the scene.

    Only 3-element vectors are added as `Vector3D`; other sizes are skipped.
    """
    try:
        ns = scene.compute_null_space(np.array(matrix, dtype=np.float32))
        if ns is None or ns.size == 0:
            self.operation_result = {'type': 'null_space', 'vectors': []}
            return

        added = []
        arr = np.array(ns)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        for i, vec in enumerate(arr):
            v_arr = np.array(vec, dtype=np.float32).flatten()
            if v_arr.size == 3:
                name = f"ns_{i+1}"
                v = Vector3D(v_arr, color=self._get_next_color(), label=name)
                scene.add_vector(v)
                added.append(name)

        self.operation_result = {'type': 'null_space', 'vectors': added}
    except Exception as e:
        self.operation_result = {'error': str(e)}


def _compute_column_space(self, scene, matrix):
    """Compute column space for `matrix` and add basis vectors to the scene.

    Only 3-element vectors are added as `Vector3D`; other sizes are skipped.
    """
    try:
        cs = scene.compute_column_space(np.array(matrix, dtype=np.float32))
        if cs is None or cs.size == 0:
            self.operation_result = {'type': 'column_space', 'vectors': []}
            return

        added = []
        cols = np.array(cs)
        if cols.ndim == 2:
            for i in range(cols.shape[1]):
                v_arr = cols[:, i].astype(np.float32).flatten()
                if v_arr.size == 3:
                    name = f"cs_{i+1}"
                    v = Vector3D(v_arr, color=self._get_next_color(), label=name)
                    scene.add_vector(v)
                    added.append(name)

        self.operation_result = {'type': 'column_space', 'vectors': added}
    except Exception as e:
        self.operation_result = {'error': str(e)}
