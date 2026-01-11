"""
Sidebar matrix operation helpers.
"""

import numpy as np
from state.actions import AddVector


def _compute_null_space(self, matrix):
    """Compute null space for `matrix` and add basis vectors to the scene.

    Only 3-element vectors are added as `Vector3D`; other sizes are skipped.
    """
    try:
        mat_np = np.array(matrix, dtype=np.float32)
        _, s_vals, vt = np.linalg.svd(mat_np)
        ns = vt[np.abs(s_vals) < 1e-10]
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
                if self._dispatch:
                    self._dispatch(AddVector(
                        coords=tuple(v_arr.tolist()),
                        color=self._get_next_color(),
                        label=name,
                    ))
                added.append(name)

        self.operation_result = {'type': 'null_space', 'vectors': added}
    except Exception as e:
        self.operation_result = {'error': str(e)}


def _compute_column_space(self, matrix):
    """Compute column space for `matrix` and add basis vectors to the scene.

    Only 3-element vectors are added as `Vector3D`; other sizes are skipped.
    """
    try:
        mat_np = np.array(matrix, dtype=np.float32)
        u_vals, s_vals, _ = np.linalg.svd(mat_np)
        cs = u_vals[:, np.abs(s_vals) > 1e-10]
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
                    if self._dispatch:
                        self._dispatch(AddVector(
                            coords=tuple(v_arr.tolist()),
                            color=self._get_next_color(),
                            label=name,
                        ))
                    added.append(name)

        self.operation_result = {'type': 'column_space', 'vectors': added}
    except Exception as e:
        self.operation_result = {'error': str(e)}
