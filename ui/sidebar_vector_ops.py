"""
Sidebar vector operation helpers.
"""

import numpy as np
from core.vector import Vector3D


def _duplicate_vector(self, scene, vector):
    """Duplicate a vector and add to the scene with a new label."""
    try:
        new_coords = vector.coords.copy()
        base = vector.label or "v"
        idx = 1
        new_label = f"{base}_copy{idx}"
        existing = {v.label for v in scene.vectors}
        while new_label in existing:
            idx += 1
            new_label = f"{base}_copy{idx}"

        v = Vector3D(new_coords, color=vector.color, label=new_label)
        scene.add_vector(v)
    except Exception:
        pass


def _add_vector(self, scene):
    """Add a new vector to the scene."""
    if not self.vec_name:
        self.vec_name = f"v{self.next_vector_id}"

    v = Vector3D(
        np.array(self.vec_input, dtype=np.float32),
        color=self.vec_color,
        label=self.vec_name
    )
    scene.add_vector(v)

    self.next_vector_id += 1
    self.vec_name = ""
    self.vec_color = self._get_next_color()
    self.vec_input = [1.0, 0.0, 0.0]


def _add_vectors(self, scene, idx1, idx2):
    """Add two vectors."""
    if len(scene.vectors) >= 2:
        v1 = scene.vectors[idx1]
        v2 = scene.vectors[idx2]
        result = v1.coords + v2.coords

        name = f"{v1.label}+{v2.label}"
        v = Vector3D(result, color=self._get_next_color(), label=name)
        scene.add_vector(v)


def _subtract_vectors(self, scene, idx1, idx2):
    """Subtract two vectors."""
    if len(scene.vectors) >= 2:
        v1 = scene.vectors[idx1]
        v2 = scene.vectors[idx2]
        result = v1.coords - v2.coords

        name = f"{v1.label}-{v2.label}"
        v = Vector3D(result, color=self._get_next_color(), label=name)
        scene.add_vector(v)


def _cross_vectors(self, scene, idx1, idx2):
    """Cross product of two vectors."""
    if len(scene.vectors) >= 2:
        v1 = scene.vectors[idx1]
        v2 = scene.vectors[idx2]
        result = np.cross(v1.coords, v2.coords)

        name = f"{v1.label}x{v2.label}"
        v = Vector3D(result, color=self._get_next_color(), label=name)
        scene.add_vector(v)


def _dot_vectors(self, scene, idx1, idx2):
    """Dot product of two vectors."""
    if len(scene.vectors) >= 2:
        v1 = scene.vectors[idx1]
        v2 = scene.vectors[idx2]
        dot = np.dot(v1.coords, v2.coords)

        self.operation_result = {
            'type': 'dot_product',
            'value': dot,
            'vectors': [v1.label, v2.label]
        }
