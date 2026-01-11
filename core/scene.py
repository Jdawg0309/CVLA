import numpy as np
from typing import List, Optional, Tuple

from core.scene_history import SceneHistoryMixin
from core.scene_solvers import SceneSolversMixin
from core.vector import Vector3D


class Scene(SceneHistoryMixin, SceneSolversMixin):
    def __init__(self):
        self.vectors: List[Vector3D] = []
        self.planes: List[dict] = []  # Store plane equations
        self.transformations: List[np.ndarray] = []
        self.matrices: List[dict] = []  # Store linear transformation matrices
        self.selected_object = None
        self.selection_type = None  # 'vector', 'plane', 'matrix'
        # Preview matrix (non-destructive preview from UI)
        self.preview_matrix: Optional[np.ndarray] = None

        # Undo/redo stacks (store snapshots)
        self._undo_stack: List[dict] = []
        self._redo_stack: List[dict] = []
        
    def add_vector(self, vector: Vector3D) -> Vector3D:
        """Add a vector to the scene."""
        # snapshot for undo
        self.push_undo()
        self.vectors.append(vector)
        # Select the newly added vector for convenience
        self.selected_object = vector
        self.selection_type = 'vector'
        return vector
    
    def add_plane(self, equation: List[float], color: Tuple[float, float, float] = (0.2, 0.4, 0.8, 0.3), 
                  label: str = "") -> dict:
        """Add a plane defined by ax + by + cz + d = 0"""
        plane = {
            'equation': np.array(equation, dtype=np.float32),
            'color': color,
            'label': label,
            'visible': True
        }
        self.planes.append(plane)
        return plane
    
    def add_matrix(self, matrix: np.ndarray, label: str = "") -> dict:
        """Add a transformation matrix."""
        # snapshot for undo
        self.push_undo()
        matrix_dict = {
            'matrix': np.array(matrix, dtype=np.float32),
            'label': label,
            'color': (0.8, 0.5, 0.2, 0.6),
            'visible': True,
            'transformations': []  # Store transformations applied by this matrix
        }
        self.matrices.append(matrix_dict)
        return matrix_dict
    
    def remove_vector(self, vector: Vector3D) -> bool:
        """Remove a vector from the scene."""
        if vector in self.vectors:
            self.push_undo()
            self.vectors.remove(vector)
            if self.selected_object is vector:
                self.selected_object = None
                self.selection_type = None
            return True
        return False
    
    def remove_plane(self, plane: dict) -> bool:
        """Remove a plane from the scene."""
        if plane in self.planes:
            self.planes.remove(plane)
            if self.selected_object is plane:
                self.selected_object = None
                self.selection_type = None
            return True
        return False
    
    def remove_matrix(self, matrix_dict: dict) -> bool:
        """Remove a matrix from the scene."""
        if matrix_dict in self.matrices:
            self.push_undo()
            self.matrices.remove(matrix_dict)
            if self.selected_object is matrix_dict:
                self.selected_object = None
                self.selection_type = None
            return True
        return False
    
    def clear_vectors(self):
        """Remove all vectors from the scene."""
        self.vectors.clear()
        if self.selection_type == 'vector':
            self.selected_object = None
            self.selection_type = None
    
    def clear_planes(self):
        """Remove all planes from the scene."""
        self.planes.clear()
        if self.selection_type == 'plane':
            self.selected_object = None
            self.selection_type = None
    
    def clear_matrices(self):
        """Remove all matrices from the scene."""
        self.matrices.clear()
        if self.selection_type == 'matrix':
            self.selected_object = None
            self.selection_type = None
    
    def get_vector_by_label(self, label: str) -> Optional[Vector3D]:
        """Get a vector by its label."""
        for v in self.vectors:
            if v.label == label:
                return v
        return None
    
    def apply_transformation(self, matrix: np.ndarray):
        """Apply a transformation matrix to all vectors."""
        # snapshot for undo
        self.push_undo()
        for v in self.vectors:
            if v.visible:
                v.coords = np.dot(matrix, v.coords)
    
    def apply_matrix_to_selected(self, matrix: np.ndarray):
        """Apply transformation matrix only to selected vector."""
        if self.selected_object and self.selection_type == 'vector':
            self.push_undo()
            self.selected_object.coords = np.dot(matrix, self.selected_object.coords)

    # ----------------
    # Preview helpers
    # ----------------
    def set_preview_matrix(self, matrix: Optional[np.ndarray]):
        if matrix is None:
            self.preview_matrix = None
        else:
            self.preview_matrix = np.array(matrix, dtype=np.float32)
