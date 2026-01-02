import numpy as np
import copy
from .vector import Vector3D
from typing import List, Optional, Tuple


class Scene:
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

    # ----------------
    # Undo/redo (simple snapshots)
    # ----------------
    def snapshot(self) -> dict:
        """Return a deep copy snapshot of the scene state relevant for undo."""
        return {
            'vectors': [
                {
                    'coords': v.coords.copy(),
                    'color': v.color,
                    'label': v.label,
                    'visible': v.visible,
                    'metadata': copy.deepcopy(v.metadata),
                }
                for v in self.vectors
            ],
            'matrices': [
                {
                    'matrix': m['matrix'].copy(),
                    'label': m.get('label', ''),
                    'visible': m.get('visible', True),
                }
                for m in self.matrices
            ],
            'selected_object': None,
            'selection_type': self.selection_type
        }

    def apply_snapshot(self, snap: dict):
        """Restore a snapshot into the scene (non-destructive replacement)."""
        # Replace vectors
        self.vectors.clear()
        for v in snap.get('vectors', []):
            vec = Vector3D(np.array(v['coords'], dtype=np.float32), color=v.get('color', (0.8,0.2,0.2)), label=v.get('label',''))
            vec.visible = v.get('visible', True)
            vec.metadata = copy.deepcopy(v.get('metadata', {}))
            self.vectors.append(vec)

        # Replace matrices
        self.matrices.clear()
        for m in snap.get('matrices', []):
            md = {
                'matrix': np.array(m['matrix'], dtype=np.float32),
                'label': m.get('label', ''),
                'visible': m.get('visible', True),
                'color': (0.8, 0.5, 0.2, 0.6),
                'transformations': []
            }
            self.matrices.append(md)

        # selection is cleared (we don't attempt to pointer-restore)
        self.selected_object = None
        self.selection_type = None

    def push_undo(self):
        try:
            snap = self.snapshot()
            self._undo_stack.append(snap)
            # clear redo stack
            self._redo_stack.clear()
        except Exception:
            pass

    def undo(self):
        if not self._undo_stack:
            return
        try:
            current = self.snapshot()
            last = self._undo_stack.pop()
            self._redo_stack.append(current)
            self.apply_snapshot(last)
        except Exception:
            pass

    def redo(self):
        if not self._redo_stack:
            return
        try:
            current = self.snapshot()
            nxt = self._redo_stack.pop()
            self._undo_stack.append(current)
            self.apply_snapshot(nxt)
        except Exception:
            pass
    
    def gaussian_elimination(self, A: np.ndarray, b: np.ndarray = None) -> dict:
        """
        Perform Gaussian elimination on matrix A (with optional vector b).
        Returns solution and steps for visualization.
        """
        if b is not None:
            # Augmented matrix
            augmented = np.hstack([A, b.reshape(-1, 1)])
        else:
            augmented = A.copy()
        
        steps = []
        n = augmented.shape[0]
        
        for i in range(n):
            # Pivot selection
            max_row = i + np.argmax(np.abs(augmented[i:, i]))
            if max_row != i:
                augmented[[i, max_row]] = augmented[[max_row, i]]
                steps.append({'type': 'swap', 'rows': (i, max_row)})
            
            # Normalize pivot row
            pivot = augmented[i, i]
            if abs(pivot) > 1e-10:
                augmented[i] = augmented[i] / pivot
                steps.append({'type': 'scale', 'row': i, 'factor': 1/pivot})
            
            # Eliminate below
            for j in range(i + 1, n):
                factor = augmented[j, i]
                if abs(factor) > 1e-10:
                    augmented[j] = augmented[j] - factor * augmented[i]
                    steps.append({'type': 'eliminate', 'row': j, 'from': i, 'factor': factor})
        
        # Back substitution
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = augmented[i, -1] - np.dot(augmented[i, i+1:n], x[i+1:])
        
        return {
            'solution': x,
            'steps': steps,
            'augmented_matrix': augmented
        }
    
    def compute_null_space(self, A: np.ndarray) -> np.ndarray:
        """Compute the null space of matrix A."""
        # Using SVD to find null space
        U, S, Vt = np.linalg.svd(A)
        null_space = Vt[np.abs(S) < 1e-10]
        return null_space
    
    def compute_column_space(self, A: np.ndarray) -> np.ndarray:
        """Compute the column space of matrix A."""
        # Non-zero singular vectors
        U, S, Vt = np.linalg.svd(A)
        column_space = U[:, np.abs(S) > 1e-10]
        return column_space