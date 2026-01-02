import numpy as np
from typing import Tuple, Optional, List


class Vector3D:
    def __init__(self, coords, color: Tuple[float, float, float] = (0.8, 0.2, 0.2), 
                 label: str = "", visible: bool = True, metadata: dict = None):
        self.coords = np.array(coords, dtype=np.float32).flatten()
        if self.coords.shape[0] != 3:
            raise ValueError(f"Vector must have 3 coordinates, got {self.coords.shape[0]}")
        
        self.color = color
        self.label = label
        self.visible = visible
        self.metadata = metadata or {}
        self.history = []  # Track transformations
        self.original_coords = self.coords.copy()
        
    def normalize(self) -> 'Vector3D':
        """Normalize this vector in-place."""
        norm = np.linalg.norm(self.coords)
        if norm > 1e-8:
            self.history.append(('normalize', norm))
            self.coords = self.coords / norm
        return self
    
    def scale(self, factor: float) -> 'Vector3D':
        """Scale this vector in-place."""
        self.history.append(('scale', factor))
        self.coords = self.coords * float(factor)
        return self
    
    def magnitude(self) -> float:
        """Return the magnitude of the vector."""
        return float(np.linalg.norm(self.coords))
    
    def dot(self, other: 'Vector3D') -> float:
        """Dot product with another vector."""
        return float(np.dot(self.coords, other.coords))
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        """Cross product with another vector."""
        result = np.cross(self.coords, other.coords)
        return Vector3D(result, color=self.color, label=f"{self.label}x{other.label}")
    
    def angle(self, other: 'Vector3D', degrees: bool = True) -> float:
        """Angle between this vector and another."""
        dot_product = self.dot(other)
        mag_product = self.magnitude() * other.magnitude()
        if mag_product < 1e-10:
            return 0.0
        
        cos_angle = dot_product / mag_product
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        if degrees:
            return float(np.degrees(angle))
        return float(angle)
    
    def project_onto(self, other: 'Vector3D') -> 'Vector3D':
        """Project this vector onto another vector."""
        scalar = self.dot(other) / other.dot(other)
        if abs(scalar) < 1e-10:
            return Vector3D([0, 0, 0], color=self.color, label=f"proj_{self.label}")
        
        projection = other.coords * scalar
        return Vector3D(projection, color=self.color, label=f"proj_{self.label}")
    
    def transform(self, matrix: np.ndarray) -> 'Vector3D':
        """Apply transformation matrix and return new vector."""
        result = np.dot(matrix, self.coords)
        return Vector3D(result, color=self.color, label=f"T({self.label})")
    
    def reset(self):
        """Reset vector to original coordinates."""
        self.coords = self.original_coords.copy()
        self.history.clear()
    
    def copy(self) -> 'Vector3D':
        """Create a copy of this vector."""
        return Vector3D(
            self.coords.copy(),
            self.color,
            f"{self.label}_copy",
            self.visible,
            self.metadata.copy()
        )
    
    def to_list(self) -> List[float]:
        """Convert coordinates to Python list."""
        return self.coords.tolist()
    
    def __str__(self) -> str:
        return f"Vector3D({self.label}: {self.coords})"
    
    def __repr__(self) -> str:
        return self.__str__()