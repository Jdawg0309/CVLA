"""
Enhanced vector operations with linear algebra functions
"""

import numpy as np
from typing import List, Tuple, Union, Optional


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector addition."""
    return np.array(a, dtype=np.float32) + np.array(b, dtype=np.float32)


def scale(v: np.ndarray, s: float) -> np.ndarray:
    """Vector scaling."""
    return np.array(v, dtype=np.float32) * float(s)


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product."""
    return float(np.dot(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)))


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product."""
    return np.cross(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32))


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.array(v, dtype=np.float32)
    return np.array(v, dtype=np.float32) / norm


def angle_between(a: np.ndarray, b: np.ndarray, degrees: bool = True) -> float:
    """Angle between two vectors."""
    dot_product = dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    
    cos_angle = dot_product / (norm_a * norm_b)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    if degrees:
        return float(np.degrees(angle))
    return float(angle)


def project(v: np.ndarray, onto: np.ndarray) -> np.ndarray:
    """Project vector v onto vector 'onto'."""
    onto_norm = np.linalg.norm(onto)
    if onto_norm < 1e-10:
        return np.zeros_like(v, dtype=np.float32)
    
    scalar = dot(v, onto) / (onto_norm * onto_norm)
    return onto * scalar


def reflect(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Reflect vector across a plane with given normal."""
    return v - 2 * dot(v, normal) * normal


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between vectors."""
    return a * (1 - t) + b * t


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two points."""
    return float(np.linalg.norm(np.array(b, dtype=np.float32) - np.array(a, dtype=np.float32)))


def gram_schmidt(vectors: List[np.ndarray]) -> List[np.ndarray]:
    """Perform Gram-Schmidt orthogonalization."""
    if not vectors:
        return []
    
    basis = []
    for v in vectors:
        w = np.array(v, dtype=np.float32)
        for b in basis:
            w = w - project(w, b)
        
        norm = np.linalg.norm(w)
        if norm > 1e-10:
            basis.append(w / norm)
    
    return basis


def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiplication."""
    return np.dot(np.array(A, dtype=np.float32), np.array(B, dtype=np.float32))


def matrix_inverse(A: np.ndarray) -> Optional[np.ndarray]:
    """Matrix inverse with error handling."""
    try:
        return np.linalg.inv(np.array(A, dtype=np.float32))
    except np.linalg.LinAlgError:
        return None


def matrix_determinant(A: np.ndarray) -> float:
    """Matrix determinant."""
    return float(np.linalg.det(np.array(A, dtype=np.float32)))


def eigen_decomposition(A: np.ndarray):
    """Eigen decomposition of a matrix."""
    try:
        eigenvalues, eigenvectors = np.linalg.eig(np.array(A, dtype=np.float32))
        return eigenvalues, eigenvectors
    except np.linalg.LinAlgError:
        return None, None


def solve_linear_system(A: np.ndarray, b: np.ndarray):
    """Solve linear system Ax = b."""
    try:
        x = np.linalg.solve(np.array(A, dtype=np.float32), np.array(b, dtype=np.float32))
        return x
    except np.linalg.LinAlgError:
        return None


def qr_decomposition(A: np.ndarray):
    """QR decomposition of a matrix."""
    Q, R = np.linalg.qr(np.array(A, dtype=np.float32))
    return Q, R


def svd_decomposition(A: np.ndarray):
    """Singular Value Decomposition."""
    U, S, Vt = np.linalg.svd(np.array(A, dtype=np.float32), full_matrices=False)
    return U, S, Vt