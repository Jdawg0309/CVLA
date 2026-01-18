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


def gaussian_elimination_steps(A: np.ndarray, b: np.ndarray, eps: float = 1e-9):
    """
    Perform Gaussian elimination with partial pivoting and record steps.

    Returns (steps, solution, status) where status is:
    - "unique": unique solution found
    - "infinite": infinite solutions (free variables)
    - "inconsistent": no solution
    """
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64).reshape(-1, 1)

    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix")
    if b.shape[0] != A.shape[0]:
        raise ValueError("b must have same number of rows as A")

    n, m = A.shape
    M = np.hstack([A, b])
    steps = []

    def snapshot():
        return tuple(tuple(float(x) for x in row) for row in M)

    def add_step(title: str, description: str):
        steps.append({
            "title": title,
            "description": description,
            "matrix": snapshot(),
        })

    add_step("Initial augmented matrix", "Start with [A | b].")

    pivot_row = 0
    for col in range(m):
        if pivot_row >= n:
            break

        # Partial pivot: choose row with largest absolute value in column.
        max_row = pivot_row + int(np.argmax(np.abs(M[pivot_row:, col])))
        if abs(M[max_row, col]) < eps:
            continue

        if max_row != pivot_row:
            M[[pivot_row, max_row]] = M[[max_row, pivot_row]]
            add_step(
                f"Swap R{pivot_row + 1} <-> R{max_row + 1}",
                "Swap rows to move a stronger pivot into position."
            )

        pivot_val = M[pivot_row, col]
        if abs(pivot_val - 1.0) > eps:
            M[pivot_row] = M[pivot_row] / pivot_val
            add_step(
                f"Normalize R{pivot_row + 1}",
                f"R{pivot_row + 1} = R{pivot_row + 1} / {pivot_val:.4g}"
            )

        for row in range(pivot_row + 1, n):
            factor = M[row, col]
            if abs(factor) > eps:
                M[row] = M[row] - factor * M[pivot_row]
                add_step(
                    f"Eliminate R{row + 1}",
                    f"R{row + 1} = R{row + 1} - ({factor:.4g}) * R{pivot_row + 1}"
                )

        pivot_row += 1

    # Check for inconsistent rows and rank.
    status = "unique"
    rank = 0
    for i in range(n):
        if np.any(np.abs(M[i, :m]) > eps):
            rank += 1
        else:
            if abs(M[i, m]) > eps:
                status = "inconsistent"
                add_step(
                    "Inconsistent row",
                    f"Row {i + 1} implies 0 = {M[i, m]:.4g}. No solution."
                )
                break

    if status != "inconsistent" and rank < m:
        status = "infinite"
        add_step(
            "Infinite solutions",
            "System has free variables; no unique solution."
        )

    solution = None
    if status == "unique":
        solution = np.zeros(m, dtype=np.float64)
        for i in range(n - 1, -1, -1):
            # Find first non-zero coefficient in row.
            lead_idx = None
            for j in range(m):
                if abs(M[i, j]) > eps:
                    lead_idx = j
                    break
            if lead_idx is None:
                continue

            rhs = M[i, m] - np.dot(M[i, lead_idx + 1:m], solution[lead_idx + 1:m])
            solution[lead_idx] = rhs / M[i, lead_idx]
            add_step(
                f"Back-substitute x{lead_idx + 1}",
                f"x{lead_idx + 1} = {solution[lead_idx]:.4g}"
            )

    return steps, solution, status


def qr_decomposition(A: np.ndarray):
    """QR decomposition of a matrix."""
    Q, R = np.linalg.qr(np.array(A, dtype=np.float32))
    return Q, R


def svd_decomposition(A: np.ndarray):
    """Singular Value Decomposition."""
    U, S, Vt = np.linalg.svd(np.array(A, dtype=np.float32), full_matrices=False)
    return U, S, Vt
