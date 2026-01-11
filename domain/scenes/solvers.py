"""
Scene linear algebra solvers.
"""

import numpy as np


class SceneSolversMixin:
    def gaussian_elimination(self, A: np.ndarray, b: np.ndarray = None) -> dict:
        """
        Perform Gaussian elimination on matrix A (with optional vector b).
        Returns solution and steps for visualization.
        """
        if b is not None:
            augmented = np.hstack([A, b.reshape(-1, 1)])
        else:
            augmented = A.copy()

        steps = []
        n = augmented.shape[0]

        for i in range(n):
            max_row = i + np.argmax(np.abs(augmented[i:, i]))
            if max_row != i:
                augmented[[i, max_row]] = augmented[[max_row, i]]
                steps.append({'type': 'swap', 'rows': (i, max_row)})

            pivot = augmented[i, i]
            if abs(pivot) > 1e-10:
                augmented[i] = augmented[i] / pivot
                steps.append({'type': 'scale', 'row': i, 'factor': 1 / pivot})

            for j in range(i + 1, n):
                factor = augmented[j, i]
                if abs(factor) > 1e-10:
                    augmented[j] = augmented[j] - factor * augmented[i]
                    steps.append({'type': 'eliminate', 'row': j, 'from': i, 'factor': factor})

        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = augmented[i, -1] - np.dot(augmented[i, i + 1:n], x[i + 1:])

        return {
            'solution': x,
            'steps': steps,
            'augmented_matrix': augmented
        }

    def compute_null_space(self, A: np.ndarray) -> np.ndarray:
        """Compute the null space of matrix A."""
        U, S, Vt = np.linalg.svd(A)
        null_space = Vt[np.abs(S) < 1e-10]
        return null_space

    def compute_column_space(self, A: np.ndarray) -> np.ndarray:
        """Compute the column space of matrix A."""
        U, S, Vt = np.linalg.svd(A)
        column_space = U[:, np.abs(S) > 1e-10]
        return column_space
