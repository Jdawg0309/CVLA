"""
Sidebar linear system helpers.
"""

import numpy as np
from core.vector import Vector3D


def _resize_equations(self):
    """Resize equation system."""
    new_count = self.equation_count

    new_equations = []
    for i in range(new_count):
        if i < len(self.equation_input):
            row = self.equation_input[i][:new_count + 1]
            if len(row) < new_count + 1:
                row.extend([0.0] * (new_count + 1 - len(row)))
            new_equations.append(row)
        else:
            row = [0.0] * (new_count + 1)
            row[i] = 1.0
            new_equations.append(row)

    self.equation_input = new_equations


def _solve_linear_system(self, scene):
    """Solve linear system of equations."""
    try:
        n = self.equation_count
        A = np.zeros((n, n), dtype=np.float32)
        b = np.zeros(n, dtype=np.float32)

        for i in range(n):
            for j in range(n):
                A[i, j] = self.equation_input[i][j]
            b[i] = self.equation_input[i][-1]

        result = scene.gaussian_elimination(A, b)
        self.operation_result = result

    except Exception as e:
        self.operation_result = {'error': str(e)}


def _add_solution_vectors(self, scene, solution):
    """Add solution as vectors to scene."""
    if len(solution) == 3:
        coords = np.array(solution, dtype=np.float32)
        v = Vector3D(coords, color=self._get_next_color(), label="solution")
        scene.add_vector(v)
        return

    for i, val in enumerate(solution):
        coords = np.zeros(3, dtype=np.float32)
        coords[i % 3] = val

        v = Vector3D(
            coords,
            color=self.color_palette[i % len(self.color_palette)],
            label=f"x{i+1}"
        )
        scene.add_vector(v)
