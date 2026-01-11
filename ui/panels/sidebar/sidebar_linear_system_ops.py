"""
Sidebar linear system helpers.
"""

import numpy as np
from state.actions import AddVector


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


def _solve_linear_system(self):
    """Solve linear system of equations."""
    try:
        n = self.equation_count
        A = np.zeros((n, n), dtype=np.float32)
        b = np.zeros(n, dtype=np.float32)

        for i in range(n):
            for j in range(n):
                A[i, j] = self.equation_input[i][j]
            b[i] = self.equation_input[i][-1]

        solution = np.linalg.solve(A, b)
        self.operation_result = {
            'solution': solution,
            'steps': [],
            'augmented_matrix': np.hstack([A, b.reshape(-1, 1)]),
        }

    except Exception as e:
        self.operation_result = {'error': str(e)}


def _add_solution_vectors(self, solution):
    """Add solution as vectors to scene."""
    if self._dispatch is None:
        return

    if len(solution) == 3:
        coords = np.array(solution, dtype=np.float32)
        self._dispatch(AddVector(
            coords=tuple(coords.tolist()),
            color=self._get_next_color(),
            label="solution",
        ))
        return

    for i, val in enumerate(solution):
        coords = np.zeros(3, dtype=np.float32)
        coords[i % 3] = val

        self._dispatch(AddVector(
            coords=tuple(coords.tolist()),
            color=self.color_palette[i % len(self.color_palette)],
            label=f"x{i+1}",
        ))
