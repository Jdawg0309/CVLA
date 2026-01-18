"""
Linear systems widget for Gaussian elimination.
"""

import imgui
import numpy as np
from typing import TYPE_CHECKING, Optional, List, Dict, Any

from state.actions import SetPipeline
from state.models import EducationalStep
from domain.vectors.vector_ops import gaussian_elimination_steps

if TYPE_CHECKING:
    from state.app_state import AppState
    from state.models.tensor_model import TensorData


class LinearSystemsWidget:
    """Widget for Gaussian elimination over Ax = b."""

    def __init__(self):
        self._steps: Optional[List[Dict[str, Any]]] = None
        self._status: Optional[str] = None
        self._solution: Optional[np.ndarray] = None

    def render(self, state: "AppState", dispatch, width: float, selected: Optional["TensorData"]):
        if state is None:
            return

        imgui.text("LINEAR SYSTEMS")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if selected is None or selected.rank != 2:
            imgui.text_colored(
                "Select a rank-2 tensor from the left panel.",
                0.6, 0.6, 0.6, 1.0
            )
            imgui.spacing()
            imgui.text_wrapped(
                "Gaussian elimination runs on the selected tensor. "
                "Use an augmented matrix [A | b] with shape n x (n+1)."
            )
            return

        rows = selected.rows
        cols = selected.cols
        if cols != rows + 1:
            imgui.text_colored(
                f"Selected tensor shape is {rows} x {cols}.",
                1.0, 0.6, 0.2, 1.0
            )
            imgui.text_wrapped(
                "Expected an augmented matrix with shape n x (n+1) "
                "to represent [A | b]."
            )
            return

        imgui.text_disabled(f"Using selected tensor: {selected.label} ({rows} x {cols})")
        imgui.spacing()
        if imgui.button("Gaussian Elimination", width - 20, 26):
            self._run_elimination(selected, dispatch)

        if self._status is not None:
            imgui.spacing()
            if self._status == "unique" and self._solution is not None:
                imgui.text_colored("Solution:", 0.2, 0.8, 0.2, 1.0)
                for i, val in enumerate(self._solution):
                    imgui.text(f"x{i + 1} = {val:.4g}")
            elif self._status == "infinite":
                imgui.text_colored("Infinite solutions (free variables).", 1.0, 0.7, 0.2, 1.0)
            elif self._status == "inconsistent":
                imgui.text_colored("No solution (inconsistent system).", 1.0, 0.4, 0.4, 1.0)

        if self._steps:
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            step_idx = min(state.pipeline_step_index, len(self._steps) - 1)
            step = self._steps[step_idx]

            imgui.text(f"Step {step_idx + 1} / {len(self._steps)}: {step['title']}")
            if step.get("description"):
                imgui.text_wrapped(step["description"])

            imgui.spacing()
            matrix = step.get("matrix")
            if matrix:
                self._render_matrix(matrix)

    def _run_elimination(self, tensor: "TensorData", dispatch):
        values = np.array(tensor.values, dtype=np.float32)
        n = values.shape[0]
        A = values[:, :n]
        b = values[:, n]

        steps, solution, status = gaussian_elimination_steps(A, b)
        self._steps = steps
        self._solution = solution
        self._status = status

        pipeline_steps = tuple(
            EducationalStep.create(
                title=step["title"],
                explanation=step["description"],
                operation="gaussian_elimination",
            )
            for step in steps
        )
        if dispatch:
            dispatch(SetPipeline(steps=pipeline_steps, index=0))

    def _render_matrix(self, matrix):
        rows = len(matrix)
        cols = len(matrix[0]) if rows else 0

        table_flags = 0
        try:
            table_flags = imgui.TABLE_BORDERS_INNER_H | imgui.TABLE_BORDERS_OUTER
        except Exception:
            pass

        if imgui.begin_table("##ls_matrix", cols, table_flags):
            for i in range(rows):
                imgui.table_next_row()
                for j in range(cols):
                    imgui.table_next_column()
                    imgui.text(f"{matrix[i][j]:.2f}")
            imgui.end_table()
