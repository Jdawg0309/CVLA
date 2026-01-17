"""
Sidebar linear systems section.
"""

import imgui

from state.actions import SetEquationCell, SetEquationCount


def _render_linear_systems(self):
    """Render linear systems solver section."""
    if self._section("Linear Systems", "🧮"):
        if imgui.button("Open Equation Solver", width=-1):
            self.show_equation_editor = not self.show_equation_editor

        imgui.spacing()

        if self.show_equation_editor:
            imgui.begin_child("##equation_editor", 0, 300, border=True)

            equation_count = self._state.input_equation_count
            equation_input = self._state.input_equations

            imgui.text("Number of equations:")
            imgui.same_line()
            imgui.push_item_width(100)
            count_changed, new_count = imgui.slider_int(
                "##eq_count",
                equation_count,
                2,
                4,
            )
            imgui.pop_item_width()

            if count_changed and self._dispatch is not None:
                self._dispatch(SetEquationCount(count=new_count))

            imgui.spacing()
            imgui.text("System Ax = b:")

            for i in range(equation_count):
                imgui.push_id(str(i))

                imgui.text(f"Eq {i+1}:")
                imgui.same_line()

                for j in range(equation_count):
                    imgui.push_item_width(50)
                    coeff_changed, new_val = self._input_number_cell(
                        f"eq_{i}_{j}", equation_input[i][j]
                    )
                    if coeff_changed and self._dispatch is not None:
                        self._dispatch(SetEquationCell(row=i, col=j, value=new_val))
                    imgui.pop_item_width()

                    imgui.same_line()
                    if j < equation_count - 1:
                        imgui.text(f"x{j+1} +")
                    else:
                        imgui.text(f"x{j+1} =")

                    imgui.same_line()

                imgui.push_item_width(50)
                rhs_changed, new_rhs = self._input_number_cell(
                    f"eq_rhs_{i}", equation_input[i][-1]
                )
                if rhs_changed and self._dispatch is not None:
                    self._dispatch(SetEquationCell(row=i, col=equation_count, value=new_rhs))
                imgui.pop_item_width()

                imgui.pop_id()

            imgui.spacing()
            if imgui.button("Solve System", width=-1):
                self._solve_linear_system()

            if self.operation_result and 'solution' in self.operation_result:
                imgui.spacing()
                imgui.separator()
                imgui.spacing()

                imgui.text_colored("Solution:", 0.2, 0.8, 0.2)
                solution = self.operation_result['solution']

                for i, val in enumerate(solution):
                    imgui.text(f"x{i+1} = {val:.4f}")

                imgui.spacing()
                if imgui.button("Add Solution Vectors", width=-1):
                    self._add_solution_vectors(solution)

            imgui.end_child()

        self._end_section()
