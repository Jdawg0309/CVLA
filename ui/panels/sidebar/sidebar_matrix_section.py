"""
Sidebar matrix operations section.

This module handles matrix operations UI.
Reads from AppState.matrices when available.
"""

import imgui
import numpy as np
from state.actions import (
    AddMatrix, DeleteMatrix, UpdateMatrix,
    ApplyMatrixToSelected, ApplyMatrixToAll,
    SetInputMatrixCell, SetInputMatrixSize, SetInputMatrixLabel, SelectMatrix,
    ToggleMatrixEditor, TogglePreview, ToggleMatrixPlot,
)


def _render_matrix_operations(self):
    """
    Render matrix operations section.

    Uses state.matrices for reading when available.
    """
    if self._section("Matrix Operations", "📐"):
        if self._state is None or self._dispatch is None:
            imgui.text_disabled("Matrix operations unavailable (no state).")
            self._end_section()
            return

        matrices = list(self._state.matrices)
        input_matrix = [list(row) for row in self._state.input_matrix]
        matrix_size = self._state.input_matrix_size
        matrix_name = self._state.input_matrix_label
        show_matrix_editor = self._state.show_matrix_editor
        preview_enabled = self._state.preview_enabled
        selected_matrix_id = None
        if self.selected_matrix_idx is not None and self.selected_matrix_idx < len(matrices):
            selected_matrix_id = matrices[self.selected_matrix_idx].id

        imgui.text("Saved Matrices:")
        imgui.spacing()
        imgui.begin_child("##matrix_list", 0, 120, border=True)
        if not matrices:
            imgui.text_disabled("No matrices saved")
        else:
            for i, mat in enumerate(matrices):
                label = mat.label or f"Matrix {i+1}"
                rows, cols = mat.shape
                is_selected = (self.selected_matrix_idx == i)

                selectable_label = f"{label} ({rows}x{cols})##mat_{i}"
                if imgui.selectable(selectable_label, is_selected)[0]:
                    self.selected_matrix_idx = i
                    self._dispatch(SelectMatrix(id=mat.id))
                    self._dispatch(SetInputMatrixSize(size=rows))
                    for r in range(rows):
                        for c in range(cols):
                            self._dispatch(SetInputMatrixCell(
                                row=r,
                                col=c,
                                value=float(mat.values[r][c]),
                            ))
                    self._dispatch(SetInputMatrixLabel(label=label))
                    if not self._state.show_matrix_editor:
                        self._dispatch(ToggleMatrixEditor())

                imgui.same_line()
                if imgui.small_button(f"Del##mat_del_{i}"):
                    self._dispatch(DeleteMatrix(id=mat.id))
                    if self.selected_matrix_idx == i:
                        self.selected_matrix_idx = None
        imgui.end_child()

        imgui.spacing()
        imgui.text(f"New Matrix Size: {matrix_size}x{matrix_size}")
        if imgui.button("Open Matrix Editor", width=-1):
            self._dispatch(ToggleMatrixEditor())

        imgui.spacing()

        if show_matrix_editor:
            imgui.begin_child("##matrix_editor", 0, 200, border=True)

            imgui.text("Matrix Size:")
            imgui.same_line()
            imgui.push_item_width(100)
            size_changed, new_size = imgui.slider_int(
                "##matrix_size",
                matrix_size,
                2,
                4
            )
            imgui.pop_item_width()

            if size_changed:
                self._dispatch(SetInputMatrixSize(size=new_size))
                matrix_size = new_size

            imgui.spacing()
            changed, new_matrix = self._matrix_input_widget(input_matrix)
            if changed:
                for r in range(len(new_matrix)):
                    for c in range(len(new_matrix[r])):
                        if new_matrix[r][c] != self._state.input_matrix[r][c]:
                            self._dispatch(SetInputMatrixCell(
                                row=r,
                                col=c,
                                value=float(new_matrix[r][c]),
                            ))

            imgui.spacing()
            imgui.text("Name:")
            imgui.same_line()
            imgui.push_item_width(100)
            name_changed, new_name = imgui.input_text(
                "##matrix_name",
                matrix_name,
                16
            )
            imgui.pop_item_width()
            if name_changed:
                self._dispatch(SetInputMatrixLabel(label=new_name))
                matrix_name = new_name

            imgui.same_line()
            prev_changed, preview_value = imgui.checkbox("Preview", preview_enabled)
            if prev_changed:
                self._dispatch(TogglePreview())

            imgui.same_line()
            plot_enabled = self._state.matrix_plot_enabled
            plot_changed, plot_enabled = imgui.checkbox("3D Matrix Plot", plot_enabled)
            if plot_changed:
                self._dispatch(ToggleMatrixPlot())

            imgui.spacing()
            imgui.columns(3, "##matrix_buttons", border=False)

            is_selected = selected_matrix_id is not None
            if not is_selected:
                if imgui.button("Add Matrix", width=-1):
                    matrix_tuple = tuple(tuple(row) for row in input_matrix)
                    self._dispatch(AddMatrix(values=matrix_tuple, label=matrix_name))
                    self.operation_result = {'type': 'add_matrix', 'label': matrix_name}
            else:
                if imgui.button("Save Matrix", width=-1):
                    try:
                        matrix_tuple = tuple(tuple(row) for row in input_matrix)
                        self._dispatch(UpdateMatrix(
                            id=selected_matrix_id,
                            values=matrix_tuple,
                            label=matrix_name,
                        ))
                        self.operation_result = {'type': 'save_matrix', 'id': selected_matrix_id}
                    except Exception as e:
                        self.operation_result = {'error': str(e)}

            imgui.next_column()

            if imgui.button("Apply to Selected", width=-1):
                if selected_matrix_id:
                    self._dispatch(ApplyMatrixToSelected(matrix_id=selected_matrix_id))

            imgui.next_column()

            if imgui.button("Apply to All", width=-1):
                try:
                    if selected_matrix_id:
                        self._dispatch(ApplyMatrixToAll(matrix_id=selected_matrix_id))
                        self.operation_result = {'type': 'apply_all'}
                except Exception as e:
                    self.operation_result = {'error': str(e)}

            imgui.next_column()
            if imgui.button("Null Space", width=-1):
                try:
                    mat = np.array(input_matrix, dtype=np.float32)
                    self._compute_null_space(mat)
                except Exception as e:
                    self.operation_result = {'error': str(e)}

            imgui.next_column()
            if imgui.button("Column Space", width=-1):
                try:
                    mat = np.array(input_matrix, dtype=np.float32)
                    self._compute_column_space(mat)
                except Exception as e:
                    self.operation_result = {'error': str(e)}

            imgui.columns(1)

            imgui.end_child()

        if self.operation_result:
            result = self.operation_result
            if "error" in result:
                imgui.text_colored(f"Error: {result['error']}", 0.9, 0.4, 0.4, 1.0)
            elif result.get("type") in ("null_space", "column_space"):
                title = "Null Space" if result["type"] == "null_space" else "Column Space"
                imgui.text(f"{title} Basis:")
                basis = result.get("basis") or []
                if basis:
                    imgui.begin_child("##basis_view", 0, 90, border=True)
                    for vec in basis:
                        row_str = ", ".join(f"{v:.3f}" for v in vec)
                        imgui.text(f"[{row_str}]")
                    imgui.end_child()
                else:
                    imgui.text_disabled("(No basis vectors)")
                added = result.get("vectors") or []
                if added:
                    imgui.text(f"Added to scene: {', '.join(added)}")

        self._end_section()
