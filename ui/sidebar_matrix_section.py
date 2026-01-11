"""
Sidebar matrix operations section.
"""

import imgui
import numpy as np


def _render_matrix_operations(self, scene):
    """Render matrix operations section."""
    if self._section("Matrix Operations", "📐"):
        imgui.text("Saved Matrices:")
        imgui.spacing()
        imgui.begin_child("##matrix_list", 0, 120, border=True)
        if not scene.matrices:
            imgui.text_disabled("No matrices saved")
        else:
            for i, mat in enumerate(scene.matrices):
                label = mat.get('label') or f"Matrix {i+1}"
                shape = mat.get('matrix').shape if 'matrix' in mat else None
                selectable_label = f"{label} ({shape[0]}x{shape[1]})##mat_{i}"
                is_selected = (self.selected_matrix_idx == i)
                if imgui.selectable(selectable_label, is_selected)[0]:
                    self.selected_matrix_idx = i
                    scene.selected_object = mat
                    scene.selection_type = 'matrix'
                    try:
                        m = mat['matrix']
                        self.matrix_size = m.shape[0] if m.ndim == 2 else self.matrix_size
                        self.matrix_input = [list(row) for row in np.array(m).tolist()]
                        self.matrix_name = mat.get('label', self.matrix_name)
                        self.show_matrix_editor = True
                    except Exception:
                        pass

                imgui.same_line()
                if imgui.small_button(f"Del##mat_del_{i}"):
                    scene.remove_matrix(mat)
                    if self.selected_matrix_idx == i:
                        self.selected_matrix_idx = None
        imgui.end_child()

        imgui.spacing()
        imgui.text(f"New Matrix Size: {self.matrix_size}x{self.matrix_size}")
        if imgui.button("Open Matrix Editor", width=-1):
            self.show_matrix_editor = not self.show_matrix_editor

        imgui.spacing()

        if self.show_matrix_editor:
            imgui.begin_child("##matrix_editor", 0, 200, border=True)

            imgui.text("Matrix Size:")
            imgui.same_line()
            imgui.push_item_width(100)
            size_changed, self.matrix_size = imgui.slider_int("##matrix_size",
                                                            self.matrix_size, 2, 4)
            imgui.pop_item_width()

            if size_changed:
                self._resize_matrix()

            imgui.spacing()
            changed, self.matrix_input = self._matrix_input_widget(self.matrix_input)

            imgui.spacing()
            imgui.text("Name:")
            imgui.same_line()
            imgui.push_item_width(100)
            name_changed, self.matrix_name = imgui.input_text("##matrix_name",
                                                            self.matrix_name, 16)
            imgui.pop_item_width()

            imgui.same_line()
            prev_changed, self.preview_matrix_enabled = imgui.checkbox("Preview", self.preview_matrix_enabled)
            if prev_changed:
                if self.preview_matrix_enabled:
                    try:
                        scene.set_preview_matrix(np.array(self.matrix_input, dtype=np.float32))
                    except Exception:
                        scene.set_preview_matrix(None)
                else:
                    scene.set_preview_matrix(None)

            imgui.spacing()
            imgui.columns(3, "##matrix_buttons", border=False)

            if self.selected_matrix_idx is None:
                if imgui.button("Add Matrix", width=-1):
                    self._resize_matrix()
                    self._add_matrix(scene)
            else:
                if imgui.button("Save Matrix", width=-1):
                    try:
                        matrix = np.array(self.matrix_input, dtype=np.float32)
                        scene.matrices[self.selected_matrix_idx]['matrix'] = matrix
                        scene.matrices[self.selected_matrix_idx]['label'] = self.matrix_name
                        self.operation_result = {'type': 'save_matrix', 'index': self.selected_matrix_idx}
                    except Exception as e:
                        self.operation_result = {'error': str(e)}

            imgui.next_column()

            if imgui.button("Apply to Selected", width=-1):
                self._apply_matrix_to_selected(scene)

            imgui.next_column()

            if imgui.button("Apply to All", width=-1):
                try:
                    matrix = np.array(self.matrix_input, dtype=np.float32)
                    scene.apply_transformation(matrix)
                    self.operation_result = {'type': 'apply_all'}
                except Exception as e:
                    self.operation_result = {'error': str(e)}

            imgui.next_column()
            if imgui.button("Null Space", width=-1):
                try:
                    mat = np.array(self.matrix_input, dtype=np.float32)
                    self._compute_null_space(scene, mat)
                except Exception as e:
                    self.operation_result = {'error': str(e)}

            imgui.next_column()
            if imgui.button("Column Space", width=-1):
                try:
                    mat = np.array(self.matrix_input, dtype=np.float32)
                    self._compute_column_space(scene, mat)
                except Exception as e:
                    self.operation_result = {'error': str(e)}

            imgui.columns(1)

            imgui.end_child()

        self._end_section()
