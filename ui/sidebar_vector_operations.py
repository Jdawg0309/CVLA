"""
Sidebar vector operations section.
"""

import imgui


def _render_vector_operations(self, scene, selected):
    """Render vector operations section."""
    if self._section("Vector Operations", "⚡"):
        imgui.columns(2, "##vec_ops_cols", border=False)

        if self._styled_button("Normalize", (0.3, 0.3, 0.6, 1.0), width=-1):
            if selected:
                selected.normalize()

        imgui.next_column()

        if self._styled_button("Reset", (0.6, 0.4, 0.2, 1.0), width=-1):
            if selected:
                selected.reset()

        imgui.next_column()
        imgui.spacing()

        imgui.text("Scale by:")
        imgui.next_column()

        imgui.push_item_width(-1)
        scale_changed, self.scale_factor = imgui.input_float("##scale",
                                                           1.0, 0.1, 1.0, "%.2f")
        imgui.pop_item_width()

        imgui.next_column()

        if self._styled_button("Apply Scale", (0.3, 0.5, 0.3, 1.0), width=-1):
            if selected:
                selected.scale(self.scale_factor)

        imgui.columns(1)
        imgui.spacing()

        imgui.text("Vector Algebra:")
        imgui.spacing()

        if len(scene.vectors) >= 2:
            imgui.columns(2, "##algebra_cols", border=False)

            v1_idx = 0
            v2_idx = min(1, len(scene.vectors) - 1)

            imgui.text("Vector 1:")
            imgui.next_column()
            imgui.text("Vector 2:")
            imgui.next_column()

            imgui.push_item_width(-1)
            if imgui.begin_combo("##v1_select", scene.vectors[v1_idx].label):
                for i, v in enumerate(scene.vectors):
                    if imgui.selectable(v.label, i == v1_idx)[0]:
                        v1_idx = i
                imgui.end_combo()
            imgui.pop_item_width()

            imgui.next_column()

            imgui.push_item_width(-1)
            if imgui.begin_combo("##v2_select", scene.vectors[v2_idx].label):
                for i, v in enumerate(scene.vectors):
                    if imgui.selectable(v.label, i == v2_idx)[0]:
                        v2_idx = i
                imgui.end_combo()
            imgui.pop_item_width()

            imgui.next_column()
            imgui.spacing()

            op_buttons = [
                ("Add", lambda: self._add_vectors(scene, v1_idx, v2_idx), (0.2, 0.5, 0.2, 1.0)),
                ("Subtract", lambda: self._subtract_vectors(scene, v1_idx, v2_idx), (0.5, 0.2, 0.2, 1.0)),
                ("Cross Product", lambda: self._cross_vectors(scene, v1_idx, v2_idx), (0.2, 0.2, 0.5, 1.0)),
                ("Dot Product", lambda: self._dot_vectors(scene, v1_idx, v2_idx), (0.5, 0.5, 0.2, 1.0)),
            ]

            for i, (label, func, color) in enumerate(op_buttons):
                if i % 2 == 0 and i > 0:
                    imgui.next_column()

                if self._styled_button(label, color, width=-1):
                    func()

                if i % 2 == 0:
                    imgui.next_column()

            imgui.columns(1)

        self._end_section()
