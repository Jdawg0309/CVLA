"""
Inspector computed properties section.
"""

import imgui
import numpy as np
from state.actions import AddVector


def _render_computed_properties(self, vector, state, dispatch):
    """Render computed properties relative to other vectors."""
    if imgui.collapsing_header("Computed Properties",
                              flags=imgui.TREE_NODE_DEFAULT_OPEN):

        if len(state.vectors) > 1:
            imgui.text("Compare with:")
            imgui.same_line()

            other_vectors = [v for v in state.vectors if v.id != vector.id]
            if other_vectors:
                if not hasattr(self, "_compare_vector_id"):
                    self._compare_vector_id = other_vectors[0].id
                current_other = None
                for v in other_vectors:
                    if v.id == self._compare_vector_id:
                        current_other = v
                        break
                if current_other is None:
                    current_other = other_vectors[0]
                    self._compare_vector_id = current_other.id

                imgui.push_item_width(150)
                if imgui.begin_combo("##compare_vector", current_other.label):
                    for v in other_vectors:
                        if imgui.selectable(v.label, v.id == current_other.id)[0]:
                            current_other = v
                            self._compare_vector_id = v.id
                    imgui.end_combo()
                imgui.pop_item_width()

                imgui.spacing()

                vec_a = np.array(vector.coords, dtype=np.float32)
                vec_b = np.array(current_other.coords, dtype=np.float32)
                dot = float(np.dot(vec_a, vec_b))
                imgui.text(f"Dot product: {dot:.6f}")

                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)
                angle_deg = 0.0
                if norm_a > 1e-10 and norm_b > 1e-10:
                    cos_angle = np.clip(dot / (norm_a * norm_b), -1.0, 1.0)
                    angle_deg = float(np.degrees(np.arccos(cos_angle)))
                imgui.text(f"Angle: {angle_deg:.2f}°")

                imgui.spacing()
                if imgui.button("Compute Cross Product", width=-1):
                    if dispatch:
                        cross = np.cross(vec_a, vec_b)
                        dispatch(AddVector(
                            coords=tuple(cross.tolist()),
                            color=(0.2, 0.6, 0.9),
                            label=f"{vector.label}x{current_other.label}",
                        ))

            else:
                imgui.text_disabled("No other vectors to compare with")

        else:
            imgui.text_disabled("Add another vector for comparisons")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text("Projections onto axes:")

        axes = [
            ("X", np.array([1, 0, 0], dtype=np.float32)),
            ("Y", np.array([0, 1, 0], dtype=np.float32)),
            ("Z", np.array([0, 0, 1], dtype=np.float32))
        ]

        vec = np.array(vector.coords, dtype=np.float32)
        for axis_name, axis_vec in axes:
            axis_norm = np.linalg.norm(axis_vec)
            proj_mag = 0.0
            if axis_norm > 1e-10:
                proj_mag = float(np.dot(vec, axis_vec) / axis_norm)

            imgui.text(f"  {axis_name}-axis: {proj_mag:.6f}")
