"""
Inspector computed properties section.
"""

import imgui
from state.actions import AddVector
from state.selectors import get_vector_axis_projections, get_vector_dot_angle


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

                dot, angle_deg = get_vector_dot_angle(vector, current_other)
                imgui.text(f"Dot product: {dot:.6f}")
                imgui.text(f"Angle: {angle_deg:.2f}°")

                imgui.spacing()
                if imgui.button("Compute Cross Product", width=-1):
                    if dispatch:
                        ax, ay, az = vector.coords
                        bx, by, bz = current_other.coords
                        cross = (
                            (ay * bz) - (az * by),
                            (az * bx) - (ax * bz),
                            (ax * by) - (ay * bx),
                        )
                        dispatch(AddVector(
                            coords=cross,
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

        proj_x, proj_y, proj_z = get_vector_axis_projections(vector)
        imgui.text(f"  X-axis: {proj_x:.6f}")
        imgui.text(f"  Y-axis: {proj_y:.6f}")
        imgui.text(f"  Z-axis: {proj_z:.6f}")
