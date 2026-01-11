"""
Inspector computed properties section.
"""

import imgui
import numpy as np

from core.vector import Vector3D


def _render_computed_properties(self, vector, scene):
    """Render computed properties relative to other vectors."""
    if imgui.collapsing_header("Computed Properties",
                              flags=imgui.TREE_NODE_DEFAULT_OPEN):

        if len(scene.vectors) > 1:
            imgui.text("Compare with:")
            imgui.same_line()

            other_vectors = [v for v in scene.vectors if v is not vector]
            if other_vectors:
                current_other = other_vectors[0]

                imgui.push_item_width(150)
                if imgui.begin_combo("##compare_vector", current_other.label):
                    for v in other_vectors:
                        if imgui.selectable(v.label, v is current_other)[0]:
                            current_other = v
                    imgui.end_combo()
                imgui.pop_item_width()

                imgui.spacing()

                dot = vector.dot(current_other)
                imgui.text(f"Dot product: {dot:.6f}")

                angle_deg = vector.angle(current_other, degrees=True)
                imgui.text(f"Angle: {angle_deg:.2f}°")

                imgui.spacing()
                if imgui.button("Compute Cross Product", width=-1):
                    result = vector.cross(current_other)
                    scene.add_vector(result)

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

        for axis_name, axis_vec in axes:
            projection = vector.project_onto(Vector3D(axis_vec))
            proj_mag = projection.magnitude()

            imgui.text(f"  {axis_name}-axis: {proj_mag:.6f}")
