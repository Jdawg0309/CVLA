"""
Rank-1 tensor operations widget for the operations panel.

Provides UI for rank-1 operations.
"""

import imgui
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from state.app_state import AppState
    from state.models.tensor_model import TensorData

from state.actions.tensor_actions import ApplyOperation, PreviewOperation
from state.selectors import get_vectors


class VectorOpsWidget:
    """Widget for vector operations."""

    UNARY_OPS = [
        ("normalize", "Normalize", "Scale to unit length"),
        ("negate", "Negate", "Reverse direction"),
    ]

    BINARY_OPS = [
        ("add", "Add", "Add two vectors"),
        ("subtract", "Subtract", "Subtract vectors"),
        ("dot", "Dot Product", "Compute dot product"),
        ("cross", "Cross Product", "Compute cross product (3D)"),
        ("project", "Project", "Project onto another vector"),
    ]

    SCALAR_OPS = [
        ("scale", "Scale", "Multiply by scalar"),
    ]

    def __init__(self):
        self._scale_factor = 1.0
        self._selected_other_idx = 0

    def render(self, tensor: "TensorData", state: "AppState", dispatch, width: float):
        """Render vector operations UI."""
        if tensor is None or tensor.rank != 1:
            return

        imgui.text("RANK-1 OPERATIONS")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Unary operations
        imgui.text("Transform:")
        imgui.spacing()

        for op_id, op_name, op_desc in self.UNARY_OPS:
            if imgui.button(op_name, width - 20, 25):
                dispatch(ApplyOperation(
                    operation_name=op_id,
                    parameters=(),
                    target_ids=(tensor.id,),
                    create_new=True
                ))
            if imgui.is_item_hovered():
                imgui.set_tooltip(op_desc)
            imgui.spacing()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Scalar operations
        imgui.text("Scale:")
        imgui.spacing()

        imgui.push_item_width(width - 100)
        _, self._scale_factor = imgui.input_float(
            "Factor",
            self._scale_factor,
            0.1, 1.0,
            "%.2f"
        )
        imgui.pop_item_width()

        imgui.same_line()
        if imgui.button("Apply##scale", 60, 0):
            dispatch(ApplyOperation(
                operation_name="scale",
                parameters=(("factor", str(self._scale_factor)),),
                target_ids=(tensor.id,),
                create_new=True
            ))

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Binary operations (need another rank-1 tensor)
        imgui.text("With Another Rank-1 Tensor:")
        imgui.spacing()

        # Get other vectors
        other_vectors = [v for v in get_vectors(state) if v.id != tensor.id]

        if not other_vectors:
            imgui.text_colored("No other rank-1 tensors available", 0.5, 0.5, 0.5, 1.0)
        else:
            # Vector selector
            vector_names = [v.label for v in other_vectors]
            if self._selected_other_idx >= len(vector_names):
                self._selected_other_idx = 0

            imgui.push_item_width(width - 20)
            _, self._selected_other_idx = imgui.combo(
                "##other_vector",
                self._selected_other_idx,
                vector_names
            )
            imgui.pop_item_width()

            imgui.spacing()

            other = other_vectors[self._selected_other_idx]

            # Binary operation buttons
            button_width = (width - 30) / 2
            for i, (op_id, op_name, op_desc) in enumerate(self.BINARY_OPS):
                if i > 0 and i % 2 == 0:
                    pass  # New line
                elif i > 0:
                    imgui.same_line()

                # Special case: cross product only for 3D
                if op_id == "cross" and len(tensor.coords) != 3:
                    imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)

                if imgui.button(f"{op_name}##bin", button_width, 25):
                    if op_id != "cross" or len(tensor.coords) == 3:
                        dispatch(ApplyOperation(
                            operation_name=op_id,
                            parameters=(("other_id", other.id),),
                            target_ids=(tensor.id, other.id),
                            create_new=True
                        ))

                if op_id == "cross" and len(tensor.coords) != 3:
                    imgui.pop_style_var()

                if imgui.is_item_hovered():
                    imgui.set_tooltip(op_desc)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Quick actions
        imgui.text("Quick Actions:")
        imgui.spacing()

        half_width = (width - 30) / 2

        if imgui.button("Duplicate", half_width, 25):
            from state.actions.tensor_actions import DuplicateTensor
            dispatch(DuplicateTensor(id=tensor.id))

        imgui.same_line()

        if imgui.button("To Origin", half_width, 25):
            dispatch(ApplyOperation(
                operation_name="to_origin",
                parameters=(),
                target_ids=(tensor.id,),
                create_new=False
            ))
