"""
Sidebar vector operations section.

This module handles vector algebra operations (add, subtract, cross, dot).
Reads vectors via selectors from tensors.
"""

import imgui
from state.actions import AddVector, UpdateVector
from state.selectors import get_vectors


def _render_vector_operations(self):
    """
    Render vector operations section.

    Uses state.vectors for reading, dispatches actions for results.
    """
    if self._section("Vector Operations", "⚡"):
        if self._state is None or self._dispatch is None:
            imgui.text_disabled("Vector operations unavailable (no state).")
            self._end_section()
            return

        vectors = list(get_vectors(self._state))
        selected_id = self._state.selected_tensor_id or self._state.selected_id
        selected_vector = None
        for v in vectors:
            if v.id == selected_id:
                selected_vector = v
                break

        imgui.columns(2, "##vec_ops_cols", border=False)

        # Normalize button
        if self._styled_button("Normalize", (0.3, 0.3, 0.6, 1.0), width=-1):
            if selected_vector:
                coords = selected_vector.coords
                norm_sq = sum(val * val for val in coords)
                if norm_sq > 1e-10:
                    norm = norm_sq ** 0.5
                    new_coords = tuple(val / norm for val in coords)
                    self._dispatch(UpdateVector(id=selected_vector.id, coords=new_coords))

        imgui.next_column()

        # Scale input
        if self._styled_button("Reset", (0.6, 0.4, 0.2, 1.0), width=-1):
            if selected_vector:
                self._dispatch(UpdateVector(id=selected_vector.id, coords=(1.0, 0.0, 0.0)))

        imgui.next_column()
        imgui.spacing()

        imgui.text("Scale by:")
        imgui.next_column()

        imgui.push_item_width(-1)
        scale_changed, self.scale_factor = imgui.input_float("##scale",
                                                           self.scale_factor, 0.1, 1.0, "%.2f")
        imgui.pop_item_width()

        imgui.next_column()

        if self._styled_button("Apply Scale", (0.3, 0.5, 0.3, 1.0), width=-1):
            if selected_vector:
                coords = selected_vector.coords
                new_coords = tuple(val * self.scale_factor for val in coords)
                self._dispatch(UpdateVector(id=selected_vector.id, coords=new_coords))

        imgui.columns(1)
        imgui.spacing()

        imgui.text("Vector Algebra:")
        imgui.spacing()

        if len(vectors) >= 2:
            imgui.columns(2, "##algebra_cols", border=False)

            # Use persistent indices stored on self
            if not hasattr(self, '_v1_idx'):
                self._v1_idx = 0
            if not hasattr(self, '_v2_idx'):
                self._v2_idx = min(1, len(vectors) - 1)

            # Clamp indices to valid range
            self._v1_idx = min(self._v1_idx, len(vectors) - 1)
            self._v2_idx = min(self._v2_idx, len(vectors) - 1)

            imgui.text("Vector 1:")
            imgui.next_column()
            imgui.text("Vector 2:")
            imgui.next_column()

            imgui.push_item_width(-1)
            if imgui.begin_combo("##v1_select", vectors[self._v1_idx].label):
                for i, v in enumerate(vectors):
                    if imgui.selectable(v.label, i == self._v1_idx)[0]:
                        self._v1_idx = i
                imgui.end_combo()
            imgui.pop_item_width()

            imgui.next_column()

            imgui.push_item_width(-1)
            if imgui.begin_combo("##v2_select", vectors[self._v2_idx].label):
                for i, v in enumerate(vectors):
                    if imgui.selectable(v.label, i == self._v2_idx)[0]:
                        self._v2_idx = i
                imgui.end_combo()
            imgui.pop_item_width()

            imgui.next_column()
            imgui.spacing()

            # Get the selected vectors
            v1 = vectors[self._v1_idx]
            v2 = vectors[self._v2_idx]

            # Operation buttons
            if self._styled_button("Add", (0.2, 0.5, 0.2, 1.0), width=-1):
                self._do_vector_algebra(v1, v2, "add")

            imgui.next_column()

            if self._styled_button("Subtract", (0.5, 0.2, 0.2, 1.0), width=-1):
                self._do_vector_algebra(v1, v2, "subtract")

            imgui.next_column()

            cross_enabled = len(v1.coords) == 3 and len(v2.coords) == 3
            if not cross_enabled:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
            if self._styled_button("Cross", (0.2, 0.2, 0.5, 1.0), width=-1):
                if cross_enabled:
                    self._do_vector_algebra(v1, v2, "cross")
            if not cross_enabled:
                imgui.pop_style_var()

            imgui.next_column()

            if self._styled_button("Dot", (0.5, 0.5, 0.2, 1.0), width=-1):
                self._do_vector_algebra(v1, v2, "dot")

            imgui.columns(1)

        self._end_section()


def _do_vector_algebra(self, v1, v2, operation):
    """
    Perform vector algebra and dispatch result.

    Args:
        v1, v2: VectorData objects
        operation: "add", "subtract", "cross", or "dot"
    """
    if len(v1.coords) != len(v2.coords):
        self.operation_result = {
            'error': 'Vector dimensions must match.'
        }
        return
    coords1 = v1.coords
    coords2 = v2.coords

    if operation == "add":
        result = tuple(a + b for a, b in zip(coords1, coords2))
        label = f"{v1.label}+{v2.label}"
    elif operation == "subtract":
        result = tuple(a - b for a, b in zip(coords1, coords2))
        label = f"{v1.label}-{v2.label}"
    elif operation == "cross":
        ax, ay, az = coords1
        bx, by, bz = coords2
        result = (
            (ay * bz) - (az * by),
            (az * bx) - (ax * bz),
            (ax * by) - (ay * bx),
        )
        label = f"{v1.label}x{v2.label}"
    elif operation == "dot":
        dot = float(sum(a * b for a, b in zip(coords1, coords2)))
        self.operation_result = {
            'type': 'dot_product',
            'value': dot,
            'vectors': [v1.label, v2.label]
        }
        return  # Dot product doesn't create a new vector

    # Dispatch new vector
    if self._dispatch:
        self._dispatch(AddVector(
            coords=tuple(result),
            color=self._get_next_color(),
            label=label
        ))
