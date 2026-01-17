"""Vectors tab - vector creation and operations."""

import imgui
from typing import Optional, Callable, Any

from ui.ribbon.ribbon_tab import RibbonTab
from ui.ribbon.ribbon_group import RibbonGroup
from ui.ribbon.ribbon_button import RibbonButton
from state.actions import (
    AddVector, DeleteVector, ClearAllVectors, DuplicateVector,
    SetInputVector, SelectVector, DeselectVector,
)


class VectorsTab(RibbonTab):
    """Vectors tab with creation, editing, and operations."""

    def __init__(self):
        groups = [
            RibbonGroup("Create", [
                RibbonButton("Add\nVector", "+V", tooltip="Add a new vector"),
                RibbonButton("Unit\nVector", "1", tooltip="Add a unit vector"),
                RibbonButton("From\nPoints", "P>V", tooltip="Create from two points", enabled=False),
            ]),
            RibbonGroup("Edit", [
                RibbonButton("Duplicate", "Dup", tooltip="Duplicate selected vector"),
                RibbonButton("Delete", "Del", tooltip="Delete selected vector"),
                RibbonButton("Clear\nAll", "Clr", tooltip="Clear all vectors"),
            ]),
            RibbonGroup("Operations", [
                RibbonButton("Add", "A+B", tooltip="Add two vectors", enabled=False),
                RibbonButton("Subtract", "A-B", tooltip="Subtract vectors", enabled=False),
                RibbonButton("Cross", "AxB", tooltip="Cross product", enabled=False),
                RibbonButton("Dot", "A.B", tooltip="Dot product", enabled=False),
            ]),
            RibbonGroup("Normalize", [
                RibbonButton("Normalize", "N", tooltip="Normalize to unit length", enabled=False),
                RibbonButton("Scale", "Sc", tooltip="Scale by factor", enabled=False),
            ]),
        ]
        super().__init__(groups)

        # Input state for quick vector creation
        self._quick_coords = [1.0, 0.0, 0.0]
        self._quick_color = [0.8, 0.2, 0.2]

    def render(
        self,
        state: Any,
        dispatch: Optional[Callable] = None,
        camera: Any = None,
        view_config: Any = None,
    ) -> None:
        """Render Vectors tab with creation panel and operations."""
        has_selection = state and state.selected_id and state.selected_type == "vector"
        has_vectors = state and len(state.vectors) > 0

        # Update button states
        if state:
            # Edit group
            edit_group = self._groups[1]
            edit_group.buttons[0].enabled = has_selection  # Duplicate
            edit_group.buttons[1].enabled = has_selection  # Delete
            edit_group.buttons[2].enabled = has_vectors    # Clear All

            # Normalize group
            norm_group = self._groups[3]
            norm_group.buttons[0].enabled = has_selection  # Normalize
            norm_group.buttons[1].enabled = has_selection  # Scale

        # Create group with special handling for Add Vector
        create_group = self._groups[0]

        imgui.begin_group()
        # Add Vector button with inline inputs
        if imgui.button("+V\nAdd\nVector", 64, 56):
            if dispatch and state:
                dispatch(AddVector(
                    coords=tuple(state.input_vector_coords),
                    color=state.input_vector_color,
                    label=state.input_vector_label or None
                ))

        if imgui.is_item_hovered():
            imgui.set_tooltip("Add a new vector with the specified coordinates")

        imgui.same_line()

        # Unit vector button
        if imgui.button("1\nUnit\nVector", 64, 56):
            if dispatch:
                dispatch(AddVector(coords=(1.0, 0.0, 0.0), color=(0.9, 0.3, 0.3), label="i"))
                dispatch(AddVector(coords=(0.0, 1.0, 0.0), color=(0.3, 0.9, 0.3), label="j"))
                dispatch(AddVector(coords=(0.0, 0.0, 1.0), color=(0.3, 0.3, 0.9), label="k"))

        if imgui.is_item_hovered():
            imgui.set_tooltip("Add unit basis vectors i, j, k")

        imgui.spacing()
        imgui.text_disabled("Create")
        imgui.end_group()

        imgui.same_line()
        imgui.dummy(16, 1)
        imgui.same_line()

        # Vector input panel
        imgui.begin_group()
        imgui.text("Coordinates:")

        if state:
            coords = list(state.input_vector_coords)
        else:
            coords = [1.0, 0.0, 0.0]

        imgui.push_item_width(50)
        changed_x, coords[0] = imgui.input_float("##vx", coords[0], 0, 0, "%.1f")
        imgui.same_line()
        changed_y, coords[1] = imgui.input_float("##vy", coords[1], 0, 0, "%.1f")
        imgui.same_line()
        changed_z, coords[2] = imgui.input_float("##vz", coords[2], 0, 0, "%.1f")
        imgui.pop_item_width()

        if (changed_x or changed_y or changed_z) and dispatch:
            dispatch(SetInputVector(coords=tuple(coords)))

        # Color picker
        if state:
            color = list(state.input_vector_color)
        else:
            color = [0.8, 0.2, 0.2]

        imgui.same_line()
        color_changed, color = imgui.color_edit3(
            "##vec_color",
            *color,
            imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL
        )
        if color_changed and dispatch:
            dispatch(SetInputVector(color=color))

        imgui.spacing()
        imgui.text_disabled("Vector Input")
        imgui.end_group()

        imgui.same_line()
        imgui.dummy(16, 1)
        imgui.same_line()

        # Edit group
        imgui.begin_group()

        if imgui.button("Dup\nDuplicate", 64, 56) and has_selection and dispatch:
            dispatch(DuplicateVector(id=state.selected_id))

        if imgui.is_item_hovered():
            imgui.set_tooltip("Duplicate the selected vector")

        imgui.same_line()

        if imgui.button("Del\nDelete", 64, 56) and has_selection and dispatch:
            dispatch(DeleteVector(id=state.selected_id))

        if imgui.is_item_hovered():
            imgui.set_tooltip("Delete the selected vector")

        imgui.same_line()

        if imgui.button("Clr\nClear\nAll", 64, 56) and has_vectors and dispatch:
            dispatch(ClearAllVectors())

        if imgui.is_item_hovered():
            imgui.set_tooltip("Clear all vectors")

        imgui.spacing()
        imgui.text_disabled("Edit")
        imgui.end_group()

        imgui.same_line()
        imgui.dummy(16, 1)
        imgui.same_line()

        # Vector list (scrollable)
        imgui.begin_group()
        imgui.text("Vectors:")

        imgui.begin_child("##vec_list", 180, 50, border=True)
        if state and state.vectors:
            for vec in state.vectors:
                is_selected = state.selected_id == vec.id
                label = vec.label or vec.id[:8]
                coords_str = f"({vec.coords[0]:.1f}, {vec.coords[1]:.1f}, {vec.coords[2]:.1f})"

                if imgui.selectable(f"{label}: {coords_str}", is_selected)[0]:
                    if dispatch:
                        if is_selected:
                            dispatch(DeselectVector())
                        else:
                            dispatch(SelectVector(id=vec.id))
        else:
            imgui.text_disabled("No vectors")
        imgui.end_child()

        imgui.spacing()
        imgui.text_disabled("Vector List")
        imgui.end_group()
