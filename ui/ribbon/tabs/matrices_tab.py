"""Matrices tab - matrix creation and operations."""

import imgui
from typing import Optional, Callable, Any

from ui.ribbon.ribbon_tab import RibbonTab
from ui.ribbon.ribbon_group import RibbonGroup
from ui.ribbon.ribbon_button import RibbonButton
from state.actions import (
    AddMatrix, DeleteMatrix, UpdateMatrix, SelectMatrix,
    ApplyMatrixToSelected, ApplyMatrixToAll,
    SetInputMatrixCell, SetInputMatrixSize, SetInputMatrixLabel,
    ToggleMatrixEditor, TogglePreview,
)


class MatricesTab(RibbonTab):
    """Matrices tab with creation, standard transforms, and operations."""

    def __init__(self):
        groups = [
            RibbonGroup("Create", [
                RibbonButton("Add\nMatrix", "+M", tooltip="Create a new matrix"),
                RibbonButton("Identity", "I", tooltip="Add identity matrix"),
                RibbonButton("Zero", "0", tooltip="Add zero matrix"),
            ]),
            RibbonGroup("Standard", [
                RibbonButton(
                    "Rotation", "Rot",
                    tooltip="Rotation matrices",
                    dropdown_items=[
                        ("Rotate X (90)", None),
                        ("Rotate Y (90)", None),
                        ("Rotate Z (90)", None),
                        ("-", None),
                        ("Custom Rotation...", None),
                    ]
                ),
                RibbonButton(
                    "Scale", "Sc",
                    tooltip="Scale matrices",
                    dropdown_items=[
                        ("Scale 2x", None),
                        ("Scale 0.5x", None),
                        ("-", None),
                        ("Custom Scale...", None),
                    ]
                ),
                RibbonButton("Shear", "Sh", tooltip="Shear matrix"),
                RibbonButton(
                    "Reflect", "Ref",
                    tooltip="Reflection matrices",
                    dropdown_items=[
                        ("Reflect XY", None),
                        ("Reflect XZ", None),
                        ("Reflect YZ", None),
                    ]
                ),
            ]),
            RibbonGroup("Apply", [
                RibbonButton("To\nSelected", "->1", tooltip="Apply to selected vector"),
                RibbonButton("To\nAll", "->*", tooltip="Apply to all vectors"),
                RibbonButton("Preview", "Eye", tooltip="Toggle preview mode", is_toggle=True),
            ]),
            RibbonGroup("Analysis", [
                RibbonButton("Null\nSpace", "Null", tooltip="Compute null space"),
                RibbonButton("Column\nSpace", "Col", tooltip="Compute column space"),
                RibbonButton("Eigen", "Eig", tooltip="Eigenvalues/vectors", enabled=False),
            ]),
        ]
        super().__init__(groups)

        self._selected_matrix_idx = None
        self._show_editor = False

    def render(
        self,
        state: Any,
        dispatch: Optional[Callable] = None,
        camera: Any = None,
        view_config: Any = None,
    ) -> None:
        """Render Matrices tab with matrix editor and operations."""
        has_matrices = state and len(state.matrices) > 0
        has_vector_selection = state and state.selected_id and state.selected_type == "vector"

        # Update button states
        apply_group = self._groups[2]
        apply_group.buttons[0].enabled = has_matrices and has_vector_selection  # To Selected
        apply_group.buttons[1].enabled = has_matrices  # To All
        apply_group.buttons[2].is_active = state and state.preview_enabled  # Preview

        # Matrix editor section
        imgui.begin_group()

        # Matrix size selector
        matrix_size = state.input_matrix_size if state else 3
        imgui.text("Size:")
        imgui.same_line()
        imgui.push_item_width(60)
        changed, new_size = imgui.slider_int("##msize", matrix_size, 2, 4)
        imgui.pop_item_width()
        if changed and dispatch:
            dispatch(SetInputMatrixSize(size=new_size))
            matrix_size = new_size

        # Matrix input grid
        if state:
            input_matrix = [list(row) for row in state.input_matrix]
        else:
            input_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        imgui.push_item_width(40)
        for r in range(matrix_size):
            for c in range(matrix_size):
                if r < len(input_matrix) and c < len(input_matrix[r]):
                    val = input_matrix[r][c]
                else:
                    val = 1.0 if r == c else 0.0

                changed, new_val = imgui.input_float(f"##m{r}{c}", val, 0, 0, "%.2f")
                if changed and dispatch:
                    dispatch(SetInputMatrixCell(row=r, col=c, value=new_val))

                if c < matrix_size - 1:
                    imgui.same_line()
        imgui.pop_item_width()

        imgui.spacing()
        imgui.text_disabled("Matrix Editor")
        imgui.end_group()

        imgui.same_line()
        imgui.dummy(16, 1)
        imgui.same_line()

        # Create group
        imgui.begin_group()

        if imgui.button("+M\nAdd\nMatrix", 64, 56):
            if dispatch and state:
                matrix_tuple = tuple(tuple(row[:matrix_size]) for row in input_matrix[:matrix_size])
                dispatch(AddMatrix(
                    values=matrix_tuple,
                    label=state.input_matrix_label or "M"
                ))

        if imgui.is_item_hovered():
            imgui.set_tooltip("Add the current matrix to the scene")

        imgui.same_line()

        if imgui.button("I\nIdentity", 64, 56):
            if dispatch:
                identity = tuple(tuple(1.0 if i == j else 0.0 for j in range(3)) for i in range(3))
                dispatch(AddMatrix(values=identity, label="I"))

        if imgui.is_item_hovered():
            imgui.set_tooltip("Add a 3x3 identity matrix")

        imgui.same_line()

        if imgui.button("0\nZero", 64, 56):
            if dispatch:
                zero = tuple(tuple(0.0 for _ in range(3)) for _ in range(3))
                dispatch(AddMatrix(values=zero, label="O"))

        if imgui.is_item_hovered():
            imgui.set_tooltip("Add a 3x3 zero matrix")

        imgui.spacing()
        imgui.text_disabled("Create")
        imgui.end_group()

        imgui.same_line()
        imgui.dummy(16, 1)
        imgui.same_line()

        # Saved matrices list
        imgui.begin_group()
        imgui.text("Saved:")

        imgui.begin_child("##mat_list", 140, 50, border=True)
        if state and state.matrices:
            for i, mat in enumerate(state.matrices):
                label = mat.label or f"M{i+1}"
                is_selected = self._selected_matrix_idx == i

                if imgui.selectable(f"{label} ({mat.shape[0]}x{mat.shape[1]})", is_selected)[0]:
                    self._selected_matrix_idx = i
                    # Load matrix into editor
                    if dispatch:
                        dispatch(SetInputMatrixSize(size=mat.shape[0]))
                        for r in range(mat.shape[0]):
                            for c in range(mat.shape[1]):
                                dispatch(SetInputMatrixCell(row=r, col=c, value=float(mat.values[r][c])))
                        dispatch(SetInputMatrixLabel(label=label))
        else:
            imgui.text_disabled("No matrices")
        imgui.end_child()

        imgui.spacing()
        imgui.text_disabled("Matrix List")
        imgui.end_group()

        imgui.same_line()
        imgui.dummy(16, 1)
        imgui.same_line()

        # Apply group
        imgui.begin_group()

        apply_enabled = has_matrices and self._selected_matrix_idx is not None
        selected_matrix_id = None
        if state and state.matrices and self._selected_matrix_idx is not None:
            if self._selected_matrix_idx < len(state.matrices):
                selected_matrix_id = state.matrices[self._selected_matrix_idx].id

        if imgui.button("->1\nTo\nSelected", 64, 56) and apply_enabled and has_vector_selection and dispatch:
            dispatch(ApplyMatrixToSelected(matrix_id=selected_matrix_id))

        if imgui.is_item_hovered():
            imgui.set_tooltip("Apply matrix to selected vector")

        imgui.same_line()

        if imgui.button("->*\nTo\nAll", 64, 56) and apply_enabled and dispatch:
            dispatch(ApplyMatrixToAll(matrix_id=selected_matrix_id))

        if imgui.is_item_hovered():
            imgui.set_tooltip("Apply matrix to all vectors")

        imgui.same_line()

        preview_active = state and state.preview_enabled
        if preview_active:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.5, 0.7, 1.0)

        if imgui.button("Eye\nPreview", 64, 56) and dispatch:
            dispatch(TogglePreview())

        if preview_active:
            imgui.pop_style_color()

        if imgui.is_item_hovered():
            imgui.set_tooltip("Toggle transformation preview")

        imgui.spacing()
        imgui.text_disabled("Apply")
        imgui.end_group()
