"""
Inspector panel for detailed object inspection and editing.
"""

import imgui
from state.actions import AddVector, UpdateMatrix, UpdateMatrixCell, UpdateVector
from state.selectors import (
    get_selected_matrix,
    get_selected_vector,
    get_vector_axis_projections,
    get_vector_dot_angle,
    get_vector_magnitude,
    get_vectors,
)

from ui.utils import set_next_window_position, set_next_window_size

_SET_WINDOW_POS_FIRST = getattr(imgui, "SET_WINDOW_POS_FIRST_USE_EVER", 0)
_SET_WINDOW_SIZE_FIRST = getattr(imgui, "SET_WINDOW_SIZE_FIRST_USE_EVER", 0)
_WINDOW_RESIZABLE = getattr(imgui, "WINDOW_RESIZABLE", 1)
_WINDOW_NO_COLLAPSE = getattr(imgui, "WINDOW_NO_COLLAPSE", 1)

from ui.panels.images.images_tab import (
    _render_color_mode_selector,
    _render_image_info_section,
    _render_image_render_options,
    _render_image_tab_selector,
)


# Header section
def _render_header(self, vector, dispatch):
    """Render inspector header."""
    draw_list = imgui.get_window_draw_list()
    pos = imgui.get_cursor_screen_pos()
    draw_list.add_circle_filled(
        pos.x + 15, pos.y + 15,
        10, imgui.get_color_u32_rgba(*vector.color, 1.0)
    )

    imgui.dummy(30, 0)
    imgui.same_line()

    imgui.push_font()
    imgui.text_colored(vector.label, 0.9, 0.9, 1.0, 1.0)
    imgui.pop_font()

    imgui.same_line()
    imgui.text_disabled("(Vector)")

    imgui.text_disabled("3D Position Vector")

    imgui.same_line(200)
    changed, visible = imgui.checkbox("Visible", vector.visible)
    if changed and dispatch:
        dispatch(UpdateVector(id=vector.id, visible=visible))


# Coordinates section
def _render_coordinate_editor(self, vector, dispatch):
    """Render coordinate editor."""
    imgui.text_colored("Coordinates", 0.8, 0.8, 0.2, 1.0)
    imgui.spacing()

    coords = list(vector.coords)
    for idx, value in enumerate(coords):
        label = f"C{idx+1}" if idx > 2 else ["X", "Y", "Z"][idx]
        imgui.text(f"{label}: {value:.6f}")
        if idx % 3 != 2 and idx != len(coords) - 1:
            imgui.same_line(100 + (idx % 3) * 100)

    imgui.spacing()

    imgui.push_item_width(80)

    for idx, value in enumerate(coords):
        label = f"C{idx+1}" if idx > 2 else ["X", "Y", "Z"][idx]
        imgui.text(f"{label}:")
        imgui.same_line()
        changed, new_val = imgui.input_float(f"##edit_{idx}", value, format="%.4f")
        if changed and dispatch:
            coords[idx] = new_val
            dispatch(UpdateVector(id=vector.id, coords=tuple(coords)))
        if idx % 3 != 2 and idx != len(coords) - 1:
            imgui.same_line(120 + (idx % 3) * 120)

    imgui.pop_item_width()

    imgui.spacing()
    imgui.columns(2, "##dim_controls", border=False)
    if imgui.button("+ Component", width=-1) and dispatch:
        coords.append(0.0)
        dispatch(UpdateVector(id=vector.id, coords=tuple(coords)))
    imgui.next_column()
    if imgui.button("- Component", width=-1) and dispatch:
        if len(coords) > 1:
            coords = coords[:-1]
            dispatch(UpdateVector(id=vector.id, coords=tuple(coords)))
    imgui.columns(1)
    imgui.spacing()
    imgui.columns(3, "##quick_edit", border=False)

    quick_edits = [
        ("Zero", [0, 0, 0], (0.5, 0.5, 0.5, 1.0)),
        ("Unit X", [1, 0, 0], (1.0, 0.3, 0.3, 1.0)),
        ("Unit Y", [0, 1, 0], (0.3, 1.0, 0.3, 1.0)),
        ("Unit Z", [0, 0, 1], (0.3, 0.5, 1.0, 1.0)),
        ("Normalize", None, (0.3, 0.3, 0.8, 1.0)),
        ("Reset", None, (0.8, 0.5, 0.2, 1.0))
    ]

    for i, (label, coords, color) in enumerate(quick_edits):
        if i > 0 and i % 3 == 0:
            imgui.next_column()

        imgui.push_style_color(imgui.COLOR_BUTTON, *color)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED,
                             color[0]*1.2, color[1]*1.2, color[2]*1.2, 1.0)

        if imgui.button(label, width=-1):
            if not dispatch:
                continue

            if coords is not None:
                dispatch(UpdateVector(id=vector.id, coords=tuple(coords)))
            elif label == "Normalize":
                norm_sq = sum(val * val for val in vector.coords)
                if norm_sq > 1e-10:
                    norm = norm_sq ** 0.5
                    scaled = tuple(val / norm for val in vector.coords)
                    dispatch(UpdateVector(id=vector.id, coords=scaled))
            elif label == "Reset":
                size = max(1, len(vector.coords))
                reset = [0.0] * size
                reset[0] = 1.0
                dispatch(UpdateVector(id=vector.id, coords=tuple(reset)))

        imgui.pop_style_color(2)

        if i % 3 != 2:
            imgui.same_line()

    imgui.columns(1)


# Properties section
def _render_properties(self, vector, dispatch):
    """Render vector properties."""
    imgui.text_colored("Properties", 0.8, 0.8, 0.2, 1.0)
    imgui.spacing()

    magnitude = get_vector_magnitude(vector)
    imgui.text(f"Magnitude: {magnitude:.6f}")

    imgui.spacing()
    imgui.text("Color:")
    imgui.same_line()

    color_changed, new_color = imgui.color_edit3(
        "##vec_color_edit",
        *vector.color,
        imgui.COLOR_EDIT_NO_INPUTS
    )
    if color_changed and dispatch:
        dispatch(UpdateVector(id=vector.id, color=new_color))

    imgui.spacing()
    imgui.text("Label:")
    imgui.same_line()

    imgui.push_item_width(150)
    label_changed, new_label = imgui.input_text(
        "##vec_label_edit",
        vector.label,
        32
    )
    imgui.pop_item_width()

    if label_changed and dispatch:
        dispatch(UpdateVector(id=vector.id, label=new_label))

    if hasattr(vector, "metadata") and vector.metadata:
        imgui.spacing()
        imgui.text("Metadata:")
        for key, value in vector.metadata.items():
            imgui.text_disabled(f"  {key}: {value}")


# Computed section
def _render_computed_properties(self, vector, state, dispatch):
    """Render computed properties relative to other vectors."""
    if imgui.collapsing_header("Computed Properties",
                              flags=imgui.TREE_NODE_DEFAULT_OPEN):

        all_vectors = list(get_vectors(state))
        if len(all_vectors) > 1:
            imgui.text("Compare with:")
            imgui.same_line()

            other_vectors = [v for v in all_vectors if v.id != vector.id]
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
                cross_enabled = len(vector.coords) == 3 and len(current_other.coords) == 3
                if not cross_enabled:
                    imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                if imgui.button("Compute Cross Product", width=-1):
                    if dispatch and cross_enabled:
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
                if not cross_enabled:
                    imgui.pop_style_var()

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


class Inspector:
    def __init__(self):
        self.show_computed_properties = True

    def render(self, state, dispatch, rect):
        """Render inspector panel."""
        if state is None or dispatch is None:
            return

        selected_vector = get_selected_vector(state)
        selected_matrix = get_selected_matrix(state)

        x, y, width, height = rect
        set_next_window_position(x, y, cond=_SET_WINDOW_POS_FIRST)
        set_next_window_size((width, height), cond=_SET_WINDOW_SIZE_FIRST)
        imgui.set_next_window_size_constraints(
            (260, 240),
            (width + 80, height + 140),
        )

        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 2.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (10, 8))

        flags = _WINDOW_RESIZABLE | _WINDOW_NO_COLLAPSE
        if imgui.begin("Inspector", flags=flags):

            imgui.text("Inspector")
            imgui.same_line()
            if selected_vector:
                imgui.text_disabled(f"({selected_vector.label})")
            elif selected_matrix:
                label = selected_matrix.label or "Matrix"
                imgui.text_disabled(f"({label})")
            else:
                imgui.text_disabled("(No selection)")
            imgui.separator()

            if not selected_vector and not selected_matrix:
                if state.active_mode == "images" and state.current_image is not None:
                    self._render_image_inspector(state, dispatch)
                else:
                    imgui.text_wrapped("Select a vector or matrix to see detailed properties.")
            elif selected_vector:
                if imgui.collapsing_header("Coordinates", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    self._render_coordinate_editor(selected_vector, dispatch)
                if imgui.collapsing_header("Properties", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    self._render_properties(selected_vector, dispatch)
                if self.show_computed_properties and imgui.collapsing_header("Computed", 0)[0]:
                    self._render_computed_properties(selected_vector, state, dispatch)
            else:
                self._render_matrix_details(selected_matrix, state, dispatch)


        imgui.end()
        imgui.pop_style_var(2)

    _render_header = _render_header
    _render_coordinate_editor = _render_coordinate_editor
    _render_properties = _render_properties
    _render_computed_properties = _render_computed_properties

    def _render_matrix_details(self, selected_matrix, state, dispatch):
        rows, cols = selected_matrix.shape
        imgui.text(f"Size: {rows}x{cols}")
        imgui.spacing()
        imgui.text("Label:")
        imgui.same_line()
        imgui.push_item_width(160)
        label_changed, new_label = imgui.input_text(
            "##matrix_label_edit",
            selected_matrix.label or "",
            16
        )
        imgui.pop_item_width()
        if label_changed and dispatch:
            dispatch(UpdateMatrix(id=selected_matrix.id, label=new_label))

        imgui.spacing()
        imgui.text("Values:")
        imgui.spacing()
        imgui.push_item_width(70)
        for r, row in enumerate(selected_matrix.values):
            imgui.push_id(f"row_{r}")
            for c, value in enumerate(row):
                imgui.push_id(f"cell_{c}")
                changed, new_value = imgui.input_float(
                    "##val",
                    float(value),
                    format="%.3f"
                )
                if changed and dispatch:
                    dispatch(UpdateMatrixCell(
                        id=selected_matrix.id,
                        row=r,
                        col=c,
                        value=float(new_value),
                    ))
                imgui.pop_id()
                if c < cols - 1:
                    imgui.same_line()
            imgui.pop_id()
        imgui.pop_item_width()
        imgui.spacing()
        self._render_matrix_vectors(selected_matrix, state, dispatch)

    def _render_matrix_vectors(self, selected_matrix, state, dispatch):
        rows, cols = selected_matrix.shape
        if rows == 0 or cols == 0:
            return
        imgui.text("Column Vectors:")
        for c in range(cols):
            col = [selected_matrix.values[r][c] for r in range(rows)]
            display = ", ".join(f"{v:.2f}" for v in col[:4])
            suffix = "..." if len(col) > 4 else ""
            imgui.text(f"c{c+1} = [{display}{suffix}]")
        imgui.spacing()
        if imgui.button("Add Column Vectors", width=-1) and dispatch:
            base_label = selected_matrix.label or "M"
            for c in range(cols):
                col = tuple(selected_matrix.values[r][c] for r in range(rows))
                dispatch(AddVector(
                    coords=col,
                    color=(0.8, 0.2, 0.2),
                    label=f"{base_label}_c{c+1}",
                ))
        if state and state.input_vector_coords:
            v = state.input_vector_coords
            if len(v) == cols:
                result = []
                for r in range(rows):
                    value = sum(selected_matrix.values[r][c] * v[c] for c in range(cols))
                    result.append(value)
                preview = ", ".join(f"{val:.2f}" for val in result[:4])
                suffix = "..." if len(result) > 4 else ""
                imgui.text(f"M·v = [{preview}{suffix}]")

    def _render_image_inspector(self, state, dispatch):
        imgui.text("Image Inspector")
        imgui.separator()
        _render_image_tab_selector(state, dispatch)

        if state.active_image_tab == "raw":
            _render_color_mode_selector(state, dispatch)
            _render_image_render_options(state, dispatch)
            _render_image_info_section(state, dispatch)
        else:
            imgui.text_disabled("Preprocess view active")
