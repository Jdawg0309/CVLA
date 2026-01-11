"""
Sidebar utility helpers.
"""

import imgui


def _get_next_color(self):
    """Get next color from palette."""
    color = self.color_palette[self.next_color_idx]
    self.next_color_idx = (self.next_color_idx + 1) % len(self.color_palette)
    return color


def _styled_button(self, label, color=None, width=0):
    """Create a styled button."""
    if color:
        imgui.push_style_color(imgui.COLOR_BUTTON, *color)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED,
                             color[0]*1.2, color[1]*1.2, color[2]*1.2, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE,
                             color[0]*0.8, color[1]*0.8, color[2]*0.8, 1.0)

    result = imgui.button(label, width=width)

    if color:
        imgui.pop_style_color(3)

    return result


def _section(self, title, icon="", default_open=True):
    """Create a styled collapsible section."""
    flags = imgui.TREE_NODE_DEFAULT_OPEN if default_open else 0

    imgui.push_style_color(imgui.COLOR_HEADER, 0.15, 0.15, 0.18, 0.8)
    imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0.2, 0.2, 0.25, 0.9)
    imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, 0.25, 0.25, 0.3, 0.9)
    imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (0, 4))

    expanded = imgui.tree_node(f"  {icon}  {title}###{title}", flags)

    imgui.pop_style_var(1)
    imgui.pop_style_color(3)

    if expanded:
        imgui.indent(15)
        imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (4, 6))

    return expanded


def _end_section(self):
    """End a section."""
    try:
        imgui.tree_pop()
    except Exception:
        pass

    imgui.pop_style_var(1)
    imgui.unindent(15)
    imgui.spacing()
    imgui.separator()
    imgui.spacing()


def _input_float3(self, label, values, speed=0.1, format="%.2f"):
    """Custom float3 input with better styling."""
    imgui.push_item_width(-1)
    changed, new_values = imgui.input_float3(f"##{label}", *values, format=format)
    imgui.pop_item_width()
    if changed and new_values is not None:
        try:
            values = [float(new_values[0]), float(new_values[1]), float(new_values[2])]
        except Exception:
            pass

    imgui.same_line()
    imgui.text("  ")
    imgui.same_line()

    imgui.push_button_repeat(True)
    imgui.push_item_width(60)

    for i in range(3):
        imgui.same_line()
        label_char = ['X', 'Y', 'Z'][i]
        if imgui.arrow_button(f"##{label}_dec_{i}", imgui.DIRECTION_LEFT):
            values[i] -= speed
            changed = True

        imgui.same_line()
        imgui.text(f" {label_char}")
        imgui.same_line()

        if imgui.arrow_button(f"##{label}_inc_{i}", imgui.DIRECTION_RIGHT):
            values[i] += speed
            changed = True

        if i < 2:
            imgui.same_line()
            imgui.text("  ")
            imgui.same_line()

    imgui.pop_item_width()
    imgui.pop_button_repeat()

    return changed, values


def _coerce_float(self, text):
    """Parse numeric text safely, allowing partial input."""
    if text is None:
        return None
    stripped = text.strip()
    if stripped in ("", "-", "+", ".", "-.", "+."):
        return None
    try:
        return float(stripped)
    except Exception:
        return None


def _input_number_cell(self, key, value):
    """Numeric input cell with backspace-friendly editing."""
    buf = self._cell_buffers.get(key)
    if buf is None:
        buf = f"{value:.2f}"

    flags = 0
    try:
        flags = imgui.INPUT_TEXT_CHARS_DECIMAL
    except Exception:
        flags = 0

    changed, new_buf = imgui.input_text(f"##{key}", buf, 32, flags)
    if changed:
        self._cell_buffers[key] = new_buf
        parsed = self._coerce_float(new_buf)
        if parsed is not None:
            return True, parsed
    return False, value


def _matrix_input_widget(self, matrix, editable=True):
    """Widget for matrix input."""
    changed = False
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    imgui.push_style_var(imgui.STYLE_CELL_PADDING, (2, 2))

    table_flags = 0
    try:
        table_flags = imgui.TABLE_BORDERS_INNER_H | imgui.TABLE_BORDERS_OUTER
    except Exception:
        pass

    if imgui.begin_table(f"##matrix_table", cols + 1, table_flags):
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text("")

        for j in range(cols):
            imgui.table_next_column()
            imgui.text(f"Col {j+1}")

        for i in range(rows):
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(f"Row {i+1}")

            for j in range(cols):
                imgui.table_next_column()

                if editable:
                    cell_key = f"mat_{i}_{j}"
                    imgui.push_id(cell_key)
                    imgui.push_item_width(60)
                    cell_changed, new_val = self._input_number_cell(
                        cell_key, matrix[i][j]
                    )
                    imgui.pop_item_width()
                    imgui.pop_id()

                    if cell_changed:
                        matrix[i][j] = new_val
                        changed = True
                else:
                    imgui.text(f"{matrix[i][j]:.2f}")

        imgui.end_table()

    imgui.pop_style_var(1)
    return changed, matrix
