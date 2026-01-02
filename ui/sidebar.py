"""
Enhanced Sidebar with modern UI for linear algebra operations
"""

import imgui
import numpy as np
from core.vector import Vector3D
import colorsys
import json
from typing import List, Tuple, Optional


class Sidebar:
    def __init__(self):
        # Vector creation
        self.vec_input = [1.0, 0.0, 0.0]
        self.vec_name = ""
        self.vec_color = (0.8, 0.2, 0.2)
        self.next_vector_id = 4
        
        # Matrix input
        self.matrix_input = [[1.0, 0.0, 0.0], 
                            [0.0, 1.0, 0.0], 
                            [0.0, 0.0, 1.0]]
        self.matrix_name = "A"
        self.matrix_size = 3
        
        # System of equations
        self.equation_count = 3
        self.equation_input = [
            [1.0, 1.0, 1.0, 0.0],  # x + y + z = 0
            [2.0, -1.0, 0.0, 0.0], # 2x - y = 0
            [0.0, 1.0, -1.0, 0.0]  # y - z = 0
        ]
        
        # Operation state
        self.current_operation = None
        self.operation_result = None
        self.show_matrix_editor = False
        self.show_equation_editor = False
        self.show_export_dialog = False
        
        # UI state
        self.window_width = 420
        self.active_tab = "vectors"
        self.vector_list_filter = ""
        
        # Color palette for auto-colors
        self.color_palette = [
            (0.8, 0.2, 0.2),  # Red
            (0.2, 0.8, 0.2),  # Green
            (0.2, 0.2, 0.8),  # Blue
            (0.8, 0.8, 0.2),  # Yellow
            (0.8, 0.2, 0.8),  # Magenta
            (0.2, 0.8, 0.8),  # Cyan
            (0.8, 0.5, 0.2),  # Orange
            (0.5, 0.2, 0.8),  # Purple
        ]
        self.next_color_idx = 0
        # Preview toggle for matrix editor
        self.preview_matrix_enabled = False
        # Selected matrix index in the scene (for list operations)
        self.selected_matrix_idx = None
        # Runtime references and defaults
        self.scene = None
        self.scale_factor = 1.0

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
        
        # Section header
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
        # close the tree node opened in _section
        try:
            imgui.tree_pop()
        except Exception:
            # Some imgui wrappers may not require explicit tree_pop or
            # will raise if mismatched; callers ensure this is only invoked
            # when a section was expanded.
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
        # If user edited the float3 via the input, new_values contains the new tuple
        if changed and new_values is not None:
            try:
                # Ensure we return a mutable list of floats
                values = [float(new_values[0]), float(new_values[1]), float(new_values[2])]
            except Exception:
                # Fallback: ignore if unexpected
                pass

        # Add drag controls for fine adjustment (always available)
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

    def _matrix_input_widget(self, matrix, editable=True):
        """Widget for matrix input."""
        changed = False
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0
        
        imgui.push_style_var(imgui.STYLE_CELL_PADDING, (2, 2))

        # Some imgui bindings may not expose table flag constants; fall back to 0
        table_flags = 0
        try:
            table_flags = imgui.TABLE_BORDERS_INNER_H | imgui.TABLE_BORDERS_OUTER
        except Exception:
            pass

        if imgui.begin_table(f"##matrix_table", cols + 1, table_flags):
            # Header row
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("")
            
            for j in range(cols):
                imgui.table_next_column()
                imgui.text(f"Col {j+1}")
            
            # Data rows
            for i in range(rows):
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(f"Row {i+1}")
                
                for j in range(cols):
                    imgui.table_next_column()
                    
                    if editable:
                        # Use a per-cell ID to avoid collisions and focus issues
                        imgui.push_id(f"mat_{i}_{j}")
                        imgui.push_item_width(60)
                        # Avoid forcing auto-select-all behavior which prevents
                        # typical in-place editing (delete/backspace). Let the
                        # user edit freely and commit numeric changes normally.
                        cell_changed, new_val = imgui.input_float(
                            f"##mat_{i}_{j}", matrix[i][j], format="%.2f"
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

    def _render_vector_creation(self, scene):
        """Render vector creation section."""
        if self._section("Create Vector", "➕"):
            # Coordinates
            imgui.text("Coordinates:")
            changed, self.vec_input = self._input_float3("vec_coords", self.vec_input)
            
            imgui.spacing()
            
            # Name
            imgui.text("Label:")
            imgui.same_line()
            imgui.push_item_width(150)
            name_changed, self.vec_name = imgui.input_text("##vec_name", self.vec_name, 32)
            imgui.pop_item_width()
            
            # Auto-generate name if empty
            if not self.vec_name:
                imgui.same_line()
                imgui.text_disabled("(Auto: v{})".format(self.next_vector_id))
            
            imgui.spacing()
            
            # Color
            imgui.text("Color:")
            imgui.same_line()
            color_changed, self.vec_color = imgui.color_edit3("##vec_color", 
                                                            *self.vec_color,
                                                            imgui.COLOR_EDIT_NO_INPUTS)
            
            imgui.spacing()
            imgui.spacing()
            
            # Create button
            if self._styled_button("Create Vector", (0.2, 0.6, 0.2, 1.0), width=-1):
                self._add_vector(scene)
            
            self._end_section()

    def _render_vector_operations(self, scene, selected):
        """Render vector operations section."""
        if self._section("Vector Operations", "⚡"):
            imgui.columns(2, "##vec_ops_cols", border=False)
            
            # Basic operations
            if self._styled_button("Normalize", (0.3, 0.3, 0.6, 1.0), width=-1):
                if selected:
                    selected.normalize()
            
            imgui.next_column()
            
            if self._styled_button("Reset", (0.6, 0.4, 0.2, 1.0), width=-1):
                if selected:
                    selected.reset()
            
            imgui.next_column()
            imgui.spacing()
            
            # Scale operation
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
            
            # Vector algebra
            imgui.text("Vector Algebra:")
            imgui.spacing()
            
            if len(scene.vectors) >= 2:
                imgui.columns(2, "##algebra_cols", border=False)
                
                v1_idx = 0
                v2_idx = min(1, len(scene.vectors) - 1)
                
                # Vector selection for operations
                imgui.text("Vector 1:")
                imgui.next_column()
                imgui.text("Vector 2:")
                imgui.next_column()
                
                # Dropdowns for vector selection
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
                
                # Operation buttons
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

    def _render_matrix_operations(self, scene):
        """Render matrix operations section."""
        if self._section("Matrix Operations", "📐"):
            # Saved matrices list
            imgui.text("Saved Matrices:")
            imgui.spacing()
            imgui.begin_child("##matrix_list", 0, 120, border=True)
            if not scene.matrices:
                imgui.text_disabled("No matrices saved")
            else:
                for i, mat in enumerate(scene.matrices):
                    label = mat.get('label') or f"Matrix {i+1}"
                    shape = mat.get('matrix').shape if 'matrix' in mat else None
                    selectable_label = f"{label} ({shape[0]}x{shape[1]})##mat_{i}"
                    is_selected = (self.selected_matrix_idx == i)
                    if imgui.selectable(selectable_label, is_selected)[0]:
                        self.selected_matrix_idx = i
                        scene.selected_object = mat
                        scene.selection_type = 'matrix'
                        # Load into editor for quick edits
                        try:
                            m = mat['matrix']
                            self.matrix_size = m.shape[0] if m.ndim == 2 else self.matrix_size
                            # Convert to nested lists for editor
                            self.matrix_input = [list(row) for row in np.array(m).tolist()]
                            self.matrix_name = mat.get('label', self.matrix_name)
                            self.show_matrix_editor = True
                        except Exception:
                            pass

                    imgui.same_line()
                    if imgui.small_button(f"Del##mat_del_{i}"):
                        # Delete matrix
                        scene.remove_matrix(mat)
                        if self.selected_matrix_idx == i:
                            self.selected_matrix_idx = None
            imgui.end_child()

            imgui.spacing()
            # Toggle matrix editor
            if imgui.button("Open Matrix Editor", width=-1):
                self.show_matrix_editor = not self.show_matrix_editor
            
            imgui.spacing()
            
            if self.show_matrix_editor:
                imgui.begin_child("##matrix_editor", 0, 200, border=True)
                
                # Matrix size selector
                imgui.text("Matrix Size:")
                imgui.same_line()
                imgui.push_item_width(100)
                size_changed, self.matrix_size = imgui.slider_int("##matrix_size", 
                                                                self.matrix_size, 2, 4)
                imgui.pop_item_width()
                
                # Resize matrix if needed
                if size_changed:
                    self._resize_matrix()
                
                # Matrix input
                imgui.spacing()
                changed, self.matrix_input = self._matrix_input_widget(self.matrix_input)
                
                # Matrix name
                imgui.spacing()
                imgui.text("Name:")
                imgui.same_line()
                imgui.push_item_width(100)
                name_changed, self.matrix_name = imgui.input_text("##matrix_name", 
                                                                self.matrix_name, 16)
                imgui.pop_item_width()
                # Preview toggle
                imgui.same_line()
                prev_changed, self.preview_matrix_enabled = imgui.checkbox("Preview", self.preview_matrix_enabled)
                if prev_changed:
                    if self.preview_matrix_enabled:
                        try:
                            scene.set_preview_matrix(np.array(self.matrix_input, dtype=np.float32))
                        except Exception:
                            scene.set_preview_matrix(None)
                    else:
                        scene.set_preview_matrix(None)
                
                # Action buttons
                imgui.spacing()
                imgui.columns(3, "##matrix_buttons", border=False)

                # Add / Save matrix
                if self.selected_matrix_idx is None:
                    if imgui.button("Add Matrix", width=-1):
                        self._add_matrix(scene)
                else:
                    if imgui.button("Save Matrix", width=-1):
                        # Update existing matrix in scene
                        try:
                            matrix = np.array(self.matrix_input, dtype=np.float32)
                            scene.matrices[self.selected_matrix_idx]['matrix'] = matrix
                            scene.matrices[self.selected_matrix_idx]['label'] = self.matrix_name
                            self.operation_result = {'type': 'save_matrix', 'index': self.selected_matrix_idx}
                        except Exception as e:
                            self.operation_result = {'error': str(e)}

                imgui.next_column()

                if imgui.button("Apply to Selected", width=-1):
                    self._apply_matrix_to_selected(scene)

                imgui.next_column()

                if imgui.button("Apply to All", width=-1):
                    try:
                        matrix = np.array(self.matrix_input, dtype=np.float32)
                        scene.apply_transformation(matrix)
                        self.operation_result = {'type': 'apply_all'}
                    except Exception as e:
                        self.operation_result = {'error': str(e)}

                imgui.next_column()
                # Compute subspaces for selected matrix
                if imgui.button("Null Space", width=-1):
                    try:
                        mat = np.array(self.matrix_input, dtype=np.float32)
                        self._compute_null_space(scene, mat)
                    except Exception as e:
                        self.operation_result = {'error': str(e)}

                imgui.next_column()
                if imgui.button("Column Space", width=-1):
                    try:
                        mat = np.array(self.matrix_input, dtype=np.float32)
                        self._compute_column_space(scene, mat)
                    except Exception as e:
                        self.operation_result = {'error': str(e)}

                imgui.columns(1)
                
                imgui.end_child()
            
            self._end_section()

    def _render_linear_systems(self, scene):
        """Render linear systems solver section."""
        if self._section("Linear Systems", "🧮"):
            # Toggle equation editor
            if imgui.button("Open Equation Solver", width=-1):
                self.show_equation_editor = not self.show_equation_editor
            
            imgui.spacing()
            
            if self.show_equation_editor:
                imgui.begin_child("##equation_editor", 0, 300, border=True)
                
                # Equation count
                imgui.text("Number of equations:")
                imgui.same_line()
                imgui.push_item_width(100)
                count_changed, self.equation_count = imgui.slider_int("##eq_count", 
                                                                    self.equation_count, 2, 4)
                imgui.pop_item_width()
                
                # Resize equations if needed
                if count_changed:
                    self._resize_equations()
                
                # Equation input
                imgui.spacing()
                imgui.text("System Ax = b:")
                
                for i in range(self.equation_count):
                    imgui.push_id(str(i))
                    
                    # Equation coefficients
                    imgui.text(f"Eq {i+1}:")
                    imgui.same_line()
                    
                    # Input for each coefficient
                    for j in range(self.equation_count):
                        imgui.push_item_width(50)
                        coeff_changed, self.equation_input[i][j] = imgui.input_float(
                            f"##coeff_{i}_{j}", self.equation_input[i][j], 
                            format="%.2f"
                        )
                        imgui.pop_item_width()
                        
                        imgui.same_line()
                        if j < self.equation_count - 1:
                            imgui.text(f"x{j+1} +")
                        else:
                            imgui.text(f"x{j+1} =")
                        
                        imgui.same_line()
                    
                    # RHS input
                    imgui.push_item_width(50)
                    rhs_changed, self.equation_input[i][-1] = imgui.input_float(
                        f"##rhs_{i}", self.equation_input[i][-1], format="%.2f"
                    )
                    imgui.pop_item_width()
                    
                    imgui.pop_id()
                
                # Solve button
                imgui.spacing()
                if imgui.button("Solve System", width=-1):
                    self._solve_linear_system(scene)
                
                # Display result if available
                if self.operation_result and 'solution' in self.operation_result:
                    imgui.spacing()
                    imgui.separator()
                    imgui.spacing()
                    
                    imgui.text_colored("Solution:", 0.2, 0.8, 0.2)
                    solution = self.operation_result['solution']
                    
                    for i, val in enumerate(solution):
                        imgui.text(f"x{i+1} = {val:.4f}")
                    
                    # Add solution as vectors option
                    imgui.spacing()
                    if imgui.button("Add Solution Vectors", width=-1):
                        self._add_solution_vectors(scene, solution)
                
                imgui.end_child()
            
            self._end_section()

    def _render_visualization_options(self, scene, camera, view_config):
        """Render visualization options section."""
        if self._section("Visualization", "👁️"):
            # View preset
            imgui.text("View Preset:")
            imgui.same_line()
            
            presets = ["Cube", "XY Plane", "XZ Plane", "YZ Plane"]
            preset_values = ["cube", "xy", "xz", "yz"]
            
            current_idx = preset_values.index(view_config.grid_plane if view_config.grid_mode == "plane" else "cube")
            
            imgui.push_item_width(120)
            if imgui.begin_combo("##view_preset", presets[current_idx]):
                for i, preset in enumerate(presets):
                    if imgui.selectable(preset, i == current_idx)[0]:
                        preset_value = preset_values[i]
                        if preset_value == "cube":
                            view_config.grid_mode = "cube"
                        else:
                            view_config.grid_mode = "plane"
                            view_config.grid_plane = preset_value
                        camera.set_view_preset(preset_value)
                imgui.end_combo()
            imgui.pop_item_width()
            
            imgui.spacing()
            
            # Up axis
            imgui.text("Up Axis:")
            imgui.same_line()
            
            axes = ["Z (Standard)", "Y (Blender)", "X"]
            axis_values = ["z", "y", "x"]
            
            current_axis_idx = axis_values.index(view_config.up_axis)
            
            imgui.push_item_width(120)
            if imgui.begin_combo("##up_axis", axes[current_axis_idx]):
                for i, axis in enumerate(axes):
                    if imgui.selectable(axis, i == current_axis_idx)[0]:
                        view_config.up_axis = axis_values[i]
                imgui.end_combo()
            imgui.pop_item_width()
            
            imgui.spacing()
            
            # Toggles
            toggles = [
                ("Show Grid", 'show_grid'),
                ("Show Axes", 'show_axes'),
                ("Show Labels", 'show_labels'),
                ("2D Mode", None)  # Special case
            ]
            
            for label, attr in toggles:
                if attr is None:
                    # 2D Mode toggle (special)
                    changed, mode_2d = imgui.checkbox("2D Mode", camera.mode_2d)
                    if changed:
                        camera.mode_2d = mode_2d
                        if mode_2d:
                            view_config.grid_mode = "plane"
                else:
                    current_value = getattr(view_config, attr)
                    changed, new_value = imgui.checkbox(label, current_value)
                    if changed:
                        setattr(view_config, attr, new_value)
                
                imgui.same_line()
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"Toggle {label.replace('Show ', '')}")
            
            imgui.new_line()
            imgui.spacing()
            
            # Grid settings
            if view_config.show_grid:
                imgui.indent(10)
                imgui.text("Grid Settings:")
                
                # Grid size
                imgui.push_item_width(150)
                changed, view_config.grid_size = imgui.slider_int(
                    "Size", view_config.grid_size, 5, 50
                )
                
                # Tick spacing
                changed, view_config.major_tick = imgui.slider_int(
                    "Major Ticks", view_config.major_tick, 1, 10
                )
                
                changed, view_config.minor_tick = imgui.slider_int(
                    "Minor Ticks", view_config.minor_tick, 1, 5
                )
                imgui.pop_item_width()
                
                imgui.unindent(10)
            
            self._end_section()

    def _render_vector_list(self, scene, selected):
        """Render vector list with filtering."""
        if self._section("Vector List", "📋", default_open=True):
            # Filter input
            imgui.push_item_width(-1)
            filter_changed, self.vector_list_filter = imgui.input_text_with_hint(
                "##vector_filter", "Filter vectors...", self.vector_list_filter, 64
            )
            imgui.pop_item_width()
            
            imgui.spacing()
            
            # Vector list
            imgui.begin_child("##vector_list", 0, 200, border=True)
            
            filtered_vectors = scene.vectors
            if self.vector_list_filter:
                filter_lower = self.vector_list_filter.lower()
                filtered_vectors = [
                    v for v in scene.vectors
                    if filter_lower in v.label.lower() or
                    any(str(coord) for coord in v.coords if filter_lower in str(coord))
                ]
            
            if not filtered_vectors:
                imgui.text_disabled("No vectors match filter")
            else:
                for i, vector in enumerate(filtered_vectors):
                    is_selected = (vector is selected)
                    
                    # Display vector info
                    coords_str = f"({vector.coords[0]:.2f}, {vector.coords[1]:.2f}, {vector.coords[2]:.2f})"
                    label_text = f"{vector.label} {coords_str}"
                    
                    # Color indicator
                    imgui.push_style_color(imgui.COLOR_TEXT, *vector.color)
                    if imgui.selectable(f"##vec_{i}", is_selected)[0]:
                        selected = vector
                        scene.selected_object = vector
                        scene.selection_type = 'vector'
                    
                    # Draw color circle
                    draw_list = imgui.get_window_draw_list()
                    pos = imgui.get_cursor_screen_pos()
                    draw_list.add_circle_filled(
                        pos.x - 20, pos.y - 10,
                        5, imgui.get_color_u32_rgba(*vector.color, 1.0)
                    )
                    
                    imgui.same_line()
                    imgui.text(label_text)
                    imgui.pop_style_color(1)
                    
                    # Context menu on right-click
                    if imgui.begin_popup_context_item(f"vec_context_{i}"):
                        if imgui.menu_item("Duplicate")[0]:
                            # Duplicate vector (handled below)
                            self._duplicate_vector(scene, vector)
                        
                        if imgui.menu_item("Delete")[0]:
                            if vector is selected:
                                selected = None
                            scene.remove_vector(vector)
                        
                        imgui.end_popup()
            
            imgui.end_child()
            
            # List actions
            imgui.spacing()
            imgui.columns(2, "##list_actions", border=False)
            
            if imgui.button("Clear All", width=-1):
                scene.clear_vectors()
                selected = None
            
            imgui.next_column()
            
            if imgui.button("Export...", width=-1):
                self.show_export_dialog = True
            
            imgui.columns(1)
            
            self._end_section()
        
        return selected

    def _duplicate_vector(self, scene, vector):
        """Duplicate a vector and add to the scene with a new label."""
        try:
            new_coords = vector.coords.copy()
            # Create a unique label
            base = vector.label or "v"
            # Try to create a numbered suffix
            idx = 1
            new_label = f"{base}_copy{idx}"
            existing = {v.label for v in scene.vectors}
            while new_label in existing:
                idx += 1
                new_label = f"{base}_copy{idx}"

            v = Vector3D(new_coords, color=vector.color, label=new_label)
            scene.add_vector(v)
        except Exception:
            pass

    def _render_export_dialog(self):
        """Render export dialog."""
        if self.show_export_dialog:
            imgui.open_popup("Export Vectors")
        
        if imgui.begin_popup_modal("Export Vectors")[0]:
            imgui.text("Export format:")
            imgui.spacing()
            
            if imgui.button("JSON", width=100):
                self._export_json()
                self.show_export_dialog = False
                imgui.close_current_popup()
            
            imgui.same_line()
            
            if imgui.button("CSV", width=100):
                self._export_csv()
                self.show_export_dialog = False
                imgui.close_current_popup()
            
            imgui.same_line()
            
            if imgui.button("Python", width=100):
                self._export_python()
                self.show_export_dialog = False
                imgui.close_current_popup()
            
            imgui.end_popup()

    def render(self, height, scene, selected, camera, view_config):
        """Main render method."""
        # Keep a reference to the scene for helper methods (exports, etc.)
        self.scene = scene

        # Set window position and size
        imgui.set_next_window_position(10, 30)
        imgui.set_next_window_size(self.window_width, height - 40)
        
        # Window styling
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 8.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (12, 12))
        
        # Begin main sidebar window
        if imgui.begin("CVLA Controls", 
                      flags=imgui.WINDOW_NO_RESIZE | 
                            imgui.WINDOW_NO_MOVE |
                            imgui.WINDOW_NO_TITLE_BAR):
            
            # Header
            imgui.text_colored("🎯 CVLA - Linear Algebra Visualizer", 0.9, 0.9, 1.0, 1.0)
            imgui.text_disabled("Interactive 3D Mathematics")
            # Undo / Redo
            imgui.same_line(300)
            if imgui.button("Undo"):
                scene.undo()
            imgui.same_line()
            if imgui.button("Redo"):
                scene.redo()
            imgui.separator()
            imgui.spacing()
            
            # Tabs
            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 4.0)
            tab_names = ["Vectors", "Matrices", "Systems", "Visualize"]
            tab_values = ["vectors", "matrices", "systems", "visualize"]
            
            for i, (name, value) in enumerate(zip(tab_names, tab_values)):
                if i > 0:
                    imgui.same_line()
                
                if self.active_tab == value:
                    imgui.push_style_color(imgui.COLOR_BUTTON, 0.26, 0.59, 0.98, 0.6)
                    imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.26, 0.59, 0.98, 0.8)
                    imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.26, 0.59, 0.98, 1.0)
                else:
                    imgui.push_style_color(imgui.COLOR_BUTTON, 0.18, 0.18, 0.24, 1.0)
                    imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.26, 0.59, 0.98, 0.4)
                    imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.26, 0.59, 0.98, 0.6)
                
                if imgui.button(name, width=(self.window_width - 50) / 4):
                    self.active_tab = value
                
                imgui.pop_style_color(3)
            
            imgui.pop_style_var(1)
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            
            # Tab content
            if self.active_tab == "vectors":
                self._render_vector_creation(scene)
                if selected:
                    self._render_vector_operations(scene, selected)
                self._render_vector_list(scene, selected)
                
            elif self.active_tab == "matrices":
                self._render_matrix_operations(scene)
                
            elif self.active_tab == "systems":
                self._render_linear_systems(scene)
                
            elif self.active_tab == "visualize":
                self._render_visualization_options(scene, camera, view_config)
            
            # Export dialog
            self._render_export_dialog()
        
        imgui.end()
        imgui.pop_style_var(2)
        
        return selected

    # Vector operations
    def _add_vector(self, scene):
        """Add a new vector to the scene."""
        if not self.vec_name:
            self.vec_name = f"v{self.next_vector_id}"
        
        v = Vector3D(
            np.array(self.vec_input, dtype=np.float32),
            color=self.vec_color,
            label=self.vec_name
        )
        scene.add_vector(v)
        
        # Update state
        self.next_vector_id += 1
        self.vec_name = ""
        self.vec_color = self._get_next_color()
        self.vec_input = [1.0, 0.0, 0.0]

    def _add_vectors(self, scene, idx1, idx2):
        """Add two vectors."""
        if len(scene.vectors) >= 2:
            v1 = scene.vectors[idx1]
            v2 = scene.vectors[idx2]
            result = v1.coords + v2.coords
            
            name = f"{v1.label}+{v2.label}"
            v = Vector3D(result, color=self._get_next_color(), label=name)
            scene.add_vector(v)

    def _subtract_vectors(self, scene, idx1, idx2):
        """Subtract two vectors."""
        if len(scene.vectors) >= 2:
            v1 = scene.vectors[idx1]
            v2 = scene.vectors[idx2]
            result = v1.coords - v2.coords
            
            name = f"{v1.label}-{v2.label}"
            v = Vector3D(result, color=self._get_next_color(), label=name)
            scene.add_vector(v)

    def _cross_vectors(self, scene, idx1, idx2):
        """Cross product of two vectors."""
        if len(scene.vectors) >= 2:
            v1 = scene.vectors[idx1]
            v2 = scene.vectors[idx2]
            result = np.cross(v1.coords, v2.coords)
            
            name = f"{v1.label}x{v2.label}"
            v = Vector3D(result, color=self._get_next_color(), label=name)
            scene.add_vector(v)

    def _dot_vectors(self, scene, idx1, idx2):
        """Dot product of two vectors."""
        if len(scene.vectors) >= 2:
            v1 = scene.vectors[idx1]
            v2 = scene.vectors[idx2]
            dot = np.dot(v1.coords, v2.coords)
            
            # Store result for display
            self.operation_result = {
                'type': 'dot_product',
                'value': dot,
                'vectors': [v1.label, v2.label]
            }

    # Matrix operations
    def _resize_matrix(self):
        """Resize matrix input based on selected size."""
        new_size = self.matrix_size
        
        # Create new matrix
        new_matrix = []
        for i in range(new_size):
            row = []
            for j in range(new_size):
                if i < len(self.matrix_input) and j < len(self.matrix_input[0]):
                    row.append(self.matrix_input[i][j])
                else:
                    row.append(1.0 if i == j else 0.0)
            new_matrix.append(row)
        
        self.matrix_input = new_matrix

    def _add_matrix(self, scene):
        """Add matrix to scene."""
        matrix = np.array(self.matrix_input, dtype=np.float32)
        mat_dict = scene.add_matrix(matrix, label=self.matrix_name)

        # Select the newly added matrix so user gets immediate feedback
        scene.selected_object = mat_dict
        scene.selection_type = 'matrix'

        # Provide a small operation result so the UI can show confirmation
        self.operation_result = {
            'type': 'add_matrix',
            'label': self.matrix_name,
            'shape': mat_dict['matrix'].shape
        }

        # Optionally close editor after adding
        self.show_matrix_editor = False

    def _apply_matrix_to_selected(self, scene):
        """Apply matrix to selected vector."""
        if scene.selected_object and scene.selection_type == 'vector':
            matrix = np.array(self.matrix_input, dtype=np.float32)
            scene.apply_matrix_to_selected(matrix)

    # Linear system operations
    def _resize_equations(self):
        """Resize equation system."""
        new_count = self.equation_count
        
        # Create new equation array
        new_equations = []
        for i in range(new_count):
            if i < len(self.equation_input):
                # Copy existing row, pad or truncate as needed
                row = self.equation_input[i][:new_count + 1]
                if len(row) < new_count + 1:
                    row.extend([0.0] * (new_count + 1 - len(row)))
                new_equations.append(row)
            else:
                # Create new row
                row = [0.0] * (new_count + 1)
                row[i] = 1.0  # Diagonal element
                new_equations.append(row)
        
        self.equation_input = new_equations

    def _solve_linear_system(self, scene):
        """Solve linear system of equations."""
        try:
            # Extract A and b from equation input
            n = self.equation_count
            A = np.zeros((n, n), dtype=np.float32)
            b = np.zeros(n, dtype=np.float32)
            
            for i in range(n):
                for j in range(n):
                    A[i, j] = self.equation_input[i][j]
                b[i] = self.equation_input[i][-1]
            
            # Solve using Gaussian elimination
            result = scene.gaussian_elimination(A, b)
            self.operation_result = result
            
        except Exception as e:
            self.operation_result = {'error': str(e)}

    def _add_solution_vectors(self, scene, solution):
        """Add solution as vectors to scene."""
        # If the solution is a 3-element vector, add it as a single 3D vector
        if len(solution) == 3:
            coords = np.array(solution, dtype=np.float32)
            v = Vector3D(coords, color=self._get_next_color(), label="solution")
            scene.add_vector(v)
            return

        # Fallback: for non-3 solutions, add each component as an axis-aligned vector
        for i, val in enumerate(solution):
            coords = np.zeros(3, dtype=np.float32)
            coords[i % 3] = val  # Place along corresponding axis (wrap if >3)

            v = Vector3D(
                coords,
                color=self.color_palette[i % len(self.color_palette)],
                label=f"x{i+1}"
            )
            scene.add_vector(v)

    def _compute_null_space(self, scene, matrix):
        """Compute null space for `matrix` and add basis vectors to the scene.

        Only 3-element vectors are added as `Vector3D`; other sizes are skipped.
        """
        try:
            ns = scene.compute_null_space(np.array(matrix, dtype=np.float32))
            if ns is None or ns.size == 0:
                self.operation_result = {'type': 'null_space', 'vectors': []}
                return

            added = []
            # ns may be a 2D array where each row is a null vector or a vector list
            arr = np.array(ns)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            for i, vec in enumerate(arr):
                v_arr = np.array(vec, dtype=np.float32).flatten()
                if v_arr.size == 3:
                    name = f"ns_{i+1}"
                    v = Vector3D(v_arr, color=self._get_next_color(), label=name)
                    scene.add_vector(v)
                    added.append(name)

            self.operation_result = {'type': 'null_space', 'vectors': added}
        except Exception as e:
            self.operation_result = {'error': str(e)}

    def _compute_column_space(self, scene, matrix):
        """Compute column space for `matrix` and add basis vectors to the scene.

        Only 3-element vectors are added as `Vector3D`; other sizes are skipped.
        """
        try:
            cs = scene.compute_column_space(np.array(matrix, dtype=np.float32))
            if cs is None or cs.size == 0:
                self.operation_result = {'type': 'column_space', 'vectors': []}
                return

            added = []
            cols = np.array(cs)
            # If returned as (n, k) matrix use columns
            if cols.ndim == 2:
                for i in range(cols.shape[1]):
                    v_arr = cols[:, i].astype(np.float32).flatten()
                    if v_arr.size == 3:
                        name = f"cs_{i+1}"
                        v = Vector3D(v_arr, color=self._get_next_color(), label=name)
                        scene.add_vector(v)
                        added.append(name)

            self.operation_result = {'type': 'column_space', 'vectors': added}
        except Exception as e:
            self.operation_result = {'error': str(e)}

    # Export operations
    def _export_json(self):
        """Export vectors to JSON."""
        import json
        data = {
            'vectors': [
                {
                    'label': v.label,
                    'coords': v.coords.tolist(),
                    'color': v.color,
                    'visible': v.visible
                }
                for v in self.scene.vectors
            ]
        }
        
        # In a real implementation, you would save this to a file
        print("JSON Export (first vector):", json.dumps(data['vectors'][0], indent=2))

    def _export_csv(self):
        """Export vectors to CSV."""
        csv_lines = ["Label,X,Y,Z,R,G,B"]
        for v in self.scene.vectors:
            csv_lines.append(
                f'{v.label},{v.coords[0]},{v.coords[1]},{v.coords[2]},'
                f'{v.color[0]},{v.color[1]},{v.color[2]}'
            )
        
        # In a real implementation, you would save this to a file
        print("CSV Export (first few lines):")
        for line in csv_lines[:3]:
            print(line)

    def _export_python(self):
        """Export vectors as Python code."""
        python_code = "# CVLA Vector Export\n"
        python_code += "import numpy as np\n\n"
        python_code += "vectors = [\n"
        
        for v in self.scene.vectors:
            python_code += f"    # {v.label}\n"
            python_code += f"    np.array({v.coords.tolist()}, dtype=np.float32),\n"
        
        python_code += "]\n"
        
        print("Python Export ready")