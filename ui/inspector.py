"""
Inspector panel for detailed object inspection and editing
"""

import imgui
import numpy as np
from core.vector import Vector3D


class Inspector:
    def __init__(self):
        self.window_width = 320
        self.show_transform_history = True
        self.show_computed_properties = True
        
    def render(self, scene, selected_vector, screen_width, screen_height):
        """Render inspector panel."""
        if not selected_vector or not isinstance(selected_vector, Vector3D):
            return
        
        # Position window on right side
        imgui.set_next_window_position(screen_width - self.window_width - 10, 30)
        imgui.set_next_window_size(self.window_width, screen_height - 40)
        
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 8.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (12, 12))
        
        if imgui.begin("Inspector", 
                      flags=imgui.WINDOW_NO_RESIZE | 
                            imgui.WINDOW_NO_MOVE |
                            imgui.WINDOW_NO_TITLE_BAR):
            
            # Header with vector info
            self._render_header(selected_vector)
            imgui.separator()
            
            # Coordinate editing
            self._render_coordinate_editor(selected_vector)
            imgui.separator()
            
            # Vector properties
            self._render_properties(selected_vector, scene)
            imgui.separator()
            
            # Transform history
            if self.show_transform_history:
                self._render_transform_history(selected_vector)
                imgui.separator()
            
            # Computed properties
            if self.show_computed_properties:
                self._render_computed_properties(selected_vector, scene)
            
        imgui.end()
        imgui.pop_style_var(2)
    
    def _render_header(self, vector):
        """Render inspector header."""
        # Color indicator
        draw_list = imgui.get_window_draw_list()
        pos = imgui.get_cursor_screen_pos()
        draw_list.add_circle_filled(
            pos.x + 15, pos.y + 15,
            10, imgui.get_color_u32_rgba(*vector.color, 1.0)
        )
        
        imgui.dummy(30, 0)
        imgui.same_line()
        
        # Vector label
        imgui.push_font()  # Use larger font for title
        imgui.text_colored(vector.label, 0.9, 0.9, 1.0, 1.0)
        imgui.pop_font()
        
        imgui.same_line()
        imgui.text_disabled("(Vector)")
        
        # Type indicator
        imgui.text_disabled("3D Position Vector")
        
        # Visibility toggle
        imgui.same_line(200)
        changed, vector.visible = imgui.checkbox("Visible", vector.visible)
    
    def _render_coordinate_editor(self, vector):
        """Render coordinate editor."""
        imgui.text_colored("Coordinates", 0.8, 0.8, 0.2, 1.0)
        imgui.spacing()
        
        # Current coordinates display
        x, y, z = vector.coords
        imgui.text(f"X: {x:.6f}")
        imgui.same_line(100)
        imgui.text(f"Y: {y:.6f}")
        imgui.same_line(200)
        imgui.text(f"Z: {z:.6f}")
        
        imgui.spacing()
        
        # Coordinate editing
        imgui.push_item_width(80)
        
        # X coordinate
        imgui.text("X:")
        imgui.same_line()
        x_changed, new_x = imgui.input_float("##edit_x", x, format="%.4f")
        if x_changed:
            vector.coords[0] = new_x
        
        imgui.same_line(120)
        
        # Y coordinate
        imgui.text("Y:")
        imgui.same_line()
        y_changed, new_y = imgui.input_float("##edit_y", y, format="%.4f")
        if y_changed:
            vector.coords[1] = new_y
        
        imgui.same_line(240)
        
        # Z coordinate
        imgui.text("Z:")
        imgui.same_line()
        z_changed, new_z = imgui.input_float("##edit_z", z, format="%.4f")
        if z_changed:
            vector.coords[2] = new_z
        
        imgui.pop_item_width()
        
        # Quick edit buttons
        imgui.spacing()
        imgui.columns(3, "##quick_edit", border=False)
        
        quick_edits = [
            ("Zero", [0, 0, 0], (0.5, 0.5, 0.5, 1.0)),
            ("Unit X", [1, 0, 0], (1.0, 0.3, 0.3, 1.0)),
            ("Unit Y", [0, 1, 0], (0.3, 1.0, 0.3, 1.0)),
            ("Unit Z", [0, 0, 1], (0.3, 0.5, 1.0, 1.0)),
            ("Normalize", None, (0.3, 0.3, 0.8, 1.0)),  # Special case
            ("Reset", None, (0.8, 0.5, 0.2, 1.0))       # Special case
        ]
        
        for i, (label, coords, color) in enumerate(quick_edits):
            if i > 0 and i % 3 == 0:
                imgui.next_column()
            
            imgui.push_style_color(imgui.COLOR_BUTTON, *color)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 
                                 color[0]*1.2, color[1]*1.2, color[2]*1.2, 1.0)
            
            if imgui.button(label, width=-1):
                if coords is not None:
                    vector.coords = np.array(coords, dtype=np.float32)
                elif label == "Normalize":
                    vector.normalize()
                elif label == "Reset":
                    vector.reset()
            
            imgui.pop_style_color(2)
            
            if i % 3 != 2:
                imgui.same_line()
        
        imgui.columns(1)
    
    def _render_properties(self, vector, scene):
        """Render vector properties."""
        imgui.text_colored("Properties", 0.8, 0.8, 0.2, 1.0)
        imgui.spacing()
        
        # Magnitude
        magnitude = vector.magnitude()
        imgui.text(f"Magnitude: {magnitude:.6f}")
        
        # Color editor
        imgui.spacing()
        imgui.text("Color:")
        imgui.same_line()
        
        color_changed, new_color = imgui.color_edit3("##vec_color_edit", 
                                                   *vector.color,
                                                   imgui.COLOR_EDIT_NO_INPUTS)
        if color_changed:
            vector.color = new_color
        
        # Label editor
        imgui.spacing()
        imgui.text("Label:")
        imgui.same_line()
        
        imgui.push_item_width(150)
        label_changed, new_label = imgui.input_text("##vec_label_edit", 
                                                  vector.label, 32)
        imgui.pop_item_width()
        
        if label_changed:
            vector.label = new_label
        
        # Metadata
        if vector.metadata:
            imgui.spacing()
            imgui.text("Metadata:")
            for key, value in vector.metadata.items():
                imgui.text_disabled(f"  {key}: {value}")
    
    def _render_transform_history(self, vector):
        """Render transformation history."""
        if imgui.collapsing_header("Transform History", 
                                  flags=imgui.TREE_NODE_DEFAULT_OPEN):
            if not vector.history:
                imgui.text_disabled("No transformations applied")
            else:
                for i, (op, param) in enumerate(vector.history):
                    if op == 'scale':
                        imgui.text(f"{i+1}. Scaled by {param:.4f}")
                    elif op == 'normalize':
                        imgui.text(f"{i+1}. Normalized")
                    elif op == 'transform':
                        imgui.text(f"{i+1}. Matrix transform")
                
                # Clear history button
                imgui.spacing()
                if imgui.button("Clear History", width=-1):
                    vector.history.clear()
    
    def _render_computed_properties(self, vector, scene):
        """Render computed properties relative to other vectors."""
        if imgui.collapsing_header("Computed Properties", 
                                  flags=imgui.TREE_NODE_DEFAULT_OPEN):
            
            if len(scene.vectors) > 1:
                # Select another vector for comparison
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
                    
                    # Compute properties
                    imgui.spacing()
                    
                    # Dot product
                    dot = vector.dot(current_other)
                    imgui.text(f"Dot product: {dot:.6f}")
                    
                    # Angle
                    angle_deg = vector.angle(current_other, degrees=True)
                    imgui.text(f"Angle: {angle_deg:.2f}°")
                    
                    # Cross product preview
                    imgui.spacing()
                    if imgui.button("Compute Cross Product", width=-1):
                        result = vector.cross(current_other)
                        scene.add_vector(result)
                
                else:
                    imgui.text_disabled("No other vectors to compare with")
            
            else:
                imgui.text_disabled("Add another vector for comparisons")
            
            # Projection onto axes
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