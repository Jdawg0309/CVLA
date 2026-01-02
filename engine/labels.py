"""
Enhanced Label Renderer with improved readability and dynamic scaling
"""

import numpy as np
import imgui


class LabelRenderer:
    def __init__(self):
        self.viewconfig = None
        self.font = None
        self.label_cache = {}
        
        # Style settings
        self.axis_colors = {
            'x': (1.0, 0.3, 0.3, 1.0),
            'y': (0.3, 1.0, 0.3, 1.0),
            'z': (0.3, 0.5, 1.0, 1.0)
        }
        
        self.grid_color = (0.8, 0.8, 0.9, 0.9)
        self.shadow_color = (0.0, 0.0, 0.0, 0.7)
        self.background_color = (0.1, 0.1, 0.12, 0.8)
        
        self.label_offsets = {
            'axis': (8, -12),
            'grid': (4, -8),
            'vector': (6, 6)
        }
        
    def update_view(self, viewconfig):
        """Update view configuration."""
        self.viewconfig = viewconfig
    
    def world_to_screen(self, camera, world_pos, width, height):
        """Convert world coordinates to screen coordinates."""
        screen_pos = camera.world_to_screen(world_pos, width, height)
        if screen_pos is None:
            return None, None, None
        
        x, y, depth = screen_pos
        return x, y, depth
    
    def draw_axes(self, camera, width, height):
        """Draw axis labels with proper positioning."""
        if not self.viewconfig or not self.viewconfig.show_labels:
            return
        
        dl = imgui.get_background_draw_list()
        
        # Get axis vectors from view config
        axis_vectors = self.viewconfig.axis_vectors()
        
        # Axis endpoints
        axis_length = 6.0
        endpoints = {
            'x': np.array([axis_length, 0, 0], dtype=np.float32),
            'y': np.array([0, axis_length, 0], dtype=np.float32),
            'z': np.array([0, 0, axis_length], dtype=np.float32)
        }
        
        # Apply axis mapping based on up_axis
        if self.viewconfig.up_axis == 'y':
            endpoints = {
                'x': np.array([axis_length, 0, 0], dtype=np.float32),
                'y': np.array([0, 0, axis_length], dtype=np.float32),  # Z becomes Y
                'z': np.array([0, axis_length, 0], dtype=np.float32)   # Y becomes Z
            }
        elif self.viewconfig.up_axis == 'x':
            endpoints = {
                'x': np.array([0, axis_length, 0], dtype=np.float32),  # Y becomes X
                'y': np.array([0, 0, axis_length], dtype=np.float32),  # Z becomes Y
                'z': np.array([axis_length, 0, 0], dtype=np.float32)   # X becomes Z
            }
        
        # Draw axis labels
        for axis in ['x', 'y', 'z']:
            endpoint = endpoints[axis]
            x, y, depth = self.world_to_screen(camera, endpoint, width, height)
            
            if x is not None and depth > -0.9:  # Don't draw if too far behind
                label = self._get_axis_label(axis)
                color = self.axis_colors[axis]
                offset_x, offset_y = self.label_offsets['axis']
                
                # Draw shadow
                shadow_color = imgui.get_color_u32_rgba(*self.shadow_color)
                dl.add_text(x + offset_x + 1, y + offset_y + 1, shadow_color, label)
                
                # Draw main text
                text_color = imgui.get_color_u32_rgba(*color)
                dl.add_text(x + offset_x, y + offset_y, text_color, label)
    
    def draw_grid_numbers(self, camera, width, height, viewconfig=None, grid_size=20, major=5):
        """Draw grid coordinate numbers."""
        if not self.viewconfig or not self.viewconfig.show_labels:
            return
        
        if viewconfig:
            self.viewconfig = viewconfig
        
        dl = imgui.get_background_draw_list()
        
        # Get active planes
        active_planes = self._get_active_planes()
        
        # Adjust density based on camera distance
        camera_dist = max(1.0, camera.radius)
        density_factor = max(1, int(camera_dist / 8))
        step = major * density_factor
        
        # Draw labels for each active plane
        for plane in active_planes:
            for i in range(-grid_size, grid_size + 1, step):
                if i == 0:
                    continue
                
                # Generate world positions for this tick
                positions = self._get_grid_positions(plane, i, grid_size)
                
                for pos, axis in positions:
                    x, y, depth = self.world_to_screen(camera, pos, width, height)
                    
                    if x is not None and -0.8 < depth < 0.8:
                        label = str(i)
                        color = self.axis_colors[axis]
                        offset_x, offset_y = self.label_offsets['grid']
                        
                        # Calculate background size
                        text_size = imgui.calc_text_size(label)
                        padding = 2
                        
                        # Draw background for readability
                        bg_color = imgui.get_color_u32_rgba(*self.background_color)
                        dl.add_rect_filled(
                            x + offset_x - padding,
                            y + offset_y - padding,
                            x + offset_x + text_size.x + padding,
                            y + offset_y + text_size.y + padding,
                            bg_color, 3.0
                        )
                        
                        # Draw shadow
                        shadow_color = imgui.get_color_u32_rgba(*self.shadow_color)
                        dl.add_text(x + offset_x + 1, y + offset_y + 1, shadow_color, label)
                        
                        # Draw text
                        text_color = imgui.get_color_u32_rgba(*color)
                        dl.add_text(x + offset_x, y + offset_y, text_color, label)
    
    def draw_vector_labels(self, camera, vectors, width, height, selected_vector=None):
        """Draw labels for vectors."""
        if not self.viewconfig or not self.viewconfig.show_labels:
            return
        
        dl = imgui.get_background_draw_list()
        
        for vector in vectors:
            if not vector.visible or not vector.label:
                continue
            
            # Get vector tip position
            x, y, depth = self.world_to_screen(camera, vector.coords, width, height)
            
            if x is not None and depth > -0.5:
                label = vector.label
                offset_x, offset_y = self.label_offsets['vector']
                
                # Determine color
                if vector is selected_vector:
                    # Brighten selected vector label
                    r, g, b = vector.color
                    color = (min(1.0, r * 1.5), min(1.0, g * 1.5), min(1.0, b * 1.5), 1.0)
                else:
                    color = (0.9, 0.9, 0.9, 0.9)
                
                # Calculate text size
                text_size = imgui.calc_text_size(label)
                padding = 4
                
                # Draw background
                bg_color = imgui.get_color_u32_rgba(*self.background_color)
                dl.add_rect_filled(
                    x + offset_x - padding,
                    y + offset_y - padding,
                    x + offset_x + text_size.x + padding,
                    y + offset_y + text_size.y + padding,
                    bg_color, 4.0
                )
                
                # Draw border if selected
                if vector is selected_vector:
                    border_color = imgui.get_color_u32_rgba(*color)
                    dl.add_rect(
                        x + offset_x - padding,
                        y + offset_y - padding,
                        x + offset_x + text_size.x + padding,
                        y + offset_y + text_size.y + padding,
                        border_color, 4.0, 1.0
                    )
                
                # Draw text
                text_color = imgui.get_color_u32_rgba(*color)
                dl.add_text(x + offset_x, y + offset_y, text_color, label)
    
    def _get_axis_label(self, axis):
        """Get formatted axis label based on view config."""
        if self.viewconfig.up_axis == 'z':
            labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}
        elif self.viewconfig.up_axis == 'y':
            labels = {'x': 'X', 'y': 'Z', 'z': 'Y'}  # Swapped
        else:  # 'x'
            labels = {'x': 'Y', 'y': 'Z', 'z': 'X'}  # Rotated
        
        return labels.get(axis, axis.upper())
    
    def _get_active_planes(self):
        """Get active grid planes based on view config."""
        if not self.viewconfig:
            return ['xy']
        
        if self.viewconfig.grid_mode == "cube":
            return ['xy', 'xz', 'yz']
        else:
            return [self.viewconfig.grid_plane]
    
    def _get_grid_positions(self, plane, value, grid_size):
        """Get world positions for grid labels."""
        positions = []
        
        if plane == 'xy':
            # X-axis labels
            positions.append((np.array([value, 0, 0], dtype=np.float32), 'x'))
            # Y-axis labels
            positions.append((np.array([0, value, 0], dtype=np.float32), 'y'))
        elif plane == 'xz':
            # X-axis labels
            positions.append((np.array([value, 0, 0], dtype=np.float32), 'x'))
            # Z-axis labels
            positions.append((np.array([0, 0, value], dtype=np.float32), 'z'))
        elif plane == 'yz':
            # Y-axis labels
            positions.append((np.array([0, value, 0], dtype=np.float32), 'y'))
            # Z-axis labels
            positions.append((np.array([0, 0, value], dtype=np.float32), 'z'))
        
        return positions