"""
Main renderer for CVLA - Enhanced with beautiful cubic visualization
"""

import moderngl
import numpy as np
from engine.viewconfig import ViewConfig
from engine.gizmos import Gizmos


class Renderer:
    def __init__(self, ctx, camera, view=None):
        self.ctx = ctx
        self.camera = camera
        self.view = view or ViewConfig()
        
        # Initialize gizmos
        self.gizmos = Gizmos(ctx)
        
        # Rendering state
        self.vector_scale = 3.0
        self.show_vector_labels = True
        self.show_plane_visuals = True
        self.show_vector_components = True
        self.show_vector_spans = False
        
        # Enable OpenGL features
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Cache for frequently used matrices
        self._vp_cache = None
        self._vp_cache_dirty = True
        
        # Cubic view settings
        self.cube_face_colors = [
            (0.3, 0.3, 0.8, 0.05),   # XY+ face
            (0.8, 0.3, 0.3, 0.05),   # XZ+ face  
            (0.3, 0.8, 0.3, 0.05),   # YZ+ face
            (0.8, 0.8, 0.3, 0.05),   # XY- face
            (0.8, 0.3, 0.8, 0.05),   # XZ- face
            (0.3, 0.8, 0.8, 0.05),   # YZ- face
        ]

    def _get_view_projection(self):
        """Get cached view-projection matrix."""
        if self._vp_cache_dirty or self._vp_cache is None:
            self._vp_cache = self.camera.vp()
            self._vp_cache_dirty = False
        return self._vp_cache
    
    def update_view(self, view_config):
        """Update view configuration."""
        self.view = view_config
        self._vp_cache_dirty = True
        
        # Update rendering parameters from view config
        if hasattr(view_config, 'vector_scale'):
            self.vector_scale = view_config.vector_scale
        
        if hasattr(view_config, 'show_plane_visuals'):
            self.show_plane_visuals = view_config.show_plane_visuals

    def render(self, scene):
        """Main rendering method."""
        # Clear buffers with gradient background
        self._clear_with_gradient()
        
        # Get current view-projection matrix
        vp = self._get_view_projection()
        
        # Render cubic environment
        if self.view.grid_mode == "cube":
            self._render_cubic_environment(vp, scene)
        else:
            self._render_planar_environment(vp)
        
        # Render special linear algebra visualizations
        self._render_linear_algebra_visuals(scene, vp)
        
        # Render vectors with enhanced visualization
        self._render_vectors_with_enhancements(scene, vp)
        
        # Render selection highlights
        if scene.selected_object and scene.selection_type == 'vector':
            self._render_selection_highlight(scene.selected_object, vp)

    def _clear_with_gradient(self):
        """Clear with a subtle gradient background."""
        if self.view.grid_mode == "cube":
            # Darker gradient for cubic view
            self.ctx.clear(
                color=(0.05, 0.06, 0.08, 1.0),
                depth=1.0
            )
        else:
            # Standard gradient
            self.ctx.clear(
                color=(0.08, 0.08, 0.10, 1.0),
                depth=1.0
            )

    def _render_cubic_environment(self, vp, scene):
        """Render a beautiful 3D cubic environment."""
        if self.view.show_grid:
            # Draw the main cubic grid
            self.gizmos.draw_cubic_grid(
                vp, 
                size=self.view.grid_size,
                major_step=self.view.major_tick,
                minor_step=self.view.minor_tick,
                color_major=(0.35, 0.37, 0.4, 0.9),
                color_minor=(0.2, 0.22, 0.25, 0.6)
            )
            
            # Draw translucent cube faces for better spatial awareness
            if self.show_plane_visuals:
                self._render_cube_faces(vp)
            
            # Draw axis indicators on cube corners
            self._render_cube_corner_indicators(vp)
        
        if self.view.show_axes:
            self._render_3d_axes_with_depths(vp)
    
    def _render_planar_environment(self, vp):
        """Render planar grid environment."""
        if self.view.show_grid:
            self.gizmos.draw_grid(
                vp, 
                size=self.view.grid_size, 
                step=self.view.minor_tick, 
                plane=self.view.grid_plane
            )
        
        if self.view.show_axes:
            self.gizmos.draw_axes(vp, length=6.0, thickness=3.0)

    def _render_cube_faces(self, vp):
        """Draw translucent faces of the bounding cube."""
        size = float(self.view.grid_size)
        
        # Define cube faces (6 faces, 2 triangles each)
        faces = [
            # XY+ face (z = size)
            {"normal": [0, 0, 1], "vertices": [
                [-size, -size, size], [size, -size, size], 
                [size, size, size], [-size, size, size]
            ]},
            # XY- face (z = -size)
            {"normal": [0, 0, -1], "vertices": [
                [-size, -size, -size], [-size, size, -size],
                [size, size, -size], [size, -size, -size]
            ]},
            # XZ+ face (y = size)
            {"normal": [0, 1, 0], "vertices": [
                [-size, size, -size], [-size, size, size],
                [size, size, size], [size, size, -size]
            ]},
            # XZ- face (y = -size)
            {"normal": [0, -1, 0], "vertices": [
                [-size, -size, -size], [size, -size, -size],
                [size, -size, size], [-size, -size, size]
            ]},
            # YZ+ face (x = size)
            {"normal": [1, 0, 0], "vertices": [
                [size, -size, -size], [size, size, -size],
                [size, size, size], [size, -size, size]
            ]},
            # YZ- face (x = -size)
            {"normal": [-1, 0, 0], "vertices": [
                [-size, -size, -size], [-size, -size, size],
                [-size, size, size], [-size, size, -size]
            ]}
        ]
        
        for i, face in enumerate(faces):
            vertices = face["vertices"]
            normal = face["normal"]
            
            # Create two triangles for each face
            tri_vertices = [
                vertices[0], vertices[1], vertices[2],  # Triangle 1
                vertices[0], vertices[2], vertices[3]   # Triangle 2
            ]
            
            # All vertices have same normal
            normals = [normal] * 6
            
            # Get face color
            color = self.cube_face_colors[i % len(self.cube_face_colors)]
            
            # Draw the face
            self.gizmos.draw_triangles(
                tri_vertices, normals, [color] * 6, 
                vp, use_lighting=False
            )
            
            # Draw face border
            border_vertices = [
                vertices[0], vertices[1],
                vertices[1], vertices[2],
                vertices[2], vertices[3],
                vertices[3], vertices[0]
            ]
            border_color = (color[0]*1.5, color[1]*1.5, color[2]*1.5, 0.8)
            self.gizmos.draw_lines(
                border_vertices, [border_color] * 8, 
                vp, width=1.0
            )

    def _render_cube_corner_indicators(self, vp):
        """Draw indicators at cube corners showing axis directions."""
        size = float(self.view.grid_size)
        corners = [
            [size, size, size],    # +++
            [size, size, -size],   # ++-
            [size, -size, size],   # +-+
            [size, -size, -size],  # +--
            [-size, size, size],   # -++
            [-size, size, -size],  # -+-
            [-size, -size, size],  # --+
            [-size, -size, -size], # ---
        ]
        
        # Draw small spheres at corners
        corner_color = (0.8, 0.8, 0.9, 0.6)
        self.gizmos.draw_points(corners, [corner_color] * len(corners), vp, size=4.0)
        
        # Draw axis lines extending from corners
        axis_length = size * 0.2
        for corner in corners:
            # X axis line (red)
            x_end = [corner[0] + axis_length, corner[1], corner[2]]
            self.gizmos.draw_lines(
                [corner, x_end], 
                [(1.0, 0.3, 0.3, 0.7), (1.0, 0.3, 0.3, 0.7)], 
                vp, width=1.5
            )
            
            # Y axis line (green)
            y_end = [corner[0], corner[1] + axis_length, corner[2]]
            self.gizmos.draw_lines(
                [corner, y_end], 
                [(0.3, 1.0, 0.3, 0.7), (0.3, 1.0, 0.3, 0.7)], 
                vp, width=1.5
            )
            
            # Z axis line (blue)
            z_end = [corner[0], corner[1], corner[2] + axis_length]
            self.gizmos.draw_lines(
                [corner, z_end], 
                [(0.3, 0.5, 1.0, 0.7), (0.3, 0.5, 1.0, 0.7)], 
                vp, width=1.5
            )

    def _render_3d_axes_with_depths(self, vp):
        """Render 3D axes with depth cues."""
        length = 8.0
        
        # Main axes from origin
        axes = [
            {"points": [[0, 0, 0], [length, 0, 0]], "color": (1.0, 0.3, 0.3, 1.0)},
            {"points": [[0, 0, 0], [0, length, 0]], "color": (0.3, 1.0, 0.3, 1.0)},
            {"points": [[0, 0, 0], [0, 0, length]], "color": (0.3, 0.5, 1.0, 1.0)},
        ]
        
        for axis in axes:
            self.gizmos.draw_lines(
                axis["points"], 
                [axis["color"], axis["color"]], 
                vp, width=3.0
            )
        
        # Draw extended faint axes through entire space
        size = self.view.grid_size * 1.5
        faint_axes = [
            {"points": [[-size, 0, 0], [size, 0, 0]], "color": (1.0, 0.3, 0.3, 0.15)},
            {"points": [[0, -size, 0], [0, size, 0]], "color": (0.3, 1.0, 0.3, 0.15)},
            {"points": [[0, 0, -size], [0, 0, size]], "color": (0.3, 0.5, 1.0, 0.15)},
        ]
        
        for axis in faint_axes:
            self.gizmos.draw_lines(
                axis["points"], 
                [axis["color"], axis["color"]], 
                vp, width=1.0
            )
        
        # Draw axis cones
        self._draw_axis_cones(vp, length)

    def _draw_axis_cones(self, vp, length):
        """Draw 3D cones at axis tips."""
        cone_height = length * 0.15
        cone_radius = length * 0.05
        
        axes = [
            {"tip": [length, 0, 0], "direction": [1, 0, 0], "color": (1.0, 0.3, 0.3, 1.0)},
            {"tip": [0, length, 0], "direction": [0, 1, 0], "color": (0.3, 1.0, 0.3, 1.0)},
            {"tip": [0, 0, length], "direction": [0, 0, 1], "color": (0.3, 0.5, 1.0, 1.0)},
        ]
        
        for axis in axes:
            tip = np.array(axis["tip"])
            direction = np.array(axis["direction"])
            base = tip - direction * cone_height
            
            # Create cone geometry
            cone_verts = []
            cone_norms = []
            cone_colors = []
            segments = 12
            
            # Choose perpendicular vectors based on axis
            if abs(direction[0]) < 0.9:
                perp1 = np.array([1.0, 0.0, 0.0])
            else:
                perp1 = np.array([0.0, 1.0, 0.0])
            
            perp2 = np.cross(direction, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)
            perp1 = np.cross(perp2, direction)
            
            for i in range(segments):
                angle1 = 2 * np.pi * i / segments
                angle2 = 2 * np.pi * (i + 1) / segments
                
                p1 = base + cone_radius * (np.cos(angle1) * perp1 + np.sin(angle1) * perp2)
                p2 = base + cone_radius * (np.cos(angle2) * perp1 + np.sin(angle2) * perp2)
                
                # Triangle: base->p1->tip
                cone_verts.extend(base.tolist())
                cone_verts.extend(p1.tolist())
                cone_verts.extend(tip.tolist())
                
                # Calculate normals
                v1 = p1 - base
                v2 = tip - base
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)
                
                cone_norms.extend([normal.tolist()] * 3)
                cone_colors.extend([axis["color"]] * 3)
            
            self.gizmos.draw_triangles(cone_verts, cone_norms, cone_colors, vp)

    def _render_linear_algebra_visuals(self, scene, vp):
        """Render linear algebra visualizations."""
        # Render vector spans if enabled
        if self.show_vector_spans and len(scene.vectors) >= 2:
            # Show span between first two vectors
            self.gizmos.draw_vector_span(vp, scene.vectors[0], scene.vectors[1])
        
        # Render parallelepiped for three vectors
        if len(scene.vectors) >= 3:
            self.gizmos.draw_parallelepiped(vp, scene.vectors[:3])
        
        # Render basis transformations
        for matrix_dict in scene.matrices:
            if not matrix_dict['visible']:
                continue
            
            matrix = matrix_dict['matrix']
            
            # Visualize as transformation of standard basis
            basis_vectors = [
                np.array([1, 0, 0], dtype='f4'),
                np.array([0, 1, 0], dtype='f4'),
                np.array([0, 0, 1], dtype='f4')
            ]
            
            transformed_basis = []
            for basis in basis_vectors:
                if matrix.shape == (3, 3):
                    transformed = matrix @ basis
                elif matrix.shape == (4, 4):
                    point = np.array([basis[0], basis[1], basis[2], 1.0])
                    transformed_h = matrix @ point
                    transformed = transformed_h[:3] / transformed_h[3]
                else:
                    continue
                transformed_basis.append(transformed)
            
            # Draw basis transformation visualization
            self.gizmos.draw_basis_transform(
                vp, 
                basis_vectors, 
                transformed_basis,
                show_original=True,
                show_transformed=True
            )

    def _render_vectors_with_enhancements(self, scene, vp):
        """Render all vectors with enhanced visualizations."""
        # Auto-scale vectors based on scene bounds
        try:
            if len(scene.vectors) > 0:
                visible_vectors = [v for v in scene.vectors if v.visible]
                if visible_vectors:
                    max_mag = max([np.linalg.norm(v.coords) for v in visible_vectors] + [1.0])
                    desired = max(1.0, self.camera.radius * 0.2)
                    scale_factor = desired / max_mag
                    scale_factor = float(np.clip(scale_factor, 0.3, 5.0))
                    self.vector_scale = float(self.view.vector_scale) * scale_factor
                else:
                    self.vector_scale = float(self.view.vector_scale)
        except Exception:
            self.vector_scale = float(self.view.vector_scale)
        
        # Render each vector
        for vector in scene.vectors:
            if vector.visible:
                is_selected = (vector is scene.selected_object and 
                              scene.selection_type == 'vector')
                
                self.gizmos.draw_vector_with_details(
                    vp, vector, is_selected, self.vector_scale,
                    show_components=self.show_vector_components,
                    show_span=False
                )
                
                # Draw vector projection lines if in 2D mode
                if self.camera.mode_2d:
                    self._render_vector_projections(vector, vp)

    def _render_vector_projections(self, vector, vp):
        """Render vector projection lines in 2D mode."""
        tip = vector.coords * self.vector_scale
        
        # Project onto current view plane
        if self.camera.view_preset == "xy":
            # Project onto XY plane
            proj_tip = np.array([tip[0], tip[1], 0])
            proj_color = (vector.color[0], vector.color[1], vector.color[2], 0.3)
            
            # Draw projection line
            line = [[tip[0], tip[1], tip[2]], [proj_tip[0], proj_tip[1], proj_tip[2]]]
            self.gizmos.draw_lines(line, [proj_color, proj_color], vp, width=1.0)
            
            # Draw projection point
            self.gizmos.draw_points([proj_tip], [proj_color], vp, size=4.0)
        
        elif self.camera.view_preset == "xz":
            # Project onto XZ plane
            proj_tip = np.array([tip[0], 0, tip[2]])
            proj_color = (vector.color[0], vector.color[1], vector.color[2], 0.3)
            
            line = [[tip[0], tip[1], tip[2]], [proj_tip[0], proj_tip[1], proj_tip[2]]]
            self.gizmos.draw_lines(line, [proj_color, proj_color], vp, width=1.0)
            self.gizmos.draw_points([proj_tip], [proj_color], vp, size=4.0)
        
        elif self.camera.view_preset == "yz":
            # Project onto YZ plane
            proj_tip = np.array([0, tip[1], tip[2]])
            proj_color = (vector.color[0], vector.color[1], vector.color[2], 0.3)
            
            line = [[tip[0], tip[1], tip[2]], [proj_tip[0], proj_tip[1], proj_tip[2]]]
            self.gizmos.draw_lines(line, [proj_color, proj_color], vp, width=1.0)
            self.gizmos.draw_points([proj_tip], [proj_color], vp, size=4.0)

    def _render_selection_highlight(self, vector, vp):
        """Render special highlight for selected vector."""
        tip = vector.coords * self.vector_scale
        
        # Draw a pulsating circle around the vector tip
        import time
        pulse = (np.sin(time.time() * 5) * 0.2 + 0.8)  # 0.6 to 1.0
        
        circle_verts = []
        circle_colors = []
        segments = 24
        radius = 0.4 * pulse
        
        for i in range(segments):
            angle1 = 2 * np.pi * i / segments
            angle2 = 2 * np.pi * (i + 1) / segments
            
            # Create circle in XY plane (adjust based on view)
            p1 = tip + radius * np.array([np.cos(angle1), np.sin(angle1), 0])
            p2 = tip + radius * np.array([np.cos(angle2), np.sin(angle2), 0])
            
            circle_verts.extend(p1.tolist())
            circle_verts.extend(p2.tolist())
            
            # Pulsating yellow color
            pulse_color = (1.0, 1.0, 0.2, 0.7 * pulse)
            circle_colors.extend([pulse_color, pulse_color])
        
        self.gizmos.draw_lines(circle_verts, circle_colors, vp, width=2.0, depth=False)
        
        # Draw line from origin to tip with highlight
        origin_line = [[0, 0, 0], tip.tolist()]
        highlight_color = (1.0, 1.0, 0.2, 0.8)
        self.gizmos.draw_lines(origin_line, [highlight_color, highlight_color], vp, width=2.0, depth=False)
        
        # Draw selection sphere at tip
        sphere_color = (1.0, 1.0, 0.2, 0.9)
        self.gizmos.draw_points([tip.tolist()], [sphere_color], vp, size=14.0)