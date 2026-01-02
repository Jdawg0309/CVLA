"""
Gizmos: immediate-mode debug drawing with enhanced cubic visualization
"""

import numpy as np
import moderngl


class Gizmos:
    def __init__(self, ctx):
        self.ctx = ctx
        
        # Shader programs
        self.line_program = self._create_line_program()
        self.triangle_program = self._create_triangle_program()
        self.point_program = self._create_point_program()
        self.volume_program = self._create_volume_program()  # For volume visualization
        
        # VAOs and VBOs
        self.line_vao = None
        self.line_vbo = None
        self.triangle_vao = None
        self.triangle_vbo = None
        self.point_vao = None
        self.point_vbo = None
        self.volume_vao = None
        self.volume_vbo = None
        
        self._init_buffers()
    
    def _create_line_program(self):
        """Create shader program for line rendering."""
        return self.ctx.program(
            vertex_shader="""
            #version 330
            in vec3 in_position;
            in vec4 in_color;
            uniform mat4 mvp;
            out vec4 v_color;
            void main() {
                gl_Position = mvp * vec4(in_position, 1.0);
                v_color = in_color;
            }
            """,
            fragment_shader="""
            #version 330
            in vec4 v_color;
            out vec4 frag_color;
            void main() {
                frag_color = v_color;
            }
            """
        )
    
    def _create_triangle_program(self):
        """Create shader program for triangle rendering (planes)."""
        return self.ctx.program(
            vertex_shader="""
            #version 330
            in vec3 in_position;
            in vec3 in_normal;
            in vec4 in_color;
            uniform mat4 mvp;
            uniform mat4 model;
            uniform vec3 light_pos;
            out vec4 v_color;
            out vec3 v_normal;
            out vec3 v_position;
            void main() {
                vec4 world_pos = model * vec4(in_position, 1.0);
                v_position = world_pos.xyz;
                v_normal = mat3(model) * in_normal;
                v_color = in_color;
                gl_Position = mvp * world_pos;
            }
            """,
            fragment_shader="""
            #version 330
            in vec4 v_color;
            in vec3 v_normal;
            in vec3 v_position;
            uniform vec3 light_pos;
            uniform vec3 view_pos;
            uniform bool use_lighting;
            out vec4 frag_color;
            
            vec3 calculate_lighting(vec3 base_color) {
                if (!use_lighting) return base_color;
                
                // Light properties
                vec3 light_color = vec3(1.0, 1.0, 0.95);
                vec3 ambient_color = vec3(0.15, 0.15, 0.2);
                
                // Ambient
                vec3 ambient = ambient_color * base_color;
                
                // Diffuse
                vec3 norm = normalize(v_normal);
                vec3 light_dir = normalize(light_pos - v_position);
                float diff = max(dot(norm, light_dir), 0.0);
                vec3 diffuse = light_color * diff * base_color;
                
                // Specular
                vec3 view_dir = normalize(view_pos - v_position);
                vec3 reflect_dir = reflect(-light_dir, norm);
                float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
                vec3 specular = light_color * spec * 0.5;
                
                return ambient + diffuse + specular;
            }
            
            void main() {
                vec3 lit_color = calculate_lighting(v_color.rgb);
                frag_color = vec4(lit_color, v_color.a);
            }
            """
        )
    
    def _create_point_program(self):
        """Create shader program for point rendering."""
        return self.ctx.program(
            vertex_shader="""
            #version 330
            in vec3 in_position;
            in vec4 in_color;
            uniform mat4 mvp;
            uniform float point_size;
            out vec4 v_color;
            void main() {
                gl_Position = mvp * vec4(in_position, 1.0);
                gl_PointSize = point_size;
                v_color = in_color;
            }
            """,
            fragment_shader="""
            #version 330
            in vec4 v_color;
            out vec4 frag_color;
            void main() {
                vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
                float dist = dot(circCoord, circCoord);
                if (dist > 1.0) {
                    discard;
                }
                // Smooth circle edge
                float alpha = smoothstep(1.0, 0.8, dist);
                frag_color = vec4(v_color.rgb, v_color.a * alpha);
            }
            """
        )
    
    def _create_volume_program(self):
        """Create shader program for volume visualization (spans, parallelepiped)."""
        return self.ctx.program(
            vertex_shader="""
            #version 330
            in vec3 in_position;
            in vec4 in_color;
            uniform mat4 mvp;
            uniform float opacity;
            out vec4 v_color;
            void main() {
                gl_Position = mvp * vec4(in_position, 1.0);
                v_color = vec4(in_color.rgb, in_color.a * opacity);
            }
            """,
            fragment_shader="""
            #version 330
            in vec4 v_color;
            out vec4 frag_color;
            void main() {
                frag_color = v_color;
            }
            """
        )
    
    def _init_buffers(self):
        """Initialize vertex buffers."""
        # Line buffer with color attribute
        self.line_vbo = self.ctx.buffer(reserve=2 * 1024 * 1024)  # 2MB for lines
        self.line_vao = self.ctx.vertex_array(
            self.line_program,
            [
                (self.line_vbo, '3f 4f', 'in_position', 'in_color')
            ]
        )
        
        # Triangle buffer with color attribute
        self.triangle_vbo = self.ctx.buffer(reserve=2 * 1024 * 1024)  # 2MB for triangles
        self.triangle_vao = self.ctx.vertex_array(
            self.triangle_program,
            [
                (self.triangle_vbo, '3f 3f 4f', 'in_position', 'in_normal', 'in_color')
            ]
        )
        
        # Point buffer with color attribute
        self.point_vbo = self.ctx.buffer(reserve=512 * 1024)  # 512KB for points
        self.point_vao = self.ctx.vertex_array(
            self.point_program,
            [(self.point_vbo, '3f 4f', 'in_position', 'in_color')]
        )
        
        # Volume buffer
        self.volume_vbo = self.ctx.buffer(reserve=1024 * 1024)  # 1MB for volumes
        self.volume_vao = self.ctx.vertex_array(
            self.volume_program,
            [(self.volume_vbo, '3f 4f', 'in_position', 'in_color')]
        )
    
    # -----------------------------------
    # Core drawing methods
    # -----------------------------------
    def draw_lines(self, vertices, colors, vp, width=2.0, depth=True):
        """Draw line segments with per-vertex colors."""
        if not vertices or len(vertices) == 0:
            return
        
        if depth:
            self.ctx.enable(moderngl.DEPTH_TEST)
        else:
            self.ctx.disable(moderngl.DEPTH_TEST)
        
        self.ctx.line_width = float(width)
        
        # Convert vertices and colors to interleaved array
        vertices = np.array(vertices, dtype='f4').reshape(-1, 3)
        colors = np.array(colors, dtype='f4').reshape(-1, 4)
        
        # Interleave vertices and colors
        interleaved = np.zeros((vertices.shape[0], 7), dtype='f4')
        interleaved[:, :3] = vertices
        interleaved[:, 3:7] = colors
        
        self.line_vbo.write(interleaved.tobytes())
        
        # Set uniforms
        self.line_program['mvp'].write(vp.astype('f4').tobytes())
        
        # Draw
        self.line_vao.render(moderngl.LINES, vertices=vertices.shape[0])
    
    def draw_triangles(self, vertices, normals, colors, vp, model_matrix=None, 
                      use_lighting=True, view_pos=(0, 0, 5), depth=True):
        """Draw triangles with per-vertex colors."""
        if not vertices or len(vertices) == 0:
            return
        
        if depth:
            self.ctx.enable(moderngl.DEPTH_TEST)
        else:
            self.ctx.disable(moderngl.DEPTH_TEST)
        
        # Enable polygon offset to avoid z-fighting (if supported)
        if hasattr(moderngl, "POLYGON_OFFSET_FILL"):
            self.ctx.enable(moderngl.POLYGON_OFFSET_FILL)
            self.ctx.polygon_offset = 1.0, 1.0
        
        # Prepare data
        vertices = np.array(vertices, dtype='f4').reshape(-1, 3)
        normals = np.array(normals, dtype='f4').reshape(-1, 3)
        colors = np.array(colors, dtype='f4').reshape(-1, 4)
        
        # Interleave vertices, normals, and colors
        interleaved = np.zeros((vertices.shape[0], 10), dtype='f4')
        interleaved[:, :3] = vertices
        interleaved[:, 3:6] = normals
        interleaved[:, 6:10] = colors
        
        self.triangle_vbo.write(interleaved.tobytes())
        
        # Set uniforms
        self.triangle_program['mvp'].write(vp.astype('f4').tobytes())
        
        if model_matrix is None:
            model_matrix = np.eye(4, dtype='f4')
        self.triangle_program['model'].write(model_matrix.astype('f4').tobytes())
        
        self.triangle_program['light_pos'].value = (5.0, 5.0, 5.0)
        self.triangle_program['view_pos'].value = view_pos
        self.triangle_program['use_lighting'].value = use_lighting
        
        # Draw
        self.triangle_vao.render(moderngl.TRIANGLES, vertices=vertices.shape[0])
        
        # Disable polygon offset (if supported)
        if hasattr(moderngl, "POLYGON_OFFSET_FILL"):
            self.ctx.disable(moderngl.POLYGON_OFFSET_FILL)
    
    def draw_points(self, vertices, colors, vp, size=8.0, depth=True):
        """Draw points with per-vertex colors."""
        if not vertices or len(vertices) == 0:
            return
        
        if depth:
            self.ctx.enable(moderngl.DEPTH_TEST)
        else:
            self.ctx.disable(moderngl.DEPTH_TEST)
        
        # Convert data
        vertices = np.array(vertices, dtype='f4').reshape(-1, 3)
        colors = np.array(colors, dtype='f4').reshape(-1, 4)
        
        # Interleave
        interleaved = np.zeros((vertices.shape[0], 7), dtype='f4')
        interleaved[:, :3] = vertices
        interleaved[:, 3:7] = colors
        
        self.point_vbo.write(interleaved.tobytes())
        
        # Set uniforms
        self.point_program['mvp'].write(vp.astype('f4').tobytes())
        self.point_program['point_size'].value = float(size)
        
        # Draw
        self.point_vao.render(moderngl.POINTS, vertices=vertices.shape[0])
    
    def draw_volume(self, vertices, colors, vp, opacity=0.3, depth=True):
        """Draw volume (spans, parallelepiped)."""
        if not vertices or len(vertices) == 0:
            return
        
        if depth:
            self.ctx.enable(moderngl.DEPTH_TEST)
        else:
            self.ctx.disable(moderngl.DEPTH_TEST)
        
        # Enable blending for transparency
        self.ctx.enable(moderngl.BLEND)
        
        # Convert data
        vertices = np.array(vertices, dtype='f4').reshape(-1, 3)
        colors = np.array(colors, dtype='f4').reshape(-1, 4)
        
        # Interleave
        interleaved = np.zeros((vertices.shape[0], 7), dtype='f4')
        interleaved[:, :3] = vertices
        interleaved[:, 3:7] = colors
        
        self.volume_vbo.write(interleaved.tobytes())
        
        # Set uniforms
        self.volume_program['mvp'].write(vp.astype('f4').tobytes())
        self.volume_program['opacity'].value = float(opacity)
        
        # Draw as triangles
        self.volume_vao.render(moderngl.TRIANGLES, vertices=vertices.shape[0])
        
        self.ctx.disable(moderngl.BLEND)
    
    # -----------------------------------
    # High-level drawing methods
    # -----------------------------------
    def draw_cubic_grid(self, vp, size=10, major_step=5, minor_step=1, 
                       color_major=(0.25, 0.27, 0.3, 0.9),
                       color_minor=(0.15, 0.16, 0.18, 0.6)):
        """Draw a beautiful 3D cubic grid."""
        vertices = []
        colors = []
        
        # Generate grid lines for all three planes
        for plane in ['xy', 'xz', 'yz']:
            for i in range(-size, size + 1, minor_step):
                # Determine line colors based on tick type
                is_major = (i % major_step == 0)
                color = color_major if is_major else color_minor
                
                if plane == 'xy':
                    # X lines in XY plane
                    vertices.extend([[i, -size, 0], [i, size, 0]])
                    colors.extend([color, color])
                    # Y lines in XY plane
                    vertices.extend([[-size, i, 0], [size, i, 0]])
                    colors.extend([color, color])
                elif plane == 'xz':
                    # X lines in XZ plane
                    vertices.extend([[i, 0, -size], [i, 0, size]])
                    colors.extend([color, color])
                    # Z lines in XZ plane
                    vertices.extend([[-size, 0, i], [size, 0, i]])
                    colors.extend([color, color])
                elif plane == 'yz':
                    # Y lines in YZ plane
                    vertices.extend([[0, i, -size], [0, i, size]])
                    colors.extend([color, color])
                    # Z lines in YZ plane
                    vertices.extend([[0, -size, i], [0, size, i]])
                    colors.extend([color, color])
        
        # Draw the grid lines
        self.draw_lines(vertices, colors, vp, width=1.0)
        
        # Draw bounding cube edges
        self.draw_cube(vp, [-size, -size, -size], [size, size, size], 
                      (0.35, 0.35, 0.38, 0.8), width=2.0)
    
    def draw_cube(self, vp, min_corner, max_corner, color, width=2.0):
        """Draw a wireframe cube."""
        x0, y0, z0 = min_corner
        x1, y1, z1 = max_corner
        
        vertices = [
            # Bottom face
            [x0, y0, z0], [x1, y0, z0],
            [x1, y0, z0], [x1, y1, z0],
            [x1, y1, z0], [x0, y1, z0],
            [x0, y1, z0], [x0, y0, z0],
            
            # Top face
            [x0, y0, z1], [x1, y0, z1],
            [x1, y0, z1], [x1, y1, z1],
            [x1, y1, z1], [x0, y1, z1],
            [x0, y1, z1], [x0, y0, z1],
            
            # Vertical edges
            [x0, y0, z0], [x0, y0, z1],
            [x1, y0, z0], [x1, y0, z1],
            [x1, y1, z0], [x1, y1, z1],
            [x0, y1, z0], [x0, y1, z1]
        ]
        
        colors = [color] * len(vertices)
        self.draw_lines(vertices, colors, vp, width=width)
    
    def draw_vector_with_details(self, vp, vector, selected=False, scale=1.0, 
                                show_components=True, show_span=False):
        """
        Draw a vector with enhanced visualizations.
        
        Args:
            vector: Vector3D object
            show_components: Show projection to axes
            show_span: Show span plane for 2 vectors
        """
        tip = vector.coords * scale
        length = np.linalg.norm(tip)
        
        if length < 1e-6:
            return
        
        # Color
        r, g, b = vector.color
        if selected:
            r = min(1.0, r * 1.5)
            g = min(1.0, g * 1.5)
            b = min(1.0, b * 1.5)
            alpha = 1.0
            shaft_width = 5.0
        else:
            alpha = 1.0
            shaft_width = 3.0
        
        color = (r, g, b, alpha)
        
        # Draw main vector
        shaft_verts = [[0, 0, 0], tip.tolist()]
        shaft_colors = [color, color]
        self.draw_lines(shaft_verts, shaft_colors, vp, width=shaft_width)
        
        # Draw arrow head
        self._draw_arrow_head(vp, tip, vector.coords, color, shaft_width)
        
        # Draw projection lines to axes if enabled
        if show_components:
            self._draw_vector_components(vp, tip, color)
        
        # Draw tip point
        self.draw_points([tip.tolist()], [color], vp, size=12.0 if selected else 8.0)
    
    def _draw_arrow_head(self, vp, tip, direction, color, shaft_width):
        """Draw a 3D arrow head."""
        length = np.linalg.norm(tip)
        if length < 0.3:  # Too small for arrow head
            return
        
        dir_norm = direction / length
        head_length = min(0.4, length * 0.3)
        head_radius = head_length * 0.4
        
        head_base = tip - dir_norm * head_length
        
        # Create perpendicular basis
        if abs(dir_norm[0]) < 0.9:
            perp = np.array([1.0, 0.0, 0.0])
        else:
            perp = np.array([0.0, 1.0, 0.0])
        right = np.cross(dir_norm, perp)
        rnorm = np.linalg.norm(right)
        if rnorm > 1e-8:
            right = right / rnorm
        else:
            right = np.array([0.0, 1.0, 0.0])
        up = np.cross(right, dir_norm)
        
        # Create head vertices
        head_verts = []
        head_colors = []
        segments = 8
        
        for i in range(segments):
            angle1 = 2 * np.pi * i / segments
            angle2 = 2 * np.pi * (i + 1) / segments
            
            p1 = head_base + head_radius * (np.cos(angle1) * right + np.sin(angle1) * up)
            p2 = head_base + head_radius * (np.cos(angle2) * right + np.sin(angle2) * up)
            
            # Lines from tip to base circle
            head_verts.extend(tip.tolist())
            head_verts.extend(p1.tolist())
            head_colors.extend([color, color])
            
            # Lines around base circle
            head_verts.extend(p1.tolist())
            head_verts.extend(p2.tolist())
            head_colors.extend([color, color])
        
        self.draw_lines(head_verts, head_colors, vp, width=shaft_width)
    
    def _draw_vector_components(self, vp, tip, color):
        """Draw projection lines to axes."""
        proj_color = (color[0], color[1], color[2], 0.4)
        
        # Projection to XY plane
        xy_proj = [tip[0], tip[1], 0]
        vertices = [tip.tolist(), xy_proj]
        colors = [proj_color, proj_color]
        self.draw_lines(vertices, colors, vp, width=1.0)
        
        # Projection to XZ plane
        xz_proj = [tip[0], 0, tip[2]]
        vertices = [tip.tolist(), xz_proj]
        self.draw_lines(vertices, colors, vp, width=1.0)
        
        # Projection to YZ plane
        yz_proj = [0, tip[1], tip[2]]
        vertices = [tip.tolist(), yz_proj]
        self.draw_lines(vertices, colors, vp, width=1.0)
    
    def draw_vector_span(self, vp, vector1, vector2, color=(0.2, 0.4, 0.8, 0.3)):
        """Draw the span (parallelogram) between two vectors."""
        v1 = vector1.coords
        v2 = vector2.coords
        
        # Vertices of the parallelogram
        vertices = [
            [0, 0, 0],      # Origin
            v1.tolist(),    # Tip of v1
            (v1 + v2).tolist(),  # Tip of v1+v2
            v2.tolist()     # Tip of v2
        ]
        
        # Two triangles to make the parallelogram
        tri_vertices = [
            vertices[0], vertices[1], vertices[2],  # Triangle 1
            vertices[0], vertices[2], vertices[3]   # Triangle 2
        ]
        
        # Normals (same for all vertices)
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else [0, 0, 1]
        normals = [normal.tolist()] * 6
        
        # Colors with slight variation
        colors = []
        for i in range(6):
            if i < 3:
                # First triangle - slight gradient
                colors.append((color[0], color[1], color[2], color[3] * 0.8))
            else:
                # Second triangle
                colors.append((color[0], color[1], color[2], color[3] * 0.6))
        
        self.draw_triangles(tri_vertices, normals, colors, vp, use_lighting=True)
        
        # Draw border
        border_vertices = [
            vertices[0], vertices[1],
            vertices[1], vertices[2],
            vertices[2], vertices[3],
            vertices[3], vertices[0]
        ]
        border_colors = [(color[0]*0.7, color[1]*0.7, color[2]*0.7, 1.0)] * 8
        self.draw_lines(border_vertices, border_colors, vp, width=1.5)
    
    def draw_parallelepiped(self, vp, vectors, color=(0.3, 0.6, 0.9, 0.2)):
        """Draw parallelepiped spanned by three vectors."""
        if len(vectors) < 3:
            return
        
        v1, v2, v3 = vectors[0].coords, vectors[1].coords, vectors[2].coords
        
        # All 8 vertices of the parallelepiped
        vertices = [
            np.array([0, 0, 0]),                    # 0
            v1,                                     # 1
            v2,                                     # 2
            v3,                                     # 3
            v1 + v2,                                # 4
            v1 + v3,                                # 5
            v2 + v3,                                # 6
            v1 + v2 + v3                            # 7
        ]
        
        # Define 12 edges (wireframe)
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 5),
            (2, 4), (2, 6),
            (3, 5), (3, 6),
            (4, 7), (5, 7), (6, 7)
        ]
        
        # Draw edges
        edge_vertices = []
        for i, j in edges:
            edge_vertices.append(vertices[i].tolist())
            edge_vertices.append(vertices[j].tolist())
        
        edge_colors = [(color[0]*0.8, color[1]*0.8, color[2]*0.8, 1.0)] * len(edge_vertices)
        self.draw_lines(edge_vertices, edge_colors, vp, width=2.0)
        
        # Draw faces (transparent)
        faces = [
            [vertices[0], vertices[1], vertices[4], vertices[2]],  # v1-v2 face
            [vertices[0], vertices[1], vertices[5], vertices[3]],  # v1-v3 face
            [vertices[0], vertices[2], vertices[6], vertices[3]],  # v2-v3 face
            [vertices[7], vertices[5], vertices[1], vertices[4]],  # opposite v1-v2
            [vertices[7], vertices[6], vertices[2], vertices[4]],  # opposite v2-v3
            [vertices[7], vertices[6], vertices[3], vertices[5]]   # opposite v1-v3
        ]
        
        for face in faces:
            # Each face as two triangles
            tri_vertices = [
                face[0].tolist(), face[1].tolist(), face[2].tolist(),
                face[0].tolist(), face[2].tolist(), face[3].tolist()
            ]
            
            # Calculate normal for the face
            normal = np.cross(face[1] - face[0], face[2] - face[0])
            normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else [0, 0, 1]
            normals = [normal.tolist()] * 6
            
            # Face colors with slight gradient
            face_color = (color[0], color[1], color[2], color[3] * 0.15)
            face_colors = [face_color] * 6
            
            self.draw_triangles(tri_vertices, normals, face_colors, vp, use_lighting=False)
    
    def draw_basis_transform(self, vp, original_basis, transformed_basis, 
                           show_original=True, show_transformed=True):
        """Visualize basis transformation."""
        colors = [
            (1.0, 0.3, 0.3, 0.8),  # Red for X
            (0.3, 1.0, 0.3, 0.8),  # Green for Y
            (0.3, 0.5, 1.0, 0.8),  # Blue for Z
        ]
        
        if show_original:
            for i, basis in enumerate(original_basis):
                length = np.linalg.norm(basis)
                if length > 0:
                    tip = basis * 2.0 / length if length > 2.0 else basis
                    vertices = [[0, 0, 0], tip.tolist()]
                    line_colors = [colors[i], colors[i]]
                    self.draw_lines(vertices, line_colors, vp, width=2.0)
        
        if show_transformed:
            for i, basis in enumerate(transformed_basis):
                length = np.linalg.norm(basis)
                if length > 0:
                    tip = basis * 2.0 / length if length > 2.0 else basis
                    vertices = [[0, 0, 0], tip.tolist()]
                    # Brighter colors for transformed basis
                    bright_color = (
                        min(1.0, colors[i][0] * 1.5),
                        min(1.0, colors[i][1] * 1.5),
                        min(1.0, colors[i][2] * 1.5),
                        1.0
                    )
                    line_colors = [bright_color, bright_color]
                    self.draw_lines(vertices, line_colors, vp, width=3.0)
                    
                    # Draw connection line from original to transformed
                    if show_original and i < len(original_basis):
                        orig_tip = original_basis[i]
                        orig_tip = orig_tip * 2.0 / np.linalg.norm(orig_tip) if np.linalg.norm(orig_tip) > 2.0 else orig_tip
                        vertices = [orig_tip.tolist(), tip.tolist()]
                        line_colors = [(0.8, 0.8, 0.2, 0.5), (0.8, 0.8, 0.2, 0.5)]
                        self.draw_lines(vertices, line_colors, vp, width=1.0)

    def draw_grid(self, vp, size=10, step=1, plane='xy', color_major=(0.25,0.27,0.3,0.9), color_minor=(0.15,0.16,0.18,0.6)):
        """Draw a simple planar grid on specified plane ('xy','xz','yz')."""
        vertices = []
        colors = []
        half = int(size)
        step = int(step) if step > 0 else 1

        for i in range(-half, half + 1):
            is_major = (i % (step * 5) == 0)
            color = color_major if is_major else color_minor

            if plane == 'xy':
                vertices.extend([[i, -half, 0], [i, half, 0]])
                colors.extend([color, color])
                vertices.extend([[-half, i, 0], [half, i, 0]])
                colors.extend([color, color])
            elif plane == 'xz':
                vertices.extend([[i, 0, -half], [i, 0, half]])
                colors.extend([color, color])
                vertices.extend([[-half, 0, i], [half, 0, i]])
                colors.extend([color, color])
            elif plane == 'yz':
                vertices.extend([[0, i, -half], [0, i, half]])
                colors.extend([color, color])
                vertices.extend([[0, -half, i], [0, half, i]])
                colors.extend([color, color])

        self.draw_lines(vertices, colors, vp, width=1.0)

    def draw_axes(self, vp, length=6.0, thickness=3.0):
        """Draw basic XYZ axes as lines with endpoints."""
        axes = [
            ([[0,0,0], [length,0,0]], (1.0,0.3,0.3,1.0)),
            ([[0,0,0], [0,length,0]], (0.3,1.0,0.3,1.0)),
            ([[0,0,0], [0,0,length]], (0.3,0.5,1.0,1.0)),
        ]
        for pts, col in axes:
            self.draw_lines(pts, [col, col], vp, width=thickness)