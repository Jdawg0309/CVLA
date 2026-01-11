"""
Shader program creation helpers.
"""


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

            vec3 light_color = vec3(1.0, 1.0, 0.95);
            vec3 ambient_color = vec3(0.15, 0.15, 0.2);

            vec3 ambient = ambient_color * base_color;

            vec3 norm = normalize(v_normal);
            vec3 light_dir = normalize(light_pos - v_position);
            float diff = max(dot(norm, light_dir), 0.0);
            vec3 diffuse = light_color * diff * base_color;

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
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);
            if (dist > 0.5) discard;
            float alpha = 1.0 - smoothstep(0.4, 0.5, dist);
            frag_color = vec4(v_color.rgb, v_color.a * alpha);
        }
        """
    )


def _create_volume_program(self):
    """Create shader program for volume visualization."""
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
