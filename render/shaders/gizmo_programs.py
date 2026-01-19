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
    """Create shader program for triangle rendering with PBR-lite materials."""
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

        // PBR-lite material parameters
        const float roughness = 0.5;
        const float metallic = 0.0;
        const float ambient_strength = 0.15;

        // Fresnel-Schlick approximation
        vec3 fresnelSchlick(float cosTheta, vec3 F0) {
            return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
        }

        // Hemisphere ambient lighting
        vec3 hemisphereAmbient(vec3 normal, vec3 base_color) {
            vec3 sky_color = vec3(0.25, 0.28, 0.35);
            vec3 ground_color = vec3(0.08, 0.08, 0.10);
            float hemisphere = normal.y * 0.5 + 0.5;
            return mix(ground_color, sky_color, hemisphere) * base_color;
        }

        vec3 calculate_lighting(vec3 base_color) {
            if (!use_lighting) return base_color;

            vec3 N = normalize(v_normal);
            vec3 V = normalize(view_pos - v_position);
            vec3 L = normalize(light_pos - v_position);
            vec3 H = normalize(V + L);

            // Hemisphere ambient
            vec3 ambient = hemisphereAmbient(N, base_color) * ambient_strength;

            // Diffuse (Lambertian)
            float NdotL = max(dot(N, L), 0.0);
            vec3 light_color = vec3(1.0, 0.98, 0.95);
            vec3 diffuse = light_color * NdotL * base_color * (1.0 - metallic);

            // Fresnel-Schlick for specular
            vec3 F0 = mix(vec3(0.04), base_color, metallic);
            vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

            // Specular with roughness
            float NdotH = max(dot(N, H), 0.0);
            float shininess = mix(8.0, 128.0, 1.0 - roughness);
            float spec = pow(NdotH, shininess);
            vec3 specular = F * spec * (1.0 - roughness * 0.5);

            // Rim lighting for depth separation
            float rim = 1.0 - max(dot(N, V), 0.0);
            rim = smoothstep(0.6, 1.0, rim);
            vec3 rim_color = vec3(0.3, 0.35, 0.45) * rim * 0.4;

            // Secondary fill light (soft)
            vec3 fill_dir = normalize(vec3(-1.0, 0.5, -0.5));
            float fill = max(dot(N, fill_dir), 0.0) * 0.15;
            vec3 fill_light = vec3(0.6, 0.65, 0.8) * fill * base_color;

            return ambient + diffuse + specular + rim_color + fill_light;
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
