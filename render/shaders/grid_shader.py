"""
Infinite procedural grid shader.

GPU-based grid rendering with anti-aliased lines using screen-space derivatives.
Supports XY, XZ, and YZ planes with configurable colors and scales.
"""

INFINITE_GRID_VS = """
#version 330

// Fullscreen triangle technique - no vertex buffer needed
out vec3 v_near_point;
out vec3 v_far_point;

uniform mat4 u_view;
uniform mat4 u_projection;

// Grid vertices covering entire NDC space
vec3 grid_plane[6] = vec3[](
    vec3(1, 1, 0), vec3(-1, -1, 0), vec3(-1, 1, 0),
    vec3(-1, -1, 0), vec3(1, 1, 0), vec3(1, -1, 0)
);

vec3 unproject_point(float x, float y, float z, mat4 view, mat4 projection) {
    mat4 view_inv = inverse(view);
    mat4 proj_inv = inverse(projection);
    vec4 unprojected = view_inv * proj_inv * vec4(x, y, z, 1.0);
    return unprojected.xyz / unprojected.w;
}

void main() {
    vec3 p = grid_plane[gl_VertexID];
    v_near_point = unproject_point(p.x, p.y, -1.0, u_view, u_projection);
    v_far_point = unproject_point(p.x, p.y, 1.0, u_view, u_projection);
    gl_Position = vec4(p, 1.0);
}
"""

INFINITE_GRID_FS = """
#version 330

in vec3 v_near_point;
in vec3 v_far_point;

out vec4 fragColor;

uniform mat4 u_view;
uniform mat4 u_projection;

// Grid configuration
uniform float u_scale;           // Base grid scale
uniform float u_major_scale;     // Major grid line scale (e.g., 5x base)
uniform float u_fade_distance;   // Distance at which grid fades out

// Plane configuration (0=XY, 1=XZ, 2=YZ)
uniform int u_plane;

// Colors
uniform vec4 u_color_minor;
uniform vec4 u_color_major;
uniform vec4 u_color_axis_x;
uniform vec4 u_color_axis_y;
uniform vec4 u_color_axis_z;

float compute_depth(vec3 pos) {
    vec4 clip_space = u_projection * u_view * vec4(pos, 1.0);
    return (clip_space.z / clip_space.w) * 0.5 + 0.5;
}

float pristine_grid(vec2 uv, vec2 line_width) {
    // Use fwidth for anti-aliasing
    vec2 dd = fwidth(uv);
    vec2 grid_uv = abs(fract(uv - 0.5) - 0.5);
    vec2 grid = smoothstep(line_width + dd, line_width, grid_uv);
    return max(grid.x, grid.y);
}

float grid_line(float coord, float line_width) {
    float dd = fwidth(coord);
    float grid = abs(fract(coord - 0.5) - 0.5);
    return smoothstep(line_width + dd, line_width, grid);
}

void main() {
    // Compute intersection with the grid plane
    vec3 ray_dir = v_far_point - v_near_point;

    float t;
    vec2 grid_uv;
    int axis1, axis2;  // The two axes that form the grid plane
    float plane_denominator = 0.0;
    float near_coord = 0.0;

    if (u_plane == 0) {
        // XY plane (Z = 0)
        axis1 = 0;  // X
        axis2 = 1;  // Y
        plane_denominator = ray_dir.z;
        near_coord = v_near_point.z;
    } else if (u_plane == 1) {
        // XZ plane (Y = 0)
        axis1 = 0;  // X
        axis2 = 2;  // Z
        plane_denominator = ray_dir.y;
        near_coord = v_near_point.y;
    } else {
        // YZ plane (X = 0)
        axis1 = 1;  // Y
        axis2 = 2;  // Z
        plane_denominator = ray_dir.x;
        near_coord = v_near_point.x;
    }

    if (abs(plane_denominator) < 1e-5) {
        discard;
    }

    t = -near_coord / plane_denominator;
    if (t < 0.0) {
        discard;
    }

    vec3 frag_pos = v_near_point + t * ray_dir;

    // Get grid coordinates based on plane
    if (u_plane == 0) {
        grid_uv = frag_pos.xy;
    } else if (u_plane == 1) {
        grid_uv = frag_pos.xz;
    } else {
        grid_uv = frag_pos.yz;
    }

    // Compute grid lines at two scales
    vec2 scaled_uv = grid_uv / u_scale;
    vec2 major_uv = grid_uv / (u_scale * u_major_scale);

    float line_width = 0.02;
    float minor = pristine_grid(scaled_uv, vec2(line_width));
    float major = pristine_grid(major_uv, vec2(line_width * 1.5));

    // Distance-based fade
    float dist = length(frag_pos);
    float fade = 1.0 - smoothstep(u_fade_distance * 0.5, u_fade_distance, dist);

    // Combine minor and major grids
    vec4 color = u_color_minor * minor;
    color = mix(color, u_color_major, major * u_color_major.a);

    // Axis highlighting
    float axis_width = 0.04;

    // Highlight first axis of the plane
    float axis1_line;
    if (u_plane == 0) {
        axis1_line = grid_line(frag_pos.y / u_scale, axis_width);  // X-axis line (Y=0)
    } else if (u_plane == 1) {
        axis1_line = grid_line(frag_pos.z / u_scale, axis_width);  // X-axis line (Z=0)
    } else {
        axis1_line = grid_line(frag_pos.z / u_scale, axis_width);  // Y-axis line (Z=0)
    }

    // Highlight second axis of the plane
    float axis2_line;
    if (u_plane == 0) {
        axis2_line = grid_line(frag_pos.x / u_scale, axis_width);  // Y-axis line (X=0)
    } else if (u_plane == 1) {
        axis2_line = grid_line(frag_pos.x / u_scale, axis_width);  // Z-axis line (X=0)
    } else {
        axis2_line = grid_line(frag_pos.y / u_scale, axis_width);  // Z-axis line (Y=0)
    }

    // Apply axis colors
    vec4 axis1_color, axis2_color;
    if (u_plane == 0) {
        // XY plane: X-axis (red) and Y-axis (green)
        axis1_color = u_color_axis_x;
        axis2_color = u_color_axis_y;
    } else if (u_plane == 1) {
        // XZ plane: X-axis (red) and Z-axis (blue)
        axis1_color = u_color_axis_x;
        axis2_color = u_color_axis_z;
    } else {
        // YZ plane: Y-axis (green) and Z-axis (blue)
        axis1_color = u_color_axis_y;
        axis2_color = u_color_axis_z;
    }

    color = mix(color, axis1_color, axis1_line * axis1_color.a * 0.8);
    color = mix(color, axis2_color, axis2_line * axis2_color.a * 0.8);

    // Apply fade
    color.a *= fade;

    // Discard nearly transparent fragments
    if (color.a < 0.001) {
        discard;
    }

    // Write depth
    float depth = compute_depth(frag_pos);
    gl_FragDepth = depth;

    fragColor = color;
}
"""


def get_grid_shaders():
    """Return the infinite grid vertex and fragment shader sources."""
    return INFINITE_GRID_VS, INFINITE_GRID_FS
