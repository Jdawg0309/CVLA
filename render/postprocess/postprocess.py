"""
Post-processing pipeline orchestrator.

Manages HDR rendering, bloom, and tonemapping for CVLA.
"""

import moderngl
import numpy as np

from render.postprocess.bloom import BloomEffect
from render.themes.color_themes import ColorTheme, DEFAULT_THEME, get_theme


# Composite shader with ACES tonemapping
COMPOSITE_VS = """
#version 330

in vec2 in_position;
in vec2 in_texcoord;

out vec2 v_texcoord;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_texcoord = in_texcoord;
}
"""

COMPOSITE_FS = """
#version 330

uniform sampler2D u_scene;
uniform sampler2D u_bloom0;
uniform sampler2D u_bloom1;
uniform sampler2D u_bloom2;
uniform sampler2D u_bloom3;
uniform sampler2D u_bloom4;

uniform float u_bloom_intensity;
uniform float u_exposure;
uniform float u_gamma;
uniform bool u_bloom_enabled;

in vec2 v_texcoord;
out vec4 fragColor;

// ACES filmic tonemapping
vec3 aces_tonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 scene = texture(u_scene, v_texcoord).rgb;

    vec3 bloom = vec3(0.0);
    if (u_bloom_enabled) {
        // Multi-scale bloom compositing with decreasing weights
        bloom += texture(u_bloom0, v_texcoord).rgb * 0.5;
        bloom += texture(u_bloom1, v_texcoord).rgb * 0.3;
        bloom += texture(u_bloom2, v_texcoord).rgb * 0.15;
        bloom += texture(u_bloom3, v_texcoord).rgb * 0.04;
        bloom += texture(u_bloom4, v_texcoord).rgb * 0.01;
        bloom *= u_bloom_intensity;
    }

    // Combine scene with bloom
    vec3 color = scene + bloom;

    // Apply exposure
    color *= u_exposure;

    // ACES tonemapping
    color = aces_tonemap(color);

    // Gamma correction
    color = pow(color, vec3(1.0 / u_gamma));

    fragColor = vec4(color, 1.0);
}
"""


class PostProcessPipeline:
    """
    Manages the complete post-processing pipeline.

    Pipeline stages:
    1. Render scene to HDR framebuffer (RGBA16F)
    2. Extract bright pixels (threshold)
    3. Downsample + Gaussian blur (mip chain)
    4. Composite with ACES tonemapping
    """

    def __init__(self, ctx: moderngl.Context, width: int, height: int,
                 theme: ColorTheme = None):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.enabled = True
        self.bloom_enabled = True

        # Theme parameters
        if theme is None:
            theme = get_theme(DEFAULT_THEME)
        self.bloom_intensity = theme.bloom_intensity
        self.bloom_threshold = theme.bloom_threshold
        self.exposure = theme.exposure
        self.gamma = theme.gamma

        # Create HDR scene framebuffer
        self.hdr_texture = ctx.texture((width, height), 4, dtype='f2')
        self.hdr_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self.depth_texture = ctx.depth_texture((width, height))

        self.hdr_fbo = ctx.framebuffer(
            color_attachments=[self.hdr_texture],
            depth_attachment=self.depth_texture,
        )

        # Create bloom effect
        self.bloom = BloomEffect(ctx, width, height, mip_levels=5)

        # Create composite shader
        self.composite_program = ctx.program(
            vertex_shader=COMPOSITE_VS,
            fragment_shader=COMPOSITE_FS,
        )

        # Create fullscreen quad
        vertices = np.array([
            # position    # texcoord
            -1.0, -1.0,   0.0, 0.0,
             1.0, -1.0,   1.0, 0.0,
            -1.0,  1.0,   0.0, 1.0,
             1.0,  1.0,   1.0, 1.0,
        ], dtype='f4')

        self.quad_vbo = ctx.buffer(vertices)
        self.composite_vao = ctx.vertex_array(
            self.composite_program,
            [(self.quad_vbo, '2f 2f', 'in_position', 'in_texcoord')],
        )

    def resize(self, width: int, height: int):
        """Resize all framebuffers and textures."""
        if width == self.width and height == self.height:
            return

        self.width = width
        self.height = height

        # Recreate HDR framebuffer
        self.hdr_texture.release()
        self.depth_texture.release()
        self.hdr_fbo.release()

        self.hdr_texture = self.ctx.texture((width, height), 4, dtype='f2')
        self.hdr_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth_texture = self.ctx.depth_texture((width, height))
        self.hdr_fbo = self.ctx.framebuffer(
            color_attachments=[self.hdr_texture],
            depth_attachment=self.depth_texture,
        )

        # Resize bloom
        self.bloom.resize(width, height)

    def set_theme(self, theme: ColorTheme):
        """Update post-processing parameters from theme."""
        self.bloom_intensity = theme.bloom_intensity
        self.bloom_threshold = theme.bloom_threshold
        self.exposure = theme.exposure
        self.gamma = theme.gamma

    def begin_scene(self):
        """Begin rendering to HDR framebuffer."""
        if not self.enabled:
            return

        self.hdr_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def end_scene(self, target_fbo=None):
        """
        End scene rendering and apply post-processing.

        Args:
            target_fbo: Target framebuffer to render to (None = default framebuffer)
        """
        if not self.enabled:
            return

        # Apply bloom
        if self.bloom_enabled:
            self.bloom.extract_bright(
                self.hdr_texture,
                threshold=self.bloom_threshold,
                soft_threshold=0.5
            )
            self.bloom.blur_mip_chain()

        # Composite to target
        if target_fbo is not None:
            target_fbo.use()
        else:
            self.ctx.screen.use()

        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        # Bind textures
        self.hdr_texture.use(0)
        self.composite_program['u_scene'].value = 0

        # Bind bloom mip textures
        bloom_textures = self.bloom.get_all_mip_textures()
        for i, tex in enumerate(bloom_textures[:5]):
            tex.use(i + 1)
            self.composite_program[f'u_bloom{i}'].value = i + 1

        # Set uniforms
        self.composite_program['u_bloom_intensity'].value = self.bloom_intensity
        self.composite_program['u_exposure'].value = self.exposure
        self.composite_program['u_gamma'].value = self.gamma
        self.composite_program['u_bloom_enabled'].value = self.bloom_enabled

        # Render fullscreen quad
        self.composite_vao.render(moderngl.TRIANGLE_STRIP)

        # Re-enable states for subsequent rendering (e.g., ImGui)
        self.ctx.enable(moderngl.BLEND)

    def clear(self, color=(0.0, 0.0, 0.0, 1.0)):
        """Clear the HDR framebuffer."""
        if self.enabled:
            self.hdr_fbo.use()
            self.ctx.clear(*color, depth=1.0)

    def release(self):
        """Release all resources."""
        self.hdr_texture.release()
        self.depth_texture.release()
        self.hdr_fbo.release()
        self.bloom.release()
        self.quad_vbo.release()
        self.composite_vao.release()
        self.composite_program.release()
