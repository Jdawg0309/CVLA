"""
Bloom post-processing effect.

Implements HDR bloom with multi-pass Gaussian blur and soft threshold extraction.
"""

import moderngl
import numpy as np


# Bloom extraction shader - extracts bright pixels
BLOOM_EXTRACT_VS = """
#version 330

in vec2 in_position;
in vec2 in_texcoord;

out vec2 v_texcoord;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_texcoord = in_texcoord;
}
"""

BLOOM_EXTRACT_FS = """
#version 330

uniform sampler2D u_texture;
uniform float u_threshold;
uniform float u_soft_threshold;

in vec2 v_texcoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(u_texture, v_texcoord);
    float brightness = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));

    // Soft threshold with knee
    float knee = u_threshold * u_soft_threshold;
    float soft = brightness - u_threshold + knee;
    soft = clamp(soft, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 0.00001);

    float contribution = max(soft, brightness - u_threshold);
    contribution /= max(brightness, 0.00001);

    fragColor = color * contribution;
}
"""

# Gaussian blur shader
BLUR_VS = """
#version 330

in vec2 in_position;
in vec2 in_texcoord;

out vec2 v_texcoord;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_texcoord = in_texcoord;
}
"""

BLUR_FS = """
#version 330

uniform sampler2D u_texture;
uniform vec2 u_direction;
uniform vec2 u_resolution;

in vec2 v_texcoord;
out vec4 fragColor;

// 9-tap Gaussian weights
const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 result = texture(u_texture, v_texcoord) * weights[0];

    for (int i = 1; i < 5; i++) {
        vec2 offset = u_direction * texel * float(i);
        result += texture(u_texture, v_texcoord + offset) * weights[i];
        result += texture(u_texture, v_texcoord - offset) * weights[i];
    }

    fragColor = result;
}
"""


class BloomEffect:
    """
    Multi-pass bloom effect with configurable threshold and intensity.
    """

    def __init__(self, ctx: moderngl.Context, width: int, height: int,
                 mip_levels: int = 5):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.mip_levels = mip_levels

        # Create shader programs
        self.extract_program = ctx.program(
            vertex_shader=BLOOM_EXTRACT_VS,
            fragment_shader=BLOOM_EXTRACT_FS,
        )
        self.blur_program = ctx.program(
            vertex_shader=BLUR_VS,
            fragment_shader=BLUR_FS,
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
        self.extract_vao = ctx.vertex_array(
            self.extract_program,
            [(self.quad_vbo, '2f 2f', 'in_position', 'in_texcoord')],
        )
        self.blur_vao = ctx.vertex_array(
            self.blur_program,
            [(self.quad_vbo, '2f 2f', 'in_position', 'in_texcoord')],
        )

        # Create mip chain framebuffers
        self.mip_textures = []
        self.mip_fbos = []
        self._create_mip_chain(width, height)

    def _create_mip_chain(self, width: int, height: int):
        """Create the mip chain for progressive downsampling and blur."""
        # Clean up existing resources
        for fbo in self.mip_fbos:
            fbo.release()
        for tex in self.mip_textures:
            tex.release()

        self.mip_textures = []
        self.mip_fbos = []
        self.width = width
        self.height = height

        w, h = width, height
        for i in range(self.mip_levels):
            w = max(1, w // 2)
            h = max(1, h // 2)

            # HDR texture for each mip level (RGBA16F)
            tex = self.ctx.texture((w, h), 4, dtype='f2')
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = False
            tex.repeat_y = False

            fbo = self.ctx.framebuffer(color_attachments=[tex])

            self.mip_textures.append(tex)
            self.mip_fbos.append(fbo)

    def resize(self, width: int, height: int):
        """Resize the bloom effect buffers."""
        if width != self.width or height != self.height:
            self._create_mip_chain(width, height)

    def extract_bright(self, source_texture, threshold: float = 0.8,
                       soft_threshold: float = 0.5):
        """Extract bright pixels from source texture into first mip level."""
        if not self.mip_fbos:
            return

        self.mip_fbos[0].use()
        self.ctx.viewport = (0, 0, self.mip_textures[0].width, self.mip_textures[0].height)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        source_texture.use(0)
        self.extract_program['u_texture'].value = 0
        self.extract_program['u_threshold'].value = threshold
        self.extract_program['u_soft_threshold'].value = soft_threshold

        self.extract_vao.render(moderngl.TRIANGLE_STRIP)

    def blur_mip_chain(self):
        """Apply Gaussian blur to each mip level, progressively downsampling."""
        for i in range(len(self.mip_textures)):
            tex = self.mip_textures[i]
            fbo = self.mip_fbos[i]
            w, h = tex.width, tex.height

            # Horizontal blur
            fbo.use()
            self.ctx.viewport = (0, 0, w, h)

            if i == 0:
                tex.use(0)
            else:
                self.mip_textures[i - 1].use(0)

            self.blur_program['u_texture'].value = 0
            self.blur_program['u_direction'].value = (1.0, 0.0)
            self.blur_program['u_resolution'].value = (float(w), float(h))
            self.blur_vao.render(moderngl.TRIANGLE_STRIP)

            # Vertical blur (in place)
            tex.use(0)
            self.blur_program['u_direction'].value = (0.0, 1.0)
            self.blur_vao.render(moderngl.TRIANGLE_STRIP)

    def get_bloom_texture(self):
        """Return the final bloom texture (lowest mip level with most blur)."""
        if self.mip_textures:
            return self.mip_textures[-1]
        return None

    def get_all_mip_textures(self):
        """Return all mip textures for multi-scale compositing."""
        return self.mip_textures

    def release(self):
        """Release all resources."""
        for fbo in self.mip_fbos:
            fbo.release()
        for tex in self.mip_textures:
            tex.release()
        self.quad_vbo.release()
        self.extract_vao.release()
        self.blur_vao.release()
        self.extract_program.release()
        self.blur_program.release()
