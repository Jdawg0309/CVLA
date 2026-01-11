"""
Volume drawing helpers.
"""

import numpy as np
import moderngl


def draw_volume(self, vertices, colors, vp, opacity=0.3, depth=True):
    """Draw volume visualization (transparent cube)."""
    if not vertices or len(vertices) == 0:
        return

    if depth:
        self.ctx.enable(moderngl.DEPTH_TEST)
    else:
        self.ctx.disable(moderngl.DEPTH_TEST)

    vertices = np.array(vertices, dtype='f4').reshape(-1, 3)
    colors = np.array(colors, dtype='f4').reshape(-1, 4)

    interleaved = np.zeros((vertices.shape[0], 7), dtype='f4')
    interleaved[:, :3] = vertices
    interleaved[:, 3:7] = colors

    self.volume_vbo.write(interleaved.tobytes())
    self.volume_program['mvp'].write(vp.astype('f4').tobytes())
    self.volume_program['opacity'].value = float(opacity)

    self.volume_vao.render(moderngl.TRIANGLES, vertices=vertices.shape[0])
