"""
Line drawing helpers.
"""

import numpy as np
import moderngl


def draw_lines(self, vertices, colors, vp, width=2.0, depth=True):
    """Draw line segments with per-vertex colors."""
    if not vertices or len(vertices) == 0:
        return

    if depth:
        self.ctx.enable(moderngl.DEPTH_TEST)
    else:
        self.ctx.disable(moderngl.DEPTH_TEST)

    self.ctx.line_width = float(width)

    vertices = np.array(vertices, dtype='f4').reshape(-1, 3)
    colors = np.array(colors, dtype='f4').reshape(-1, 4)

    interleaved = np.zeros((vertices.shape[0], 7), dtype='f4')
    interleaved[:, :3] = vertices
    interleaved[:, 3:7] = colors

    self.line_vbo.write(interleaved.tobytes())
    self.line_program['mvp'].write(vp.astype('f4').tobytes())

    self.line_vao.render(moderngl.LINES, vertices=vertices.shape[0])
