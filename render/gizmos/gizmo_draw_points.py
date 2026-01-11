"""
Point drawing helpers.
"""

import numpy as np
import moderngl


def draw_points(self, vertices, colors, vp, size=8.0, depth=True):
    """Draw points with per-vertex colors."""
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

    self.point_vbo.write(interleaved.tobytes())

    self.point_program['mvp'].write(vp.astype('f4').tobytes())
    self.point_program['point_size'].value = float(size)

    self.point_vao.render(moderngl.POINTS, vertices=vertices.shape[0])
