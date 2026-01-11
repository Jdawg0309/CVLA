"""
Triangle drawing helpers.
"""

import numpy as np
import moderngl


def draw_triangles(self, vertices, normals, colors, vp, model_matrix=None,
                   light_pos=(20, 20, 20), view_pos=(0, 0, 20), use_lighting=True):
    """Draw triangles with normals and colors."""
    if not vertices or len(vertices) == 0:
        return

    vertices = np.array(vertices, dtype='f4').reshape(-1, 3)
    normals = np.array(normals, dtype='f4').reshape(-1, 3)
    colors = np.array(colors, dtype='f4').reshape(-1, 4)

    interleaved = np.zeros((vertices.shape[0], 10), dtype='f4')
    interleaved[:, :3] = vertices
    interleaved[:, 3:6] = normals
    interleaved[:, 6:10] = colors

    self.triangle_vbo.write(interleaved.tobytes())

    if model_matrix is None:
        model_matrix = np.eye(4, dtype='f4')

    self.triangle_program['mvp'].write(vp.astype('f4').tobytes())
    self.triangle_program['model'].write(model_matrix.astype('f4').tobytes())
    self.triangle_program['light_pos'].write(np.array(light_pos, dtype='f4').tobytes())
    self.triangle_program['view_pos'].write(np.array(view_pos, dtype='f4').tobytes())
    self.triangle_program['use_lighting'].value = bool(use_lighting)

    self.triangle_vao.render(moderngl.TRIANGLES, vertices=vertices.shape[0])
