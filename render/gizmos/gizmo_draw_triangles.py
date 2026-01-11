"""
Triangle drawing helpers.
"""

import numpy as np
import moderngl


def _ensure_float_array(buffer, components):
    """Ensure the buffer is a numpy array with the given component stride."""
    if isinstance(buffer, np.ndarray):
        array = buffer
    else:
        array = np.array(buffer, dtype='f4')
    return array.reshape(-1, components)


def draw_triangles(self, vertices, normals, colors, vp, model_matrix=None,
                   light_pos=(20, 20, 20), view_pos=(0, 0, 20), use_lighting=True):
    """Draw triangles with normals and colors."""
    vertices = _ensure_float_array(vertices, 3)
    if vertices.size == 0:
        return
    normals = _ensure_float_array(normals, 3)
    colors = _ensure_float_array(colors, 4)

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
