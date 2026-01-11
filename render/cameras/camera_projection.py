"""
Camera projection helpers.
"""

import numpy as np
from pyrr import Matrix44


def world_to_screen(self, world_pos, width, height):
    """Convert world coordinates to screen coordinates."""
    vp = self.vp()
    p = np.array([*world_pos, 1.0], dtype=np.float32)
    clip = vp @ p

    if abs(clip[3]) < 1e-6:
        return None

    ndc = clip[:3] / clip[3]

    if ndc[2] < -1.0:
        return None

    x = (ndc[0] * 0.5 + 0.5) * width
    y = (1.0 - (ndc[1] * 0.5 + 0.5)) * height
    return (x, y, ndc[2])


def screen_to_ray(self, screen_x, screen_y, width, height):
    """Convert screen coordinates to a ray in world space."""
    x = (2.0 * screen_x) / width - 1.0
    y = 1.0 - (2.0 * screen_y) / height
    z = 1.0

    ray_clip = np.array([x, y, -1.0, 1.0], dtype=np.float32)

    vp = self.vp()
    inv_proj_view = np.linalg.inv(vp)
    ray_eye = inv_proj_view @ ray_clip
    ray_eye = ray_eye / ray_eye[3]
    ray_eye[2] = -1.0
    ray_eye[3] = 0.0

    ray_world = inv_proj_view @ ray_eye
    ray_world = ray_world[:3]
    ray_world = ray_world / np.linalg.norm(ray_world)

    return self.position(), ray_world


def get_view_matrix(self):
    """Get just the view matrix (without projection)."""
    pos = self.position()
    target = self.target_smooth
    return np.array(Matrix44.look_at(pos, target, self.up_vector), dtype=np.float32)


def get_projection_matrix(self):
    """Get just the projection matrix."""
    if self.mode_2d:
        s = float(self.ortho_scale)
        left = -s * self.aspect
        right = s * self.aspect
        bottom = -s
        top = s
        near, far = 0.1, 500.0
        return np.array(Matrix44.orthogonal_projection(left, right, bottom, top, near, far),
                      dtype=np.float32)
    else:
        fov = 45.0 if self.cubic_mode and self.view_preset == "cube" else 50.0
        near, far = 0.1, 500.0
        return np.array(Matrix44.perspective_projection(fov, self.aspect, near, far),
                      dtype=np.float32)
