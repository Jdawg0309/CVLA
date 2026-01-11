"""
Camera core setup and matrices.
"""

import os
import time
import numpy as np
from pyrr import Matrix44


DEBUG = os.getenv("CVLA_DEBUG") == "1"


def __init__(self):
    self.radius = 15.0
    self.theta = np.pi / 4
    self.phi = np.pi / 3

    self.target = np.zeros(3, dtype=np.float32)

    self.width = 1440
    self.height = 900
    self.aspect = self.width / self.height

    self.mode_2d = False
    self.view_preset = "cube"
    self.lock_up_vector = True

    self.ortho_scale = 10.0

    self.orbit_sensitivity = 0.005
    self.pan_sensitivity = 0.01
    self.zoom_sensitivity = 0.90

    self.up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    self.target_smooth = self.target.copy()
    self.smooth_factor = 0.1

    self.cubic_mode = True
    self.cubic_rotation_speed = 1.0
    self.last_cubic_rotation = 0.0

    if DEBUG:
        print("[Camera] initialized")


def set_viewport(self, width: int, height: int):
    self.width = max(1, int(width))
    self.height = max(1, int(height))
    self.aspect = self.width / self.height


def position(self) -> np.ndarray:
    """Returns camera position in world space."""
    if self.mode_2d:
        if self.view_preset == "xy":
            return np.array([0.0, 0.0, self.radius], dtype=np.float32)
        elif self.view_preset == "xz":
            return np.array([0.0, self.radius, 0.0], dtype=np.float32)
        elif self.view_preset == "yz":
            return np.array([self.radius, 0.0, 0.0], dtype=np.float32)
        else:
            return np.array([0.0, 0.0, self.radius], dtype=np.float32)

    self.phi = float(np.clip(self.phi, 0.10, np.pi - 0.10))

    x = self.radius * np.sin(self.phi) * np.cos(self.theta)
    y = self.radius * np.cos(self.phi)
    z = self.radius * np.sin(self.phi) * np.sin(self.theta)

    pos = np.array([x, y, z], dtype=np.float32)
    pos += self.target_smooth
    return pos


def vp(self) -> np.ndarray:
    """Returns ViewProjection matrix (float32 4x4)."""
    pos = self.position()
    target = self.target_smooth

    if self.mode_2d:
        up = _get_2d_up_vector(self)
        view = Matrix44.look_at(pos, target, up)

        s = float(self.ortho_scale)
        left = -s * self.aspect
        right = s * self.aspect
        bottom = -s
        top = s
        near, far = 0.1, 500.0
        proj = Matrix44.orthogonal_projection(left, right, bottom, top, near, far)
        return np.array(proj * view, dtype=np.float32)

    fov = 45.0 if self.cubic_mode and self.view_preset == "cube" else 50.0
    view = Matrix44.look_at(pos, target, self.up_vector)
    near, far = 0.1, 500.0
    proj = Matrix44.perspective_projection(fov, self.aspect, near, far)
    return np.array(proj * view, dtype=np.float32)


def _get_2d_up_vector(self):
    """Get the up vector for 2D mode based on view preset."""
    if self.view_preset == "xy":
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    elif self.view_preset == "xz":
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    elif self.view_preset == "yz":
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
