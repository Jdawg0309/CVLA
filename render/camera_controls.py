"""
Camera control helpers.
"""

import time
import numpy as np
from pyrr import Vector3


def orbit(self, dx: float, dy: float):
    """Orbit around target with cubic view enhancements."""
    if self.mode_2d:
        return

    sensitivity = self.orbit_sensitivity
    if self.cubic_mode and self.view_preset == "cube":
        sensitivity *= 1.5

    self.theta += float(dx) * sensitivity
    self.phi -= float(dy) * sensitivity

    self.phi = float(np.clip(self.phi, 0.10, np.pi - 0.10))

    if self.theta > 2 * np.pi:
        self.theta -= 2 * np.pi
    elif self.theta < -2 * np.pi:
        self.theta += 2 * np.pi

    if self.cubic_mode:
        self.last_cubic_rotation = time.time()


def pan(self, dx: float, dy: float):
    """Pan the camera (move target)."""
    if self.mode_2d:
        return

    pos = self.position()
    forward = Vector3(self.target) - Vector3(pos)
    forward.normalize()

    right = Vector3(forward).cross(Vector3(self.up_vector))
    right.normalize()

    up = Vector3(right).cross(Vector3(forward))
    up.normalize()

    pan_speed = self.radius * self.pan_sensitivity * 0.01

    self.target[0] -= float(right.x) * dx * pan_speed
    self.target[1] += float(up.y) * dy * pan_speed
    self.target[2] -= float(right.z) * dx * pan_speed

    self.target_smooth = self.target_smooth * (1 - self.smooth_factor) + self.target * self.smooth_factor


def zoom(self, scroll_y: float):
    """Zoom in/out with cubic view adjustments."""
    amt = float(scroll_y)

    zoom_sensitivity = self.zoom_sensitivity
    if self.cubic_mode and self.view_preset == "cube":
        zoom_sensitivity = 0.93

    if self.mode_2d:
        self.ortho_scale *= (zoom_sensitivity ** amt)
        self.ortho_scale = float(np.clip(self.ortho_scale, 1.0, 100.0))
    else:
        self.radius *= (zoom_sensitivity ** amt)
        self.radius = float(np.clip(self.radius, 1.0, 500.0))


def set_view_preset(self, preset: str):
    """Set the view preset with enhanced cubic view positioning."""
    self.view_preset = preset.lower()

    if preset == "xy":
        self.theta = np.pi / 4
        self.phi = np.pi / 2 - 0.01
        self.up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.cubic_mode = False
    elif preset == "xz":
        self.theta = 0
        self.phi = np.pi / 2
        self.up_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.cubic_mode = False
    elif preset == "yz":
        self.theta = np.pi / 2
        self.phi = np.pi / 2
        self.up_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.cubic_mode = False
    elif preset == "cube":
        self.theta = np.pi / 4 + np.pi / 8
        self.phi = np.pi / 4
        self.up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.cubic_mode = True
        self.mode_2d = False
    elif preset == "3d_free":
        self.theta = np.pi / 4
        self.phi = np.pi / 3
        self.up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.cubic_mode = True
        self.mode_2d = False


def cubic_view_rotation(self, auto_rotate=False, speed=0.5):
    """Perform automatic rotation for cubic view demonstration."""
    if auto_rotate and self.cubic_mode and self.view_preset == "cube":
        current_time = time.time()
        delta = current_time - self.last_cubic_rotation
        self.last_cubic_rotation = current_time

        self.theta += delta * speed * 0.5
        if self.theta > 2 * np.pi:
            self.theta -= 2 * np.pi


def reset(self):
    """Reset camera to default position."""
    self.radius = 15.0
    self.theta = np.pi / 4 + np.pi / 8
    self.phi = np.pi / 4
    self.ortho_scale = 10.0
    self.target = np.zeros(3, dtype=np.float32)
    self.target_smooth = self.target.copy()
    self.up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    self.mode_2d = False
    self.view_preset = "cube"
    self.cubic_mode = True


def focus_on_vector(self, vector_coords):
    """Focus camera on a specific vector with cubic view adjustments."""
    if vector_coords is not None:
        self.target = np.array(vector_coords, dtype=np.float32) * 0.5
        self.target_smooth = self.target.copy()

        vector_length = np.linalg.norm(vector_coords)
        self.radius = max(4.0, vector_length * 2.0)

        if self.cubic_mode:
            self.phi = np.pi / 4
            self.theta = np.pi / 4 + np.pi / 8
