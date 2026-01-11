"""
Camera for CVLA.
"""

from render.cameras.camera_core import __init__, set_viewport, position, vp, _get_2d_up_vector
from render.cameras.camera_controls import (
    orbit, pan, zoom, set_view_preset, cubic_view_rotation, reset, focus_on_vector
)
from render.cameras.camera_projection import world_to_screen, screen_to_ray, get_view_matrix, get_projection_matrix


class Camera:
    __init__ = __init__
    set_viewport = set_viewport
    position = position
    vp = vp
    _get_2d_up_vector = _get_2d_up_vector
    orbit = orbit
    pan = pan
    zoom = zoom
    set_view_preset = set_view_preset
    cubic_view_rotation = cubic_view_rotation
    reset = reset
    focus_on_vector = focus_on_vector
    world_to_screen = world_to_screen
    screen_to_ray = screen_to_ray
    get_view_matrix = get_view_matrix
    get_projection_matrix = get_projection_matrix
