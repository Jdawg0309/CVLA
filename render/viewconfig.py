"""
View configuration for rendering.
"""

from render.viewconfig_core import __init__, update, get_display_settings, clone, __str__
from render.viewconfig_axis import _setup_axis_mapping, axis_vectors, axis_label_strings
from render.viewconfig_cubic import _setup_cubic_view, get_grid_planes, get_cube_corners, get_cube_face_centers, get_cubic_grid_settings
from render.viewconfig_grid_basis import grid_axes, get_grid_normal, get_grid_basis


class ViewConfig:
    __init__ = __init__
    update = update
    get_display_settings = get_display_settings
    clone = clone
    __str__ = __str__
    _setup_axis_mapping = _setup_axis_mapping
    axis_vectors = axis_vectors
    axis_label_strings = axis_label_strings
    _setup_cubic_view = _setup_cubic_view
    get_grid_planes = get_grid_planes
    get_cube_corners = get_cube_corners
    get_cube_face_centers = get_cube_face_centers
    get_cubic_grid_settings = get_cubic_grid_settings
    grid_axes = grid_axes
    get_grid_normal = get_grid_normal
    get_grid_basis = get_grid_basis
