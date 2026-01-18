"""
Core view configuration setup.
"""

import copy
import numpy as np


def __init__(
    self,
    up_axis="z",
    grid_mode="cube",
    grid_plane="xy",
    grid_size=20,
    major_tick=5,
    minor_tick=1,
    label_density=1,
    show_grid=True,
    show_axes=True,
    show_labels=True,
    show_plane_visuals=True,
    vector_scale=3.0,
    auto_scale_vectors=True,
    coordinate_system="right_handed",
    view_mode="perspective",
    fov=50.0,
    near_clip=0.1,
    far_clip=500.0,
    background_color=(0.08, 0.08, 0.10, 1.0),
    grid_color=(0.15, 0.16, 0.18, 0.35),
    axis_color_x=(1.0, 0.3, 0.3, 1.0),
    axis_color_y=(0.3, 1.0, 0.3, 1.0),
    axis_color_z=(0.3, 0.5, 1.0, 1.0),
    show_cube_faces=True,
    show_cube_corners=True,
    cube_face_opacity=0.03,
    cubic_grid_density=1.0,
    auto_rotate=False,
    rotation_speed=0.5,
    show_depth_cues=True,
    cubic_perspective=True,
):
    self.up_axis = up_axis.lower()
    self.grid_mode = grid_mode.lower()
    self.grid_plane = grid_plane.lower()
    self.coordinate_system = coordinate_system.lower()
    self.view_mode = view_mode.lower()

    self.grid_size = grid_size
    self.major_tick = major_tick
    self.minor_tick = minor_tick
    self._base_major_tick = int(self.major_tick)
    self._base_minor_tick = int(self.minor_tick)
    self.label_density = label_density

    self.show_grid = show_grid
    self.show_axes = show_axes
    self.show_labels = show_labels
    self.show_plane_visuals = show_plane_visuals

    self.show_cube_faces = show_cube_faces
    self.show_cube_corners = show_cube_corners
    self.cube_face_opacity = cube_face_opacity
    self.cubic_grid_density = cubic_grid_density
    self.auto_rotate = auto_rotate
    self.rotation_speed = rotation_speed
    self.show_depth_cues = show_depth_cues
    self.cubic_perspective = cubic_perspective

    self.vector_scale = vector_scale

    self.fov = fov
    self.near_clip = near_clip
    self.far_clip = far_clip

    self.background_color = background_color
    self.grid_color = grid_color
    self.axis_color_x = axis_color_x
    self.axis_color_y = axis_color_y
    self.axis_color_z = axis_color_z

    self.cube_face_colors = [
        (0.3, 0.3, 0.8, self.cube_face_opacity),
        (0.8, 0.3, 0.3, self.cube_face_opacity),
        (0.3, 0.8, 0.3, self.cube_face_opacity),
        (0.8, 0.8, 0.3, self.cube_face_opacity),
        (0.8, 0.3, 0.8, self.cube_face_opacity),
        (0.3, 0.8, 0.8, self.cube_face_opacity),
    ]

    assert self.up_axis in ("x", "y", "z")
    assert self.grid_mode in ("plane", "cube")
    assert self.grid_plane in ("xy", "xz", "yz")
    assert self.coordinate_system in ("right_handed", "left_handed")
    assert self.view_mode in ("perspective", "orthographic")

    self._setup_axis_mapping()
    self._setup_cubic_view()


def update(self, **kwargs):
    """Update multiple settings at once."""
    for key, value in kwargs.items():
        if hasattr(self, key):
            setattr(self, key, value)
            if key == 'major_tick':
                try:
                    self._base_major_tick = int(value)
                except Exception:
                    pass
            if key == 'minor_tick':
                try:
                    self._base_minor_tick = int(value)
                except Exception:
                    pass

            if key in ['up_axis', 'grid_mode', 'cubic_grid_density', 'cube_face_opacity']:
                self._setup_axis_mapping()
                self._setup_cubic_view()


def get_display_settings(self):
    """Get all display settings as a dict."""
    settings = {
        'up_axis': self.up_axis,
        'grid_mode': self.grid_mode,
        'grid_plane': self.grid_plane,
        'grid_size': self.grid_size,
        'show_grid': self.show_grid,
        'show_axes': self.show_axes,
        'show_labels': self.show_labels,
        'vector_scale': self.vector_scale,
        'view_mode': self.view_mode,
    }

    if self.grid_mode == "cube":
        settings.update({
            'show_cube_faces': self.show_cube_faces,
            'show_cube_corners': self.show_cube_corners,
            'cubic_grid_density': self.cubic_grid_density,
            'auto_rotate': self.auto_rotate,
            'rotation_speed': self.rotation_speed,
        })

    return settings


def clone(self):
    """Create a copy of this ViewConfig."""
    return copy.deepcopy(self)


def __str__(self):
    if self.grid_mode == "cube":
        return f"ViewConfig(up={self.up_axis}, mode=CUBE, size={self.grid_size})"
    else:
        return f"ViewConfig(up={self.up_axis}, grid={self.grid_plane})"
