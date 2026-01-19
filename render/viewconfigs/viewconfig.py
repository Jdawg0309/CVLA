"""
View configuration for rendering.
"""

import copy
import numpy as np

from render.themes.color_themes import ColorTheme, get_theme, DEFAULT_THEME


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
    background_color=None,
    grid_color=None,
    axis_color_x=None,
    axis_color_y=None,
    axis_color_z=None,
    show_cube_faces=True,
    show_cube_corners=True,
    cube_face_opacity=0.03,
    cubic_grid_density=1.0,
    auto_rotate=False,
    rotation_speed=0.5,
    show_depth_cues=True,
    cubic_perspective=True,
    use_infinite_grid=True,
    theme=None,
):
    # Set theme first so colors can derive from it
    if theme is None:
        self._theme = get_theme(DEFAULT_THEME)
    elif isinstance(theme, str):
        self._theme = get_theme(theme)
    else:
        self._theme = theme

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
    self.use_infinite_grid = use_infinite_grid

    self.vector_scale = vector_scale

    self.fov = fov
    self.near_clip = near_clip
    self.far_clip = far_clip

    # Use theme colors as defaults, allow override
    self.background_color = background_color if background_color is not None else self._theme.background_color
    self.grid_color = grid_color if grid_color is not None else self._theme.grid_color_minor
    self.axis_color_x = axis_color_x if axis_color_x is not None else self._theme.axis_color_x
    self.axis_color_y = axis_color_y if axis_color_y is not None else self._theme.axis_color_y
    self.axis_color_z = axis_color_z if axis_color_z is not None else self._theme.axis_color_z

    # Cube face colors from theme with opacity override
    self.cube_face_colors = [
        (c[0], c[1], c[2], self.cube_face_opacity)
        for c in self._theme.cube_face_colors
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


def _setup_axis_mapping(self):
    """Setup axis mapping based on up axis."""
    if self.up_axis == "z":
        self.axis_map = {
            'x': 0, 'y': 1, 'z': 2,
            'right': 0, 'forward': 1, 'up': 2
        }
        self.axis_names = {'x': 'X', 'y': 'Y', 'z': 'Z'}

    elif self.up_axis == "y":
        self.axis_map = {
            'x': 0, 'y': 2, 'z': 1,
            'right': 0, 'forward': 2, 'up': 1
        }
        self.axis_names = {'x': 'X', 'y': 'Z', 'z': 'Y'}

    else:
        self.axis_map = {
            'x': 1, 'y': 2, 'z': 0,
            'right': 1, 'forward': 2, 'up': 0
        }
        self.axis_names = {'x': 'Y', 'y': 'Z', 'z': 'X'}


def axis_vectors(self):
    """Get world-space axis vectors based on up_axis."""
    if self.up_axis == "z":
        return {
            "x": np.array([1, 0, 0], dtype=np.float32),
            "y": np.array([0, 1, 0], dtype=np.float32),
            "z": np.array([0, 0, 1], dtype=np.float32)
        }
    elif self.up_axis == "y":
        return {
            "x": np.array([1, 0, 0], dtype=np.float32),
            "y": np.array([0, 0, 1], dtype=np.float32),
            "z": np.array([0, 1, 0], dtype=np.float32)
        }
    else:
        return {
            "x": np.array([0, 1, 0], dtype=np.float32),
            "y": np.array([0, 0, 1], dtype=np.float32),
            "z": np.array([1, 0, 0], dtype=np.float32)
        }


def axis_label_strings(self):
    """Get descriptive axis labels."""
    return {
        "x": self.axis_names['x'],
        "y": self.axis_names['y'],
        "z": self.axis_names['z']
    }


def _setup_cubic_view(self):
    """Setup cubic view specific parameters."""
    if self.grid_mode == "cube":
        try:
            base_minor = int(getattr(self, '_base_minor_tick', self.minor_tick))
        except Exception:
            base_minor = int(self.minor_tick)
        try:
            base_major = int(getattr(self, '_base_major_tick', self.major_tick))
        except Exception:
            base_major = int(self.major_tick)

        self.minor_tick = max(1, int(base_minor * self.cubic_grid_density))
        self.major_tick = max(1, int(base_major * self.cubic_grid_density))

        if self.show_depth_cues:
            self.background_color = self._theme.background_color_cube


@property
def theme(self):
    """Get the current color theme."""
    return self._theme


def set_theme(self, theme):
    """Set a new color theme and update colors."""
    if isinstance(theme, str):
        self._theme = get_theme(theme)
    else:
        self._theme = theme

    # Update colors from theme
    self.background_color = self._theme.background_color
    self.axis_color_x = self._theme.axis_color_x
    self.axis_color_y = self._theme.axis_color_y
    self.axis_color_z = self._theme.axis_color_z
    self.cube_face_colors = [
        (c[0], c[1], c[2], self.cube_face_opacity)
        for c in self._theme.cube_face_colors
    ]

    # Re-setup cubic view to apply theme cube background
    self._setup_cubic_view()


def get_grid_planes(self):
    """Return a list of grid planes to render based on grid_mode."""
    if self.grid_mode == "cube":
        return ["xy", "xz", "yz"]
    else:
        return [self.grid_plane]


def get_cube_corners(self):
    """Get the 8 corners of the visualization cube."""
    size = float(self.grid_size)
    return [
        [size, size, size],
        [size, size, -size],
        [size, -size, size],
        [size, -size, -size],
        [-size, size, size],
        [-size, size, -size],
        [-size, -size, size],
        [-size, -size, -size],
    ]


def get_cube_face_centers(self):
    """Get centers of cube faces for potential labeling."""
    size = float(self.grid_size)
    return {
        "xy+": [0, 0, size],
        "xy-": [0, 0, -size],
        "xz+": [0, size, 0],
        "xz-": [0, -size, 0],
        "yz+": [size, 0, 0],
        "yz-": [-size, 0, 0],
    }


def get_cubic_grid_settings(self):
    """Get settings optimized for cubic view."""
    return {
        "grid_density": self.cubic_grid_density,
        "show_faces": self.show_cube_faces,
        "show_corners": self.show_cube_corners,
        "depth_cues": self.show_depth_cues,
        "perspective": self.cubic_perspective,
        "face_opacity": self.cube_face_opacity,
    }


def grid_axes(self):
    """Return indices of axes that form the grid plane."""
    if self.grid_plane == "xy":
        return (self.axis_map['x'], self.axis_map['y'])
    elif self.grid_plane == "xz":
        return (self.axis_map['x'], self.axis_map['z'])
    elif self.grid_plane == "yz":
        return (self.axis_map['y'], self.axis_map['z'])


def get_grid_normal(self):
    """Get the normal vector of the grid plane."""
    if self.grid_plane == "xy":
        return np.array([0, 0, 1], dtype=np.float32)
    elif self.grid_plane == "xz":
        return np.array([0, 1, 0], dtype=np.float32)
    else:
        return np.array([1, 0, 0], dtype=np.float32)


def get_grid_basis(self):
    """Get basis vectors for the grid plane."""
    axes = self.grid_axes()

    basis = []
    for i in range(3):
        vec = np.zeros(3, dtype=np.float32)
        if i in axes:
            vec[i] = 1.0
        basis.append(vec)

    return basis


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
    theme = theme
    set_theme = set_theme
