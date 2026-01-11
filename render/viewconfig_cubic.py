"""
Cubic view helpers for view configuration.
"""


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
            self.background_color = (0.05, 0.06, 0.08, 1.0)


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
