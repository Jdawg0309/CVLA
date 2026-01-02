"""
Enhanced ViewConfig with comprehensive cubic view settings
"""

import numpy as np


class ViewConfig:
    def __init__(
        self,
        up_axis="z",
        grid_mode="cube",          # "plane" or "cube"
        grid_plane="xy",           # if grid_mode is "plane"
        grid_size=20,
        major_tick=5,
        minor_tick=1,
        label_density=1,           # label every n major ticks
        show_grid=True,
        show_axes=True,
        show_labels=True,
        show_plane_visuals=True,
        vector_scale=3.0,          # Visual scaling for vectors
        auto_scale_vectors=True,   # Automatically scale vectors to scene bounds
        coordinate_system="right_handed",  # "right_handed" or "left_handed"
        view_mode="perspective",   # "perspective" or "orthographic"
        fov=50.0,                  # Field of view for perspective
        near_clip=0.1,
        far_clip=500.0,
        background_color=(0.08, 0.08, 0.10, 1.0),
        grid_color=(0.15, 0.16, 0.18, 0.7),
        axis_color_x=(1.0, 0.3, 0.3, 1.0),
        axis_color_y=(0.3, 1.0, 0.3, 1.0),
        axis_color_z=(0.3, 0.5, 1.0, 1.0),
        
        # Cubic view specific settings
        show_cube_faces=True,      # Show translucent cube faces
        show_cube_corners=True,    # Show indicators at cube corners
        cube_face_opacity=0.05,    # Opacity of cube faces
        cubic_grid_density=1.0,    # Density multiplier for cubic grid
        show_depth_cues=True,      # Show depth cues in cubic view
        cubic_perspective=True,    # Use perspective in cubic view
    ):
        # View orientation
        self.up_axis = up_axis.lower()
        self.grid_mode = grid_mode.lower()
        self.grid_plane = grid_plane.lower()
        self.coordinate_system = coordinate_system.lower()
        self.view_mode = view_mode.lower()
        
        # Grid settings
        self.grid_size = grid_size
        self.major_tick = major_tick
        self.minor_tick = minor_tick
        self.label_density = label_density
        
        # Display toggles
        self.show_grid = show_grid
        self.show_axes = show_axes
        self.show_labels = show_labels
        self.show_plane_visuals = show_plane_visuals
        
        # Cubic view settings
        self.show_cube_faces = show_cube_faces
        self.show_cube_corners = show_cube_corners
        self.cube_face_opacity = cube_face_opacity
        self.cubic_grid_density = cubic_grid_density
        self.show_depth_cues = show_depth_cues
        self.cubic_perspective = cubic_perspective
        
        # Visual scaling
        self.vector_scale = vector_scale
        
        # Camera/rendering settings
        self.fov = fov
        self.near_clip = near_clip
        self.far_clip = far_clip
        
        # Colors
        self.background_color = background_color
        self.grid_color = grid_color
        self.axis_color_x = axis_color_x
        self.axis_color_y = axis_color_y
        self.axis_color_z = axis_color_z
        
        # Cubic view colors
        self.cube_face_colors = [
            (0.3, 0.3, 0.8, self.cube_face_opacity),   # XY+ face
            (0.8, 0.3, 0.3, self.cube_face_opacity),   # XZ+ face  
            (0.3, 0.8, 0.3, self.cube_face_opacity),   # YZ+ face
            (0.8, 0.8, 0.3, self.cube_face_opacity),   # XY- face
            (0.8, 0.3, 0.8, self.cube_face_opacity),   # XZ- face
            (0.3, 0.8, 0.8, self.cube_face_opacity),   # YZ- face
        ]
        
        # Validation
        assert self.up_axis in ("x", "y", "z")
        assert self.grid_mode in ("plane", "cube")
        assert self.grid_plane in ("xy", "xz", "yz")
        assert self.coordinate_system in ("right_handed", "left_handed")
        assert self.view_mode in ("perspective", "orthographic")
        
        # Axis mapping based on up axis
        self._setup_axis_mapping()
        
        # Cubic view settings
        self._setup_cubic_view()

    def _setup_axis_mapping(self):
        """Setup axis mapping based on up axis."""
        if self.up_axis == "z":
            # Standard: Z up, X right, Y forward
            self.axis_map = {
                'x': 0, 'y': 1, 'z': 2,
                'right': 0, 'forward': 1, 'up': 2
            }
            self.axis_names = {'x': 'X', 'y': 'Y', 'z': 'Z'}
            
        elif self.up_axis == "y":
            # Y up (common in many 3D apps): Y up, X right, Z forward
            self.axis_map = {
                'x': 0, 'y': 2, 'z': 1,  # Y and Z swapped
                'right': 0, 'forward': 2, 'up': 1
            }
            self.axis_names = {'x': 'X', 'y': 'Z', 'z': 'Y'}
            
        else:  # "x"
            # X up (uncommon): X up, Y right, Z forward
            self.axis_map = {
                'x': 1, 'y': 2, 'z': 0,  # Rotated
                'right': 1, 'forward': 2, 'up': 0
            }
            self.axis_names = {'x': 'Y', 'y': 'Z', 'z': 'X'}

    def _setup_cubic_view(self):
        """Setup cubic view specific parameters."""
        if self.grid_mode == "cube":
            # Adjust grid density for better 3D visualization
            self.minor_tick = max(1, int(self.minor_tick * self.cubic_grid_density))
            self.major_tick = max(1, int(self.major_tick * self.cubic_grid_density))
            
            # Adjust background for better depth perception
            if self.show_depth_cues:
                self.background_color = (0.05, 0.06, 0.08, 1.0)

    def update(self, **kwargs):
        """Update multiple settings at once."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
                # Re-setup if needed
                if key in ['up_axis', 'grid_mode', 'cubic_grid_density', 'cube_face_opacity']:
                    self._setup_axis_mapping()
                    self._setup_cubic_view()

    # --------------------------------------------------------
    # Axis direction vectors
    # --------------------------------------------------------
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
                "y": np.array([0, 0, 1], dtype=np.float32),  # Z becomes Y
                "z": np.array([0, 1, 0], dtype=np.float32)   # Y becomes Z
            }
        else:  # x up
            return {
                "x": np.array([0, 1, 0], dtype=np.float32),  # Y becomes X
                "y": np.array([0, 0, 1], dtype=np.float32),  # Z becomes Y
                "z": np.array([1, 0, 0], dtype=np.float32)   # X becomes Z
            }

    def axis_label_strings(self):
        """Get descriptive axis labels."""
        return {
            "x": self.axis_names['x'],
            "y": self.axis_names['y'],
            "z": self.axis_names['z']
        }

    # --------------------------------------------------------
    # Cubic view methods
    # --------------------------------------------------------
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
            [ size,  size,  size],  # +++
            [ size,  size, -size],  # ++-
            [ size, -size,  size],  # +-+
            [ size, -size, -size],  # +--
            [-size,  size,  size],  # -++
            [-size,  size, -size],  # -+-
            [-size, -size,  size],  # --+
            [-size, -size, -size],  # ---
        ]

    def get_cube_face_centers(self):
        """Get centers of cube faces for potential labeling."""
        size = float(self.grid_size)
        return {
            "xy+": [0, 0, size],    # XY positive face
            "xy-": [0, 0, -size],   # XY negative face
            "xz+": [0, size, 0],    # XZ positive face
            "xz-": [0, -size, 0],   # XZ negative face
            "yz+": [size, 0, 0],    # YZ positive face
            "yz-": [-size, 0, 0],   # YZ negative face
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

    # --------------------------------------------------------
    # Grid basis vectors
    # --------------------------------------------------------
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
        else:  # yz
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

    # --------------------------------------------------------
    # Utility methods
    # --------------------------------------------------------
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
        
        # Add cubic view settings if applicable
        if self.grid_mode == "cube":
            settings.update({
                'show_cube_faces': self.show_cube_faces,
                'show_cube_corners': self.show_cube_corners,
                'cubic_grid_density': self.cubic_grid_density,
            })
        
        return settings

    def clone(self):
        """Create a copy of this ViewConfig."""
        import copy
        return copy.deepcopy(self)

    def __str__(self):
        if self.grid_mode == "cube":
            return f"ViewConfig(up={self.up_axis}, mode=CUBE, size={self.grid_size})"
        else:
            return f"ViewConfig(up={self.up_axis}, grid={self.grid_plane})"