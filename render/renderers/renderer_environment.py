"""
Environment rendering helpers.
"""


def _render_cubic_environment(self, vp, scene):
    """Render a beautiful 3D cubic environment."""
    if self.view.show_grid:
        radius = getattr(self.camera, "radius", 15.0)
        base_size = getattr(self.view, "grid_size", 15)
        size = int(max(6, min(base_size, radius * 0.7)))

        major_step = max(1, int(getattr(self.view, 'major_tick', 5)))
        minor_step = max(1, int(getattr(self.view, 'minor_tick', 1)))

        # Draw primary grid with depth for proper intersection with vectors, then overlay thin lines without depth.
        self.gizmos.draw_cubic_grid(
            vp,
            size=size,
            major_step=major_step,
            minor_step=minor_step,
            color_major=(0.35, 0.37, 0.4, 0.35),
            color_minor=(0.22, 0.24, 0.28, 0.20),
            depth=True,
        )
        self.gizmos.draw_cubic_grid(
            vp,
            size=size,
            major_step=major_step,
            minor_step=minor_step,
            color_major=(0.35, 0.37, 0.4, 0.55),
            color_minor=(0.22, 0.24, 0.28, 0.28),
            depth=False,
        )

        if self.show_plane_visuals:
            self._render_cube_faces(vp)

        self._render_cube_corner_indicators(vp)

    if self.view.show_axes:
        axis_len = float(min(25.0, max(8.0, getattr(self.camera, "radius", 10.0) * 0.7)))
        self._render_3d_axes_with_depths(vp, length=axis_len)


def _render_planar_environment(self, vp):
    """Render planar grid environment."""
    if self.view.show_grid:
        radius = getattr(self.camera, "radius", 15.0)
        base_size = getattr(self.view, "grid_size", 15)
        size = int(max(6, min(base_size, radius * 0.7)))

        # Draw main grid with depth, then overlay with depth disabled so lines stay on top.
        self.gizmos.draw_grid(
            vp,
            size=size,
            step=self.view.minor_tick,
            plane=self.view.grid_plane,
            color_major=(0.35, 0.37, 0.4, 0.35),
            color_minor=(0.22, 0.24, 0.28, 0.20),
            depth=True,
        )
        self.gizmos.draw_grid(
            vp,
            size=size,
            step=self.view.minor_tick,
            plane=self.view.grid_plane,
            color_major=(0.35, 0.37, 0.4, 0.55),
            color_minor=(0.22, 0.24, 0.28, 0.28),
            depth=False,
        )

    if self.view.show_axes:
        axis_len = float(min(25.0, max(8.0, getattr(self.camera, "radius", 10.0) * 0.7)))
        self.gizmos.draw_axes(vp, length=axis_len, thickness=3.0)
