"""
Environment rendering helpers.
"""


def _render_cubic_environment(self, vp, scene):
    """Render a beautiful 3D cubic environment."""
    if self.view.show_grid:
        major_step = max(1, int(getattr(self.view, 'major_tick', 5)))
        minor_step = max(1, int(getattr(self.view, 'minor_tick', 1)))

        self.gizmos.draw_cubic_grid(
            vp,
            size=self.view.grid_size,
            major_step=major_step,
            minor_step=minor_step,
            color_major=(0.35, 0.37, 0.4, 0.9),
            color_minor=(0.2, 0.22, 0.25, 0.6)
        )

        if self.show_plane_visuals:
            self._render_cube_faces(vp)

        self._render_cube_corner_indicators(vp)

    if self.view.show_axes:
        self._render_3d_axes_with_depths(vp)


def _render_planar_environment(self, vp):
    """Render planar grid environment."""
    if self.view.show_grid:
        self.gizmos.draw_grid(
            vp,
            size=self.view.grid_size,
            step=self.view.minor_tick,
            plane=self.view.grid_plane
        )

    if self.view.show_axes:
        self.gizmos.draw_axes(vp, length=6.0, thickness=3.0)
