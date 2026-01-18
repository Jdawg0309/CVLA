"""
Environment rendering helpers.
"""

import math


def _nice_step(value):
    """Pick a human-friendly step size (1, 2, 5 * 10^n)."""
    if value <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(value))
    f = value / exp
    if f <= 1.0:
        nice = 1.0
    elif f <= 2.0:
        nice = 2.0
    elif f <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return nice * exp


def _compute_grid_steps(size, density=1.0):
    """Compute major/minor/subminor steps based on semantic scale."""
    density = max(0.1, float(density))
    target = max(0.25, (size / 10.0) / density)
    minor = _nice_step(target)
    major = minor * 5.0
    subminor = minor / 2.0 if minor <= 1.0 else None
    if subminor is not None and subminor < 0.2:
        subminor = 0.2
    return minor, major, subminor


def _render_cubic_environment(self, vp, scene):
    """Render a beautiful 3D cubic environment."""
    if self.view.show_grid:
        radius = getattr(self.camera, "radius", 15.0)
        base_size = getattr(self.view, "grid_size", 15)
        size = int(max(6, min(base_size, radius * 0.7)))
        minor_step, major_step, subminor_step = _compute_grid_steps(
            size,
            getattr(self.view, "cubic_grid_density", 1.0)
        )

        # Draw primary grid with depth for proper intersection with vectors, then overlay thin lines without depth.
        self.gizmos.draw_cubic_grid(
            vp,
            size=size,
            major_step=major_step,
            minor_step=minor_step,
            subminor_step=subminor_step,
            color_major=(0.35, 0.37, 0.4, 0.42),
            color_minor=(0.22, 0.24, 0.28, 0.22),
            color_subminor=(0.2, 0.22, 0.26, 0.12),
            depth=True,
        )
        self.gizmos.draw_cubic_grid(
            vp,
            size=size,
            major_step=major_step,
            minor_step=minor_step,
            subminor_step=subminor_step,
            color_major=(0.35, 0.37, 0.4, 0.5),
            color_minor=(0.22, 0.24, 0.28, 0.28),
            color_subminor=(0.2, 0.22, 0.26, 0.16),
            depth=False,
        )

        if self.show_plane_visuals:
            self._render_cube_faces(vp)

        self._render_cube_corner_indicators(vp)

    if self.view.show_axes:
        axis_len = float(max(8.0, getattr(self.camera, "radius", 10.0) * 0.7))
        self._render_3d_axes_with_depths(vp, length=axis_len)
        self.gizmos.draw_points([[0, 0, 0]], [(0.9, 0.9, 0.95, 0.9)], vp, size=6.0, depth=False)


def _render_planar_environment(self, vp):
    """Render planar grid environment."""
    if self.view.show_grid:
        radius = getattr(self.camera, "radius", 15.0)
        base_size = getattr(self.view, "grid_size", 15)
        size = int(max(6, min(base_size, radius * 0.7)))
        minor_step, major_step, subminor_step = _compute_grid_steps(
            size,
            getattr(self.view, "cubic_grid_density", 1.0)
        )

        # Draw main grid with depth, then overlay with depth disabled so lines stay on top.
        self.gizmos.draw_grid(
            vp,
            size=size,
            step=minor_step,
            major_step=major_step,
            sub_step=subminor_step,
            plane=self.view.grid_plane,
            color_major=(0.35, 0.37, 0.4, 0.42),
            color_minor=(0.22, 0.24, 0.28, 0.22),
            color_subminor=(0.2, 0.22, 0.26, 0.12),
            depth=True,
        )
        self.gizmos.draw_grid(
            vp,
            size=size,
            step=minor_step,
            major_step=major_step,
            sub_step=subminor_step,
            plane=self.view.grid_plane,
            color_major=(0.35, 0.37, 0.4, 0.5),
            color_minor=(0.22, 0.24, 0.28, 0.28),
            color_subminor=(0.2, 0.22, 0.26, 0.16),
            depth=False,
        )

    if self.view.show_axes:
        axis_len = float(max(8.0, getattr(self.camera, "radius", 10.0) * 0.7))
        self.gizmos.draw_axes(vp, length=axis_len, thickness=3.0)
        self.gizmos.draw_points([[0, 0, 0]], [(0.9, 0.9, 0.95, 0.9)], vp, size=6.0, depth=False)
