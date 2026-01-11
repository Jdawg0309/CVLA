"""
Vector rendering helpers.
"""

import numpy as np
import time


def _render_vectors_with_enhancements(self, scene, vp):
    """Render all vectors with enhanced visualizations."""
    try:
        if len(scene.vectors) > 0:
            visible_vectors = [v for v in scene.vectors if v.visible]
            if visible_vectors:
                max_mag = max([np.linalg.norm(v.coords) for v in visible_vectors] + [1.0])
                desired = max(1.0, self.camera.radius * 0.2)
                scale_factor = desired / max_mag
                scale_factor = float(np.clip(scale_factor, 0.3, 5.0))
                self.vector_scale = float(self.view.vector_scale) * scale_factor
            else:
                self.vector_scale = float(self.view.vector_scale)
    except Exception:
        self.vector_scale = float(self.view.vector_scale)

    for vector in scene.vectors:
        if vector.visible:
            is_selected = (vector is scene.selected_object and
                           scene.selection_type == 'vector')

            self.gizmos.draw_vector_with_details(
                vp, vector, is_selected, self.vector_scale,
                show_components=self.show_vector_components,
                show_span=False
            )

            if self.camera.mode_2d:
                self._render_vector_projections(vector, vp)


def _render_vector_projections(self, vector, vp):
    """Render vector projection lines in 2D mode."""
    tip = vector.coords * self.vector_scale

    if self.camera.view_preset == "xy":
        proj_tip = np.array([tip[0], tip[1], 0])
        proj_color = (vector.color[0], vector.color[1], vector.color[2], 0.3)

        line = [[tip[0], tip[1], tip[2]], [proj_tip[0], proj_tip[1], proj_tip[2]]]
        self.gizmos.draw_lines(line, [proj_color, proj_color], vp, width=1.0)
        self.gizmos.draw_points([proj_tip], [proj_color], vp, size=4.0)

    elif self.camera.view_preset == "xz":
        proj_tip = np.array([tip[0], 0, tip[2]])
        proj_color = (vector.color[0], vector.color[1], vector.color[2], 0.3)

        line = [[tip[0], tip[1], tip[2]], [proj_tip[0], proj_tip[1], proj_tip[2]]]
        self.gizmos.draw_lines(line, [proj_color, proj_color], vp, width=1.0)
        self.gizmos.draw_points([proj_tip], [proj_color], vp, size=4.0)

    elif self.camera.view_preset == "yz":
        proj_tip = np.array([0, tip[1], tip[2]])
        proj_color = (vector.color[0], vector.color[1], vector.color[2], 0.3)

        line = [[tip[0], tip[1], tip[2]], [proj_tip[0], proj_tip[1], proj_tip[2]]]
        self.gizmos.draw_lines(line, [proj_color, proj_color], vp, width=1.0)
        self.gizmos.draw_points([proj_tip], [proj_color], vp, size=4.0)


def _render_selection_highlight(self, vector, vp):
    """Render special highlight for selected vector."""
    tip = vector.coords * self.vector_scale

    pulse = (np.sin(time.time() * 5) * 0.2 + 0.8)

    circle_verts = []
    circle_colors = []
    segments = 24
    radius = 0.4 * pulse

    for i in range(segments):
        angle1 = 2 * np.pi * i / segments
        angle2 = 2 * np.pi * (i + 1) / segments

        p1 = tip + radius * np.array([np.cos(angle1), np.sin(angle1), 0])
        p2 = tip + radius * np.array([np.cos(angle2), np.sin(angle2), 0])

        circle_verts.extend(p1.tolist())
        circle_verts.extend(p2.tolist())

        pulse_color = (1.0, 1.0, 0.2, 0.7 * pulse)
        circle_colors.extend([pulse_color, pulse_color])

    self.gizmos.draw_lines(circle_verts, circle_colors, vp, width=2.0, depth=False)

    origin_line = [[0, 0, 0], tip.tolist()]
    highlight_color = (1.0, 1.0, 0.2, 0.8)
    self.gizmos.draw_lines(origin_line, [highlight_color, highlight_color], vp, width=2.0, depth=False)

    sphere_color = (1.0, 1.0, 0.2, 0.9)
    self.gizmos.draw_points([tip.tolist()], [sphere_color], vp, size=14.0)
