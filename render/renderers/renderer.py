"""
Main renderer for CVLA.
"""

import moderngl
from render.viewconfigs.viewconfig import ViewConfig
from render.gizmos.gizmos import Gizmos

from render.renderers.renderer_environment import _render_cubic_environment, _render_planar_environment
from render.renderers.renderer_cubic_faces import _render_cube_faces, _render_cube_corner_indicators
from render.renderers.renderer_axes import _render_3d_axes_with_depths, _draw_axis_cones
from render.renderers.renderer_linear_algebra import (
    _render_linear_algebra_visuals,
    _render_matrix_3d_plot,
)
from render.renderers.renderer_vectors import _render_vectors_with_enhancements, _render_vector_projections, _render_selection_highlight
from render.renderers.renderer_image import draw_image_plane, _image_color, _resolve_image_matrix


class Renderer:
    def __init__(self, ctx, camera, view=None):
        self.ctx = ctx
        self.camera = camera
        self.view = view or ViewConfig()

        self.gizmos = Gizmos(ctx)

        self.vector_scale = 3.0
        self.show_vector_labels = True
        self.show_plane_visuals = True
        self.show_vector_components = True
        self.show_vector_spans = False

        self._image_plane_cache = {"key": None, "batches": []}

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        # Always draw both sides of planar grids/images so they remain visible when orbiting.
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self._vp_cache = None
        self._vp_cache_dirty = True

        self.cube_face_colors = [
            (0.3, 0.3, 0.8, 0.05),
            (0.8, 0.3, 0.3, 0.05),
            (0.3, 0.8, 0.3, 0.05),
            (0.8, 0.8, 0.3, 0.05),
            (0.8, 0.3, 0.8, 0.05),
            (0.3, 0.8, 0.8, 0.05),
        ]

    def _get_view_projection(self):
        """Get cached view-projection matrix."""
        if self._vp_cache_dirty or self._vp_cache is None:
            self._vp_cache = self.camera.vp()
            self._vp_cache_dirty = False
        return self._vp_cache

    def update_view(self, view_config):
        """Update view configuration."""
        self.view = view_config
        self._vp_cache_dirty = True

        if hasattr(view_config, 'vector_scale'):
            self.vector_scale = view_config.vector_scale

        if hasattr(view_config, 'show_plane_visuals'):
            self.show_plane_visuals = view_config.show_plane_visuals

        try:
            if hasattr(view_config, 'cube_face_colors'):
                self.cube_face_colors = [tuple(c) for c in view_config.cube_face_colors]
        except Exception:
            pass

    def render(self, scene, image_data=None, show_image_on_grid=False, image_render_scale=1.0,
               image_color_mode="grayscale", image_color_source=None,
               image_render_mode="plane", show_image_grid_overlay=False):
        """Main rendering method."""
        self._clear_with_gradient()

        vp = self._get_view_projection()

        if self.view.grid_mode == "cube":
            self._render_cubic_environment(vp, scene)
        else:
            self._render_planar_environment(vp)

        if image_data is not None and show_image_on_grid:
            self.draw_image_plane(
                image_data,
                vp,
                scale=image_render_scale,
                color_mode=image_color_mode,
                color_source=image_color_source,
                render_mode=image_render_mode,
            )
            if show_image_grid_overlay:
                try:
                    matrix, _ = _resolve_image_matrix(image_data)
                    height, width = matrix.shape[:2]
                    grid_size = int(max(1.0, max(width, height) * image_render_scale * 0.5))
                    self.gizmos.draw_grid(
                        vp,
                        size=grid_size,
                        step=1,
                        plane="xy",
                        color_major=(0.35, 0.35, 0.4, 0.6),
                        color_minor=(0.22, 0.22, 0.25, 0.4),
                    )
                except Exception:
                    pass

        self._render_linear_algebra_visuals(scene, vp)
        self._render_vectors_with_enhancements(scene, vp)

        if scene.selected_object and scene.selection_type == 'vector':
            self._render_selection_highlight(scene.selected_object, vp)

    def _clear_with_gradient(self):
        """Clear with a subtle gradient background."""
        if self.view.grid_mode == "cube":
            self.ctx.clear(
                color=(0.05, 0.06, 0.08, 1.0),
                depth=1.0
            )
        else:
            self.ctx.clear(
                color=(0.08, 0.08, 0.10, 1.0),
                depth=1.0
            )

    draw_image_plane = draw_image_plane
    _image_color = _image_color
    _render_cubic_environment = _render_cubic_environment
    _render_planar_environment = _render_planar_environment
    _render_cube_faces = _render_cube_faces
    _render_cube_corner_indicators = _render_cube_corner_indicators
    _render_3d_axes_with_depths = _render_3d_axes_with_depths
    _draw_axis_cones = _draw_axis_cones
    _render_linear_algebra_visuals = _render_linear_algebra_visuals
    _render_matrix_3d_plot = _render_matrix_3d_plot
    _render_matrix_3d_plot = _render_matrix_3d_plot
    _render_vectors_with_enhancements = _render_vectors_with_enhancements
    _render_vector_projections = _render_vector_projections
    _render_selection_highlight = _render_selection_highlight
