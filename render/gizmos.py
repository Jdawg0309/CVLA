"""
Gizmos: immediate-mode debug drawing.
"""

from render.gizmo_programs import (
    _create_line_program,
    _create_triangle_program,
    _create_point_program,
    _create_volume_program,
)
from render.gizmo_buffers import _init_buffers
from render.gizmo_draw_lines import draw_lines
from render.gizmo_draw_triangles import draw_triangles
from render.gizmo_draw_points import draw_points
from render.gizmo_draw_volume import draw_volume
from render.gizmo_cubic_grid import draw_cubic_grid, draw_cube
from render.gizmo_vector_details import draw_vector_with_details, _draw_arrow_head, _draw_vector_components
from render.gizmo_vector_visuals import draw_vector_span, draw_parallelepiped, draw_basis_transform
from render.gizmo_planar_grid import draw_grid, draw_axes


class Gizmos:
    def __init__(self, ctx):
        self.ctx = ctx

        self.line_program = self._create_line_program()
        self.triangle_program = self._create_triangle_program()
        self.point_program = self._create_point_program()
        self.volume_program = self._create_volume_program()

        self.line_vao = None
        self.line_vbo = None
        self.triangle_vao = None
        self.triangle_vbo = None
        self.point_vao = None
        self.point_vbo = None
        self.volume_vao = None
        self.volume_vbo = None

        self._init_buffers()

    _create_line_program = _create_line_program
    _create_triangle_program = _create_triangle_program
    _create_point_program = _create_point_program
    _create_volume_program = _create_volume_program
    _init_buffers = _init_buffers
    draw_lines = draw_lines
    draw_triangles = draw_triangles
    draw_points = draw_points
    draw_volume = draw_volume
    draw_cubic_grid = draw_cubic_grid
    draw_cube = draw_cube
    draw_vector_with_details = draw_vector_with_details
    _draw_arrow_head = _draw_arrow_head
    _draw_vector_components = _draw_vector_components
    draw_vector_span = draw_vector_span
    draw_parallelepiped = draw_parallelepiped
    draw_basis_transform = draw_basis_transform
    draw_grid = draw_grid
    draw_axes = draw_axes
