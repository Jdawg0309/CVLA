"""
Buffer initialization helpers.
"""


def _init_buffers(self):
    """Initialize vertex buffers."""
    self.line_vbo = self.ctx.buffer(reserve=2 * 1024 * 1024)
    self.line_vao = self.ctx.vertex_array(
        self.line_program,
        [
            (self.line_vbo, '3f 4f', 'in_position', 'in_color')
        ]
    )

    self.triangle_vbo = self.ctx.buffer(reserve=2 * 1024 * 1024)
    self.triangle_vao = self.ctx.vertex_array(
        self.triangle_program,
        [
            (self.triangle_vbo, '3f 3f 4f', 'in_position', 'in_normal', 'in_color')
        ]
    )

    self.point_vbo = self.ctx.buffer(reserve=512 * 1024)
    self.point_vao = self.ctx.vertex_array(
        self.point_program,
        [(self.point_vbo, '3f 4f', 'in_position', 'in_color')]
    )

    self.volume_vbo = self.ctx.buffer(reserve=1024 * 1024)
    self.volume_vao = self.ctx.vertex_array(
        self.volume_program,
        [(self.volume_vbo, '3f 4f', 'in_position', 'in_color')]
    )
