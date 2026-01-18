"""
Tensor face rendering helpers.
"""


def _render_tensor_faces(self, scene, vp):
    """Render triangle meshes for rank-2 tensors shaped (N, 3)."""
    faces = getattr(scene, "tensor_faces", None)
    if not faces:
        return

    for mesh in faces:
        vertices = mesh.get("vertices")
        normals = mesh.get("normals")
        colors = mesh.get("colors")
        if vertices is None or normals is None or colors is None:
            continue
        self.gizmos.draw_triangles(vertices, normals, colors, vp, use_lighting=False)
