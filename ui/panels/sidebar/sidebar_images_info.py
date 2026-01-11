"""
Sidebar image info section.
"""

import imgui


def _render_image_info_section(self):
    """Render image info section."""
    if self.current_image is None:
        return

    if self._section("Image as Matrix", "Matrix", default_open=True):
        img = self.current_image

        imgui.text_colored(f"{img.name}", 0.4, 0.8, 1.0, 1.0)
        imgui.text(f"Shape: {img.shape} = {img.height}x{img.width}")
        if img.is_rgb:
            imgui.text("Type: RGB (3 channels)")
        else:
            imgui.text("Type: Grayscale")

        imgui.spacing()

        stats = img.get_statistics()
        imgui.text(f"Mean: {stats['mean']:.3f}  Std: {stats['std']:.3f}")
        imgui.text(f"Min: {stats['min']:.3f}  Max: {stats['max']:.3f}")

        imgui.spacing()

        changed, self.show_image_matrix = imgui.checkbox("Show Matrix Values", self.show_image_matrix)

        if self.show_image_matrix:
            imgui.begin_child("##matrix_view", 0, 150, border=True)
            matrix = img.as_matrix()
            rows = min(8, matrix.shape[0])
            cols = min(8, matrix.shape[1])
            imgui.text_disabled("Matrix preview (8x8 max):")
            for i in range(rows):
                row_str = " ".join(f"{matrix[i, j]:.2f}" for j in range(cols))
                imgui.text(row_str)
            if matrix.shape[0] > 8 or matrix.shape[1] > 8:
                imgui.text_disabled("...")
            imgui.end_child()

        self._end_section()
