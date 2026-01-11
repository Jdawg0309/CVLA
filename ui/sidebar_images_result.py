"""
Sidebar image result section.
"""

import imgui


def _render_image_result_section(self, scene):
    """Render processed image section."""
    if self.processed_image is None:
        return

    if self._section("Result", "Result", default_open=True):
        result = self.processed_image

        imgui.text_colored(f"Output: {result.name}", 0.4, 1.0, 0.4, 1.0)
        imgui.text(f"Shape: {result.shape}")

        if result.history:
            imgui.text("Operations applied:")
            for op, param in result.history:
                imgui.bullet_text(f"{op}")

        imgui.spacing()

        if imgui.button("Use as Input", width=-1):
            self.current_image = self.processed_image
            self.processed_image = None

        if result.height <= 8 and result.width <= 8:
            imgui.spacing()
            if imgui.button("Add Matrix Rows as Vectors", width=-1):
                self._add_image_as_vectors(scene, result)

        self._end_section()
