"""
Sidebar image transform section.
"""

import imgui
from ui.sidebar_vision import AffineTransform, apply_affine_transform


def _render_image_transform_section(self):
    """Render affine transform section."""
    if self.current_image is None:
        return

    if self._section("Affine Transforms", "Transform"):
        imgui.text_colored("Linear transforms on coordinates", 0.4, 0.8, 0.4, 1.0)

        imgui.spacing()

        imgui.push_item_width(150)
        changed, self.transform_rotation = imgui.slider_float(
            "Rotation", self.transform_rotation, -180, 180, "%.1f deg"
        )
        imgui.pop_item_width()

        imgui.push_item_width(150)
        changed, self.transform_scale = imgui.slider_float(
            "Scale", self.transform_scale, 0.5, 2.0, "%.2fx"
        )
        imgui.pop_item_width()

        imgui.spacing()

        if imgui.button("Apply Transform", width=-1):
            h, w = self.current_image.height, self.current_image.width
            center = (w / 2, h / 2)
            transform = AffineTransform()
            transform.rotate(self.transform_rotation, center)
            transform.scale(self.transform_scale, center=center)
            self.processed_image = apply_affine_transform(
                self.current_image, transform
            )

        imgui.spacing()

        if imgui.button("Flip Horizontal", width=-1):
            h, w = self.current_image.height, self.current_image.width
            transform = AffineTransform()
            transform.flip_horizontal(w)
            self.processed_image = apply_affine_transform(
                self.current_image, transform
            )

        self._end_section()
