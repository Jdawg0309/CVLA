"""
Sidebar image source section.
"""

import imgui
from ui.panels.sidebar.sidebar_vision import create_sample_image, load_image


def _render_image_source_section(self):
    """Render image source section."""
    if self._section("Image Source", "Load"):
        imgui.text("Create Sample Image:")
        imgui.spacing()

        patterns = ['gradient', 'checkerboard', 'circle', 'edges', 'noise', 'rgb_gradient']
        imgui.push_item_width(150)
        if imgui.begin_combo("##pattern", self.sample_pattern):
            for p in patterns:
                if imgui.selectable(p, p == self.sample_pattern)[0]:
                    self.sample_pattern = p
            imgui.end_combo()
        imgui.pop_item_width()

        imgui.same_line()

        imgui.push_item_width(80)
        changed, self.sample_size = imgui.slider_int("Size", self.sample_size, 16, 128)
        imgui.pop_item_width()

        if imgui.button("Create Sample", width=-1):
            self.current_image = create_sample_image(self.sample_size, self.sample_pattern)
            self.processed_image = None

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text("Load from File:")
        imgui.push_item_width(-1)
        changed, self.image_path = imgui.input_text_with_hint(
            "##imgpath", "Path to image (PNG/JPG)...", self.image_path, 256
        )
        imgui.pop_item_width()

        if imgui.button("Load Image", width=-1):
            img = load_image(self.image_path, max_size=(128, 128))
            if img:
                self.current_image = img
                self.processed_image = None

        self._end_section()
