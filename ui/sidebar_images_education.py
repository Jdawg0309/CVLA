"""
Sidebar image education section.
"""

import imgui


def _render_image_education_section(self):
    """Render ML/CV education section."""
    if self._section("ML/CV Info", "Info", default_open=False):
        imgui.text_wrapped(
            "Images are matrices! Each pixel is a number (0-1). "
            "Convolution kernels slide over the image computing "
            "weighted sums - this is how CNNs detect edges, "
            "textures, and patterns. The kernels shown here are "
            "what neural networks learn automatically from data."
        )
        imgui.spacing()
        imgui.text_disabled("Pipeline: Image -> Matrix -> Transform -> Result")
        self._end_section()
