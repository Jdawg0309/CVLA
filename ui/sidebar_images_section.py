"""
Sidebar image operations section.
"""

import imgui
from ui.sidebar_vision import VISION_AVAILABLE


def _render_image_operations(self):
    """Render image processing section for ML/CV visualization."""
    if not VISION_AVAILABLE:
        imgui.text_colored("Vision module not available", 0.8, 0.4, 0.4, 1.0)
        imgui.text_disabled("Install Pillow: pip install Pillow")
        return

    self._render_image_source_section()
    self._render_image_info_section()
    self._render_image_convolution_section()
    self._render_image_transform_section()
    self._render_image_result_section()
    self._render_image_education_section()
