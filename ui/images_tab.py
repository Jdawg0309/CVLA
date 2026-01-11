"""
Images Tab UI Component - State-Driven Architecture.
"""

import imgui
from typing import Callable

from state import AppState, Action
from ui.images_source_section import _render_image_source_section
from ui.images_info_section import _render_image_info_section
from ui.images_convolution_section import _render_convolution_section
from ui.images_transform_section import _render_transform_section
from ui.images_result_section import _render_result_section
from ui.images_pipeline_section import _render_pipeline_section
from ui.images_educational_section import _render_educational_section


def render_images_tab(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """
    Render the Images tab.
    """
    try:
        from vision import list_kernels, get_kernel_by_name
        vision_available = True
    except ImportError:
        vision_available = False

    if not vision_available:
        imgui.text_colored("Vision module not available", 0.8, 0.4, 0.4, 1.0)
        imgui.text_disabled("Install Pillow: pip install Pillow")
        return

    _render_image_source_section(state, dispatch)

    if state.current_image is not None:
        _render_image_info_section(state, dispatch)

    if state.current_image is not None:
        _render_convolution_section(state, dispatch)

    if state.current_image is not None:
        _render_transform_section(state, dispatch)

    if state.processed_image is not None:
        _render_result_section(state, dispatch)

    if state.pipeline_steps:
        _render_pipeline_section(state, dispatch)

    _render_educational_section(state)
