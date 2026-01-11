"""
Images Tab UI Component - State-Driven Architecture.
"""

import imgui
from typing import Callable

from state import (
    AppState,
    Action,
    NormalizeImage,
    SetActiveImageTab,
    SetImageColorMode,
    SetImageNormalizeMean,
    SetImageNormalizeStd,
)
from ui.panels.images.images_source_section import _render_image_source_section
from ui.panels.images.images_info_section import _render_image_info_section
from ui.panels.images.images_convolution_section import _render_convolution_section
from ui.panels.images.images_transform_section import _render_transform_section
from ui.panels.images.images_result_section import _render_result_section
from ui.panels.images.images_pipeline_section import _render_pipeline_section
from ui.panels.images.images_educational_section import _render_educational_section


def render_images_tab(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """
    Render the Images tab.
    """
    try:
        from domain.images import list_kernels, get_kernel_by_name
        vision_available = True
    except ImportError:
        vision_available = False

    if not vision_available:
        imgui.text_colored("Vision module not available", 0.8, 0.4, 0.4, 1.0)
        imgui.text_disabled("Install Pillow: pip install Pillow")
        return

    _render_image_source_section(state, dispatch)

    if state.current_image is None:
        return

    _render_image_tab_selector(state, dispatch)

    if state.active_image_tab == "raw":
        _render_raw_tab(state, dispatch)
    else:
        _render_preprocess_tab(state, dispatch)

    _render_educational_section(state)


def _render_raw_tab(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render the raw image tab with color mode controls."""
    _render_color_mode_selector(state, dispatch)
    _render_image_info_section(state, dispatch)


def _render_image_tab_selector(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render the button group that switches between Raw / Preprocess."""
    tabs = [("Raw Image", "raw"), ("Preprocess", "preprocess")]
    imgui.begin_group()
    for idx, (label, tab_id) in enumerate(tabs):
        active = (state.active_image_tab == tab_id)
        if active:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.45, 0.75, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.28, 0.55, 0.85, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.2, 0.4, 0.7, 1.0)
        if imgui.button(label, 140, 28):
            dispatch(SetActiveImageTab(tab=tab_id))
        if active:
            imgui.pop_style_color(3)
        if idx < len(tabs) - 1:
            imgui.same_line()
    imgui.end_group()
    imgui.separator()


def _render_preprocess_tab(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render preprocessing controls (normalize, convolution, etc.)."""
    _render_normalization_section(state, dispatch)

    _render_convolution_section(state, dispatch)
    _render_transform_section(state, dispatch)

    if state.processed_image is not None:
        _render_result_section(state, dispatch)

    if state.pipeline_steps:
        _render_pipeline_section(state, dispatch)


def _render_color_mode_selector(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render controls to switch between RGB, grayscale, and heatmap mode."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.16, 0.16, 0.19, 0.75)
    if imgui.collapsing_header("Image Color Mode", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        imgui.indent(10)
        imgui.text("Preview processed images as:")
        imgui.spacing()
        modes = [("RGB (default)", "rgb"), ("Grayscale", "grayscale"), ("Heatmap", "heatmap")]
        for idx, (label_text, mode_id) in enumerate(modes):
            selected = (state.image_color_mode == mode_id)
            if imgui.radio_button(label_text, selected):
                dispatch(SetImageColorMode(mode=mode_id))
            if idx < len(modes) - 1:
                imgui.same_line()
        imgui.unindent(10)
        imgui.spacing()
    imgui.pop_style_color()


def _render_normalization_section(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render normalization inputs and action."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.16, 0.16, 0.19, 0.75)
    if imgui.collapsing_header("Normalize Image", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        imgui.indent(10)
        imgui.text("Standardize the raw image before downstream kernels.")

        changed, new_mean = imgui.input_float(
            "Mean",
            state.input_image_normalize_mean,
            format="%.3f"
        )
        if changed:
            dispatch(SetImageNormalizeMean(mean=new_mean))

        imgui.same_line()
        changed, new_std = imgui.input_float(
            "Std Dev",
            state.input_image_normalize_std,
            format="%.3f"
        )
        if changed:
            dispatch(SetImageNormalizeStd(std=new_std))

        imgui.spacing()
        if imgui.button("Normalize Raw Image", width=-1):
            dispatch(NormalizeImage(
                mean=state.input_image_normalize_mean,
                std=state.input_image_normalize_std,
            ))

        imgui.unindent(10)
        imgui.spacing()
    imgui.pop_style_color()
