"""
Images tab source section.
"""

import imgui
from typing import Callable

from state import (
    AppState,
    Action,
    CreateSampleImage,
    LoadImage,
    SetImagePath,
    SetSamplePattern,
    SetSampleSize,
)
from ui.images_tab_constants import PATTERN_OPTIONS


def _render_image_source_section(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render image loading/creation controls."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.15, 0.15, 0.18, 0.8)

    if imgui.collapsing_header("Image Source", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        imgui.indent(10)

        imgui.text("Create Sample Image:")
        imgui.spacing()

        imgui.push_item_width(150)
        if imgui.begin_combo("##pattern", state.input_sample_pattern):
            for pattern in PATTERN_OPTIONS:
                is_selected = (pattern == state.input_sample_pattern)
                if imgui.selectable(pattern, is_selected)[0]:
                    dispatch(SetSamplePattern(pattern=pattern))
                if is_selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        imgui.pop_item_width()

        imgui.same_line()

        imgui.push_item_width(100)
        changed, new_size = imgui.slider_int("Size##sample", state.input_sample_size, 8, 128)
        if changed:
            dispatch(SetSampleSize(size=new_size))
        imgui.pop_item_width()

        if imgui.button("Create Sample", width=260):
            dispatch(CreateSampleImage(
                pattern=state.input_sample_pattern,
                size=state.input_sample_size
            ))

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text("Load from File:")

        imgui.push_item_width(260)
        changed, new_path = imgui.input_text_with_hint(
            "##imgpath",
            "Path to image (PNG/JPG)...",
            state.input_image_path,
            256
        )
        if changed:
            dispatch(SetImagePath(path=new_path))
        imgui.pop_item_width()

        can_load = len(state.input_image_path.strip()) > 0
        if not can_load:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
        if imgui.button("Load Image", width=260) and can_load:
            dispatch(LoadImage(path=state.input_image_path))
        if not can_load:
            imgui.pop_style_var()

        imgui.unindent(10)
        imgui.spacing()

    imgui.pop_style_color()
