"""
Images tab transform section.
"""

import imgui
from typing import Callable

from state import (
    AppState,
    Action,
    ApplyTransform,
    FlipImageHorizontal,
    SetTransformRotation,
    SetTransformScale,
)


def _render_transform_section(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render affine transform controls."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.15, 0.15, 0.18, 0.8)

    if imgui.collapsing_header("Affine Transforms")[0]:
        imgui.indent(10)

        imgui.text_colored("Linear transforms on coordinates", 0.4, 0.8, 0.4, 1.0)
        imgui.spacing()

        imgui.push_item_width(200)
        changed, new_rot = imgui.slider_float(
            "Rotation##transform",
            state.input_transform_rotation,
            -180, 180, "%.1f deg"
        )
        if changed:
            dispatch(SetTransformRotation(rotation=new_rot))
        imgui.pop_item_width()

        imgui.push_item_width(200)
        changed, new_scale = imgui.slider_float(
            "Scale##transform",
            state.input_transform_scale,
            0.5, 2.0, "%.2fx"
        )
        if changed:
            dispatch(SetTransformScale(scale=new_scale))
        imgui.pop_item_width()

        imgui.spacing()

        if imgui.button("Apply Transform", width=260):
            dispatch(ApplyTransform(
                rotation=state.input_transform_rotation,
                scale=state.input_transform_scale
            ))

        imgui.spacing()

        if imgui.button("Flip Horizontal", width=260):
            dispatch(FlipImageHorizontal())

        imgui.unindent(10)
        imgui.spacing()

    imgui.pop_style_color()
