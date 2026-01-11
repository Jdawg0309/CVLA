"""
Images tab pipeline section.
"""

import imgui
from typing import Callable

from state import (
    AppState,
    Action,
    StepForward,
    StepBackward,
    JumpToStep,
    ResetPipeline,
    get_current_step,
)


def _render_pipeline_section(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render educational pipeline steps."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.18, 0.15, 0.18, 0.8)

    if imgui.collapsing_header("Pipeline Steps", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        imgui.indent(10)

        total_steps = len(state.pipeline_steps)
        current_idx = state.pipeline_step_index

        imgui.text(f"Step {current_idx + 1} of {total_steps}")

        can_back = current_idx > 0
        can_forward = current_idx < total_steps - 1

        if not can_back:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
        if imgui.button("<< Prev") and can_back:
            dispatch(StepBackward())
        if not can_back:
            imgui.pop_style_var()

        imgui.same_line()

        if not can_forward:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
        if imgui.button("Next >>") and can_forward:
            dispatch(StepForward())
        if not can_forward:
            imgui.pop_style_var()

        imgui.same_line()
        if imgui.button("Reset"):
            dispatch(ResetPipeline())

        imgui.spacing()

        imgui.begin_child("##steps", 0, 120, border=True)
        for i, step in enumerate(state.pipeline_steps):
            is_current = (i == current_idx)

            if is_current:
                imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 0.4, 1.0)

            if imgui.selectable(f"{i+1}. {step.title}", is_current)[0]:
                dispatch(JumpToStep(index=i))

            if is_current:
                imgui.pop_style_color()

        imgui.end_child()

        current_step = get_current_step(state)
        if current_step:
            imgui.spacing()
            imgui.text_wrapped(current_step.explanation)

        imgui.unindent(10)
        imgui.spacing()

    imgui.pop_style_color()
