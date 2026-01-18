"""Timeline panel for educational steps and history."""

import imgui

from state.actions import StepForward, StepBackward, JumpToStep
from ui.utils import set_next_window_position, set_next_window_size

_SET_WINDOW_POS_FIRST = getattr(imgui, "SET_WINDOW_POS_FIRST_USE_EVER", 0)
_SET_WINDOW_SIZE_FIRST = getattr(imgui, "SET_WINDOW_SIZE_FIRST_USE_EVER", 0)
_WINDOW_RESIZABLE = getattr(imgui, "WINDOW_RESIZABLE", 1)
_WINDOW_NO_COLLAPSE = getattr(imgui, "WINDOW_NO_COLLAPSE", 1)


class TimelinePanel:
    def __init__(self):
        self._step_filter = ""

    def render(self, rect, state, dispatch):
        """Render bottom timeline panel."""
        if state is None:
            return
        x, y, width, height = rect
        set_next_window_position(x, y, cond=_SET_WINDOW_POS_FIRST)
        set_next_window_size((width, height), cond=_SET_WINDOW_SIZE_FIRST)
        imgui.set_next_window_size_constraints(
            (width, 96),
            (width + 40, height + 80),
        )

        flags = _WINDOW_RESIZABLE | _WINDOW_NO_COLLAPSE
        if imgui.begin("Timeline", flags=flags):
            imgui.text("Timeline")
            imgui.same_line()
            steps = state.operations.steps if state.operations and state.operations.steps else state.pipeline_steps
            step_index = state.operations.step_index if state.operations and state.operations.steps else state.pipeline_step_index
            imgui.text_disabled(f"{len(steps)} steps")

            imgui.same_line(width - 180)
            if imgui.small_button("Prev") and dispatch:
                dispatch(StepBackward())
            imgui.same_line()
            if imgui.small_button("Next") and dispatch:
                dispatch(StepForward())

            imgui.same_line()
            imgui.push_item_width(140)
            _, self._step_filter = imgui.input_text_with_hint(
                "##step_filter", "Filter...", self._step_filter, 64
            )
            imgui.pop_item_width()

            imgui.spacing()
            imgui.begin_child(
                "##timeline_steps",
                0,
                height - 48,
                border=True,
                flags=getattr(imgui, "WINDOW_ALWAYS_VERTICAL_SCROLLBAR", 0),
            )
            for idx, step in enumerate(steps):
                title = getattr(step, "title", f"Step {idx + 1}")
                if self._step_filter and self._step_filter.lower() not in title.lower():
                    continue
                is_active = idx == step_index
                label = f"{idx+1}. {title}"
                if imgui.selectable(label, is_active)[0] and dispatch:
                    dispatch(JumpToStep(index=idx))
            imgui.end_child()

        imgui.end()
