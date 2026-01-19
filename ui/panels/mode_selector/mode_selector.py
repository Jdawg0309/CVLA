"""Mode selector rail for CVLA."""

import imgui

from state.actions import SetActiveMode
from ui.utils import set_next_window_position, set_next_window_size

_SET_WINDOW_POS_FIRST = getattr(imgui, "SET_WINDOW_POS_FIRST_USE_EVER", 0)
_SET_WINDOW_SIZE_FIRST = getattr(imgui, "SET_WINDOW_SIZE_FIRST_USE_EVER", 0)
_WINDOW_NO_RESIZE = getattr(imgui, "WINDOW_NO_RESIZE", 0)
_WINDOW_NO_COLLAPSE = getattr(imgui, "WINDOW_NO_COLLAPSE", 0)
_WINDOW_NO_SCROLLBAR = getattr(imgui, "WINDOW_NO_SCROLLBAR", 0)


class ModeSelector:
    def __init__(self):
        self._modes = [
            ("Algebra", "vectors"),
            ("View", "visualize"),
            ("Settings", "settings"),
        ]

    def render(self, rect, state, dispatch):
        x, y, width, height = rect
        set_next_window_position(x, y, cond=_SET_WINDOW_POS_FIRST)
        set_next_window_size((width, height), cond=_SET_WINDOW_SIZE_FIRST)
        flags = _WINDOW_NO_RESIZE | _WINDOW_NO_COLLAPSE | _WINDOW_NO_SCROLLBAR
        if imgui.begin("Modes", flags=flags):
            button_w = max(40, width - 12)
            button_h = 28
            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 3.0)
            imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (4, 6))
            active_mode = state.active_mode if state else "vectors"
            for label, mode_id in self._modes:
                is_active = active_mode == mode_id
                if is_active:
                    imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.4, 0.55, 1.0)
                if imgui.button(label, button_w, button_h):
                    if dispatch:
                        dispatch(SetActiveMode(mode=mode_id))
                if is_active:
                    imgui.pop_style_color(1)
            imgui.pop_style_var(2)
        imgui.end()
