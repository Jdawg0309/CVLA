"""Left-hand tool palette (Photoshop-style)."""

import imgui

from state.actions import SetActiveTool
from ui.utils import set_next_window_position, set_next_window_size


_SET_WINDOW_POS_FIRST = getattr(imgui, "SET_WINDOW_POS_FIRST_USE_EVER", 0)
_SET_WINDOW_SIZE_FIRST = getattr(imgui, "SET_WINDOW_SIZE_FIRST_USE_EVER", 0)
_WINDOW_RESIZABLE = getattr(imgui, "WINDOW_RESIZABLE", 1)
_WINDOW_NO_COLLAPSE = getattr(imgui, "WINDOW_NO_COLLAPSE", 1)
_WINDOW_NO_SCROLLBAR = getattr(imgui, "WINDOW_NO_SCROLLBAR", 0)

class ToolPalette:
    def __init__(self):
        self._tools = [
            ("Select", "select", "V"),
            ("Move", "move", "M"),
            ("Rotate", "rotate", "R"),
            ("Add Vector", "add_vector", "+V"),
            ("Add Matrix", "add_matrix", "+M"),
            ("Image", "image", "Img"),
            ("Pipeline", "pipeline", "Pipe"),
        ]
        self._mode_tools = {
            "vectors": {"select", "move", "rotate", "add_vector", "add_matrix"},
            "images": {"select", "image", "pipeline"},
            "visualize": {"select", "move", "rotate"},
        }

    def render(self, rect, state, dispatch):
        x, y, width, height = rect
        set_next_window_position(x, y, cond=_SET_WINDOW_POS_FIRST)
        set_next_window_size((width, height), cond=_SET_WINDOW_SIZE_FIRST)
        flags = (
            _WINDOW_RESIZABLE |
            _WINDOW_NO_COLLAPSE |
            _WINDOW_NO_SCROLLBAR
        )
        if imgui.begin("Tools", flags=flags):
            button_w = max(40, width - 12)
            button_h = 40
            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 3.0)
            imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (4, 8))
            active_mode = state.active_mode if state else "vectors"
            allowed = self._mode_tools.get(active_mode, {t[1] for t in self._tools})
            for label, tool_id, short in self._tools:
                if tool_id not in allowed:
                    continue
                is_active = state is not None and state.active_tool == tool_id
                if is_active:
                    imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.45, 0.7, 1.0)
                    imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.35, 0.5, 0.78, 1.0)
                if imgui.button(short, button_w, button_h):
                    if dispatch:
                        dispatch(SetActiveTool(tool=tool_id))
                if imgui.is_item_hovered():
                    imgui.begin_tooltip()
                    imgui.text(label)
                    imgui.end_tooltip()
                if is_active:
                    imgui.pop_style_color(2)
                imgui.spacing()
            imgui.pop_style_var(2)

        imgui.end()
