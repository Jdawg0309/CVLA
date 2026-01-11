"""Main UI layout for CVLA (Photoshop-style)."""

import imgui

from ui.utils import set_next_window_position, set_next_window_size

_SET_WINDOW_POS_ALWAYS = getattr(imgui, "SET_WINDOW_POS_ALWAYS", 0)
_SET_WINDOW_SIZE_ALWAYS = getattr(imgui, "SET_WINDOW_SIZE_ALWAYS", 0)
_DOCK_SPACE = getattr(imgui, "dock_space", None)
_GET_ID = getattr(imgui, "get_id", None)
_DOCK_PASSTHRU_FLAG = getattr(imgui, "DOCKNODE_FLAG_PASSTHRU_CENTRAL_NODE", 0)
_WINDOW_NO_COLLAPSE = getattr(imgui, "WINDOW_NO_COLLAPSE", 0)
_WINDOW_NO_TITLE_BAR = getattr(imgui, "WINDOW_NO_TITLE_BAR", 0)
_WINDOW_NO_RESIZE = getattr(imgui, "WINDOW_NO_RESIZE", 0)
_WINDOW_NO_MOVE = getattr(imgui, "WINDOW_NO_MOVE", 0)
_WINDOW_NO_SCROLLBAR = getattr(imgui, "WINDOW_NO_SCROLLBAR", 0)
_WINDOW_MENU_BAR = getattr(imgui, "WINDOW_MENU_BAR", 0)
_DOCK_SPACE = getattr(imgui, "dock_space", None)
_GET_ID = getattr(imgui, "get_id", None)

from ui.toolbars.toolbar import Toolbar
from ui.panels.sidebar.sidebar import Sidebar
from ui.panels.tool_palette.tool_palette import ToolPalette
from ui.inspectors.inspector import Inspector
from ui.panels.timeline.timeline_panel import TimelinePanel
from ui.themes.theme_manager import apply_theme


class WorkspaceLayout:
    def __init__(self):
        self.toolbar = Toolbar()
        self.tool_palette = ToolPalette()
        self.operations_panel = Sidebar()
        self.inspector = Inspector()
        self.timeline = TimelinePanel()
        self._last_theme = None

    def render(self, state, dispatch, camera, view_config, app):
        """Render the full UI layout."""
        if state and state.ui_theme != self._last_theme:
            apply_theme(state.ui_theme)
            self._last_theme = state.ui_theme

        display = imgui.get_io().display_size
        dock_flags = (
            _WINDOW_NO_COLLAPSE |
            _WINDOW_NO_TITLE_BAR |
            _WINDOW_NO_RESIZE |
            _WINDOW_NO_MOVE |
            _WINDOW_NO_SCROLLBAR |
            _WINDOW_MENU_BAR
        )
        set_next_window_position(0, 0, cond=_SET_WINDOW_POS_ALWAYS)
        set_next_window_size((display.x, display.y), cond=_SET_WINDOW_SIZE_ALWAYS)
        if _DOCK_SPACE and _GET_ID:
            if imgui.begin("Workspace Dockspace", flags=dock_flags):
                dockspace_id = _GET_ID("WorkspaceDockspace")
                _DOCK_SPACE(dockspace_id, 0, 0, _DOCK_PASSTHRU_FLAG)
            imgui.end()

        top_h = 58
        bottom_h = 120
        tool_w = 64
        left_w = 320
        right_w = 320

        # Top toolbar
        self.toolbar.render(state, dispatch, camera, view_config, app)

        # Left tool palette
        tool_rect = (0, top_h, tool_w, display.y - top_h - bottom_h)
        self.tool_palette.render(tool_rect, state, dispatch)

        # Left operations panel
        left_rect = (tool_w, top_h, left_w, display.y - top_h - bottom_h)
        self.operations_panel.render(left_rect, camera, view_config, state, dispatch)

        # Right inspector panel
        right_rect = (display.x - right_w, top_h, right_w, display.y - top_h - bottom_h)
        self.inspector.render(state, dispatch, right_rect)

        # Bottom timeline
        bottom_rect = (0, display.y - bottom_h, display.x, bottom_h)
        self.timeline.render(bottom_rect, state, dispatch)
