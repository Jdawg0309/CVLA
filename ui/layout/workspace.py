"""
Main UI layout for CVLA.

New Layout (2024):
- Left: Mode rail (60px) + Input Panel (360px)
- Right: Operations Panel (380px)
- Bottom: Timeline (100px)
- Center: 3D Viewport
"""

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

from ui.toolbars.toolbar import Toolbar
from ui.panels.mode_selector.mode_selector import ModeSelector
from ui.panels.input_panel import InputPanel
from ui.panels.operations_panel import OperationsPanel
from ui.panels.timeline.timeline_panel import TimelinePanel
from ui.themes.theme_manager import apply_theme

# Legacy imports (for fallback if new panels fail)
from ui.panels.sidebar.sidebar import Sidebar
from ui.inspectors.inspector import Inspector


class WorkspaceLayout:
    """
    Main workspace layout manager.

    Layout structure:
    +------------------------------------------------------------------+
    |                        TOOLBAR (40px)                            |
    +------------------------------------------------------------------+
    | MODE |         LEFT INPUT PANEL        |    RIGHT OPS PANEL      |
    | (60) |             (360px)             |        (380px)          |
    |      |                                 |                         |
    |      |   3D VIEWPORT (CENTER)          |                         |
    +------+---------------------------------+-------------------------+
    |                    TIMELINE (100px)                              |
    +------------------------------------------------------------------+
    """

    def __init__(self):
        self.toolbar = Toolbar()
        self.mode_selector = ModeSelector()

        # New panels (tensor-based UI)
        self.input_panel = InputPanel()
        self.operations_panel = OperationsPanel()

        # Legacy panels (kept for fallback)
        self._legacy_sidebar = Sidebar()
        self._legacy_inspector = Inspector()

        self.timeline = TimelinePanel()
        self._last_theme = None

        # Layout mode toggle (for gradual migration)
        self._use_new_layout = True

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

        if self._use_new_layout:
            self._render_new_layout(state, dispatch, camera, view_config, app, display)
        else:
            self._render_legacy_layout(state, dispatch, camera, view_config, app, display)

    def _render_new_layout(self, state, dispatch, camera, view_config, app, display):
        """Render the new tensor-based layout."""
        # Layout dimensions
        top_h = 40      # Toolbar height
        bottom_h = 100  # Timeline height
        rail_w = 60     # Mode selector rail width
        left_w = 360    # Input panel width
        right_w = 380   # Operations panel width

        # Top toolbar
        self.toolbar.render(state, dispatch, camera, view_config, app)

        # Left mode selector rail (simplified)
        mode_rect = (0, top_h, rail_w, display.y - top_h - bottom_h)
        self.mode_selector.render(mode_rect, state, dispatch)

        # Left Input Panel
        input_rect = (rail_w, top_h, left_w, display.y - top_h - bottom_h)
        self.input_panel.render(input_rect, state, dispatch)

        # Right Operations Panel
        right_rect = (display.x - right_w, top_h, right_w, display.y - top_h - bottom_h)
        self.operations_panel.render(right_rect, state, dispatch)

        # Bottom Timeline
        bottom_rect = (0, display.y - bottom_h, display.x, bottom_h)
        self.timeline.render(bottom_rect, state, dispatch)

    def _render_legacy_layout(self, state, dispatch, camera, view_config, app, display):
        """Render the legacy layout (original Photoshop-style)."""
        top_h = 40
        bottom_h = 120
        rail_w = 120
        mode_h = 190
        left_w = 320
        right_w = 320

        # Top toolbar
        self.toolbar.render(state, dispatch, camera, view_config, app)

        # Left mode selector
        mode_rect = (0, top_h, rail_w, mode_h)
        self.mode_selector.render(mode_rect, state, dispatch)

        # Left operations panel (sidebar)
        left_rect = (rail_w, top_h, left_w, display.y - top_h - bottom_h)
        self._legacy_sidebar.render(left_rect, camera, view_config, state, dispatch)

        # Right inspector panel
        right_rect = (display.x - right_w, top_h, right_w, display.y - top_h - bottom_h)
        self._legacy_inspector.render(state, dispatch, right_rect)

        # Bottom timeline
        bottom_rect = (0, display.y - bottom_h, display.x, bottom_h)
        self.timeline.render(bottom_rect, state, dispatch)

    def toggle_layout(self):
        """Toggle between new and legacy layout."""
        self._use_new_layout = not self._use_new_layout
        return self._use_new_layout
