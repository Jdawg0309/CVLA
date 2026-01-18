"""
Main UI layout for CVLA.

New Layout (2024):
- Left: Mode rail (60px) + Input Panel (360px)
- Right: Operations Panel (380px)
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
from ui.themes.theme_manager import apply_theme
from state.actions import DismissError

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

        self._last_theme = None

        # Layout mode toggle (for gradual migration)
        self._use_new_layout = True

        # Resizable layout state (new layout only)
        self._rail_w = 60
        self._left_w = 360
        self._right_w = 380
        self._splitter_thickness = 6

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

        # Render error modal on top of everything
        if state and state.show_error_modal:
            self._render_error_modal(state, dispatch)

    def _render_new_layout(self, state, dispatch, camera, view_config, app, display):
        """Render the new tensor-based layout."""
        # Layout dimensions
        top_h = 40      # Toolbar height
        rail_w = self._rail_w

        # Apply splitters before calculating panel rects.
        self._render_splitters(display, top_h)
        self._apply_layout_constraints(display)
        left_w = int(self._left_w)
        right_w = int(self._right_w)

        # Top toolbar
        self.toolbar.render(state, dispatch, camera, view_config, app)

        # Left mode selector rail (simplified)
        mode_rect = (0, top_h, rail_w, display.y - top_h)
        self.mode_selector.render(mode_rect, state, dispatch)

        # Left Input Panel
        input_rect = (rail_w, top_h, left_w, display.y - top_h)
        self.input_panel.render(input_rect, state, dispatch)

        # Right Operations Panel
        right_rect = (display.x - right_w, top_h, right_w, display.y - top_h)
        self.operations_panel.render(right_rect, state, dispatch)

    def _render_splitters(self, display, top_h):
        """Render draggable splitters for the new layout."""
        thickness = self._splitter_thickness
        height = max(0, display.y - top_h)
        io = imgui.get_io()

        resize_ew = getattr(imgui, "MOUSE_CURSOR_RESIZE_EW", None)
        no_bg = getattr(imgui, "WINDOW_NO_BACKGROUND", 0)
        no_focus = getattr(imgui, "WINDOW_NO_FOCUS_ON_APPEARING", 0)
        no_saved = getattr(imgui, "WINDOW_NO_SAVED_SETTINGS", 0)
        flags = (
            _WINDOW_NO_TITLE_BAR |
            _WINDOW_NO_RESIZE |
            _WINDOW_NO_MOVE |
            _WINDOW_NO_SCROLLBAR |
            _WINDOW_NO_COLLAPSE |
            no_bg |
            no_focus |
            no_saved
        )

        # Left vertical splitter (between input panel and viewport)
        left_x = self._rail_w + self._left_w - thickness / 2
        set_next_window_position(left_x, top_h, cond=_SET_WINDOW_POS_ALWAYS)
        set_next_window_size((thickness, height), cond=_SET_WINDOW_SIZE_ALWAYS)
        if imgui.begin("##split_left", flags=flags):
            imgui.invisible_button("##split_left_btn", thickness, height)
            if imgui.is_item_active():
                self._left_w += io.mouse_delta.x
            if imgui.is_item_hovered() and resize_ew is not None:
                imgui.set_mouse_cursor(resize_ew)
        imgui.end()

        # Right vertical splitter (between viewport and operations panel)
        right_x = display.x - self._right_w - thickness / 2
        set_next_window_position(right_x, top_h, cond=_SET_WINDOW_POS_ALWAYS)
        set_next_window_size((thickness, height), cond=_SET_WINDOW_SIZE_ALWAYS)
        if imgui.begin("##split_right", flags=flags):
            imgui.invisible_button("##split_right_btn", thickness, height)
            if imgui.is_item_active():
                self._right_w -= io.mouse_delta.x
            if imgui.is_item_hovered() and resize_ew is not None:
                imgui.set_mouse_cursor(resize_ew)
        imgui.end()

    def _apply_layout_constraints(self, display):
        """Clamp panel sizes so the center area remains usable."""
        min_left = 240
        min_right = 280
        min_center = 360

        available_w = max(0, display.x - self._rail_w)
        self._left_w = max(min_left, self._left_w)
        self._right_w = max(min_right, self._right_w)

        max_left = max(min_left, available_w - min_center - min_right)
        self._left_w = min(self._left_w, max_left)

        max_right = max(min_right, available_w - min_center - self._left_w)
        self._right_w = min(self._right_w, max_right)

        center_w = available_w - self._left_w - self._right_w
        if center_w < min_center:
            overflow = min_center - center_w
            reduce_right = min(self._right_w - min_right, overflow)
            self._right_w -= reduce_right
            overflow -= reduce_right
            if overflow > 0:
                reduce_left = min(self._left_w - min_left, overflow)
                self._left_w -= reduce_left

    def _render_legacy_layout(self, state, dispatch, camera, view_config, app, display):
        """Render the legacy layout (original Photoshop-style)."""
        top_h = 40
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
        left_rect = (rail_w, top_h, left_w, display.y - top_h)
        self._legacy_sidebar.render(left_rect, camera, view_config, state, dispatch)

        # Right inspector panel
        right_rect = (display.x - right_w, top_h, right_w, display.y - top_h)
        self._legacy_inspector.render(state, dispatch, right_rect)

    def toggle_layout(self):
        """Toggle between new and legacy layout."""
        self._use_new_layout = not self._use_new_layout
        return self._use_new_layout

    def _render_error_modal(self, state, dispatch):
        """Render the error modal popup."""
        # Open the popup if not already open
        imgui.open_popup("Error##modal")

        # Center the modal on screen
        display = imgui.get_io().display_size
        modal_width = 400
        modal_height = 150
        imgui.set_next_window_size(modal_width, modal_height)
        imgui.set_next_window_position(
            (display.x - modal_width) / 2,
            (display.y - modal_height) / 2
        )

        if imgui.begin_popup_modal("Error##modal", flags=_WINDOW_NO_RESIZE)[0]:
            imgui.spacing()

            # Error icon and message
            imgui.text_colored("Error", 1.0, 0.3, 0.3, 1.0)
            imgui.separator()
            imgui.spacing()

            # Word-wrap the error message
            imgui.push_text_wrap_pos(modal_width - 20)
            imgui.text(state.error_message or "An error occurred.")
            imgui.pop_text_wrap_pos()

            imgui.spacing()
            imgui.spacing()

            # OK button to dismiss
            button_width = 100
            imgui.set_cursor_pos_x((modal_width - button_width) / 2)
            if imgui.button("OK", button_width, 28):
                dispatch(DismissError())
                imgui.close_current_popup()

            imgui.end_popup()
