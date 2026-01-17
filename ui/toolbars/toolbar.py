"""Top toolbar for CVLA."""

import imgui

from state.actions import SetTheme, Undo, Redo


class Toolbar:
    def __init__(self):
        self._theme_items = ["Dark", "Light", "High Contrast"]
        self._theme_map = {
            "Dark": "dark",
            "Light": "light",
            "High Contrast": "high-contrast",
        }
    def _render_top_bar(self, state, dispatch, app):
        display = imgui.get_io().display_size
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(display.x, 40)

        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (10, 6))
        imgui.push_style_color(imgui.COLOR_BORDER, 0.12, 0.12, 0.12, 1.0)

        if imgui.begin(
            "TopBar",
            flags=imgui.WINDOW_NO_TITLE_BAR |
                  imgui.WINDOW_NO_RESIZE |
                  imgui.WINDOW_NO_MOVE |
                  imgui.WINDOW_NO_SCROLLBAR |
                  imgui.WINDOW_NO_SAVED_SETTINGS,
        ):
            imgui.text_colored("CVLA", 0.9, 0.92, 0.95, 1.0)
            imgui.same_line()
            imgui.text_disabled("Research Engine")
            imgui.same_line(220)

            can_undo = state is not None and len(state.history) > 0
            can_redo = state is not None and len(state.future) > 0
            if imgui.button("Undo") and dispatch and can_undo:
                dispatch(Undo())
            imgui.same_line()
            if imgui.button("Redo") and dispatch and can_redo:
                dispatch(Redo())
            imgui.same_line()

            imgui.same_line(display.x - 230)
            current_theme = "Dark"
            if state:
                for name, token in self._theme_map.items():
                    if state.ui_theme == token:
                        current_theme = name
                        break
            if imgui.begin_combo("Theme", current_theme):
                for name in self._theme_items:
                    if imgui.selectable(name, name == current_theme)[0]:
                        if dispatch:
                            dispatch(SetTheme(theme=self._theme_map[name]))
                imgui.end_combo()

        imgui.end()
        imgui.pop_style_var(2)
        imgui.pop_style_color(1)

    def render(self, state, dispatch, camera, view_config, app):
        """Render toolbar at top of screen."""
        self._render_top_bar(state, dispatch, app)
