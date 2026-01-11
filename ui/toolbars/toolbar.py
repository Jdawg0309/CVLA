"""Photoshop-style top toolbar for CVLA."""

import imgui

from state.actions import SetActiveTab, SetTheme, Undo, Redo


class Toolbar:
    def __init__(self):
        self._theme_items = ["Dark", "Light", "High Contrast"]
        self._theme_map = {
            "Dark": "dark",
            "Light": "light",
            "High Contrast": "high-contrast",
        }
        self._tabs = [
            ("Vectors", "vectors"),
            ("Matrices", "matrices"),
            ("Images", "images"),
            ("Pipelines", "pipelines"),
            ("View", "visualize"),
            ("Help", "help"),
        ]

    def _render_menu_bar(self, state, dispatch, view_config, app):
        if imgui.begin_main_menu_bar():
            imgui.text_colored("CVLA", 0.9, 0.92, 0.95, 1.0)
            imgui.same_line()
            imgui.text_disabled("Research Engine")
            imgui.same_line(180)

            if imgui.begin_menu("File"):
                imgui.menu_item("New Scene", None, False, False)
                imgui.menu_item("Open...", None, False, False)
                imgui.menu_item("Save", None, False, False)
                imgui.end_menu()
            if imgui.begin_menu("Edit"):
                can_undo = state is not None and len(state.history) > 0
                can_redo = state is not None and len(state.future) > 0
                if imgui.menu_item("Undo", "Ctrl+Z", False, can_undo)[0] and dispatch:
                    dispatch(Undo())
                if imgui.menu_item("Redo", "Ctrl+Shift+Z", False, can_redo)[0] and dispatch:
                    dispatch(Redo())
                imgui.end_menu()
            if imgui.begin_menu("View"):
                if imgui.menu_item("Grid", None, view_config.show_grid, True)[0]:
                    view_config.update(show_grid=not view_config.show_grid)
                if imgui.menu_item("Axes", None, view_config.show_axes, True)[0]:
                    view_config.update(show_axes=not view_config.show_axes)
                if imgui.menu_item("Labels", None, view_config.show_labels, True)[0]:
                    view_config.update(show_labels=not view_config.show_labels)
                imgui.end_menu()
            if imgui.begin_menu("Window"):
                imgui.menu_item("Workspace", None, False, False)
                imgui.end_menu()
            if imgui.begin_menu("Help"):
                imgui.menu_item("About CVLA", None, False, False)
                imgui.end_menu()

            if hasattr(app, "fps"):
                fps_color = (0.2, 0.8, 0.2, 1.0) if app.fps > 30 else (0.8, 0.2, 0.2, 1.0)
                imgui.same_line(imgui.get_window_width() - 80)
                imgui.text_colored(f"{app.fps:.0f} FPS", *fps_color)

            imgui.end_main_menu_bar()

    def _render_options_bar(self, state, dispatch, view_config):
        display = imgui.get_io().display_size
        imgui.set_next_window_position(0, 22)
        imgui.set_next_window_size(display.x, 36)

        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (10, 6))
        imgui.push_style_color(imgui.COLOR_BORDER, 0.12, 0.12, 0.12, 1.0)

        if imgui.begin(
            "OptionsBar",
            flags=imgui.WINDOW_NO_TITLE_BAR |
                  imgui.WINDOW_NO_RESIZE |
                  imgui.WINDOW_NO_MOVE |
                  imgui.WINDOW_NO_SCROLLBAR |
                  imgui.WINDOW_NO_SAVED_SETTINGS,
        ):
            active_tab = state.active_tab if state else "vectors"
            for label, tab_id in self._tabs:
                is_active = tab_id == active_tab
                if is_active:
                    imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.4, 0.55, 1.0)
                if imgui.button(label):
                    if dispatch:
                        dispatch(SetActiveTab(tab=tab_id))
                if is_active:
                    imgui.pop_style_color(1)
                imgui.same_line()

            imgui.same_line(420)
            can_undo = state is not None and len(state.history) > 0
            can_redo = state is not None and len(state.future) > 0
            if imgui.button("Undo") and dispatch and can_undo:
                dispatch(Undo())
            imgui.same_line()
            if imgui.button("Redo") and dispatch and can_redo:
                dispatch(Redo())
            imgui.same_line()

            if imgui.button("Grid"):
                view_config.update(show_grid=not view_config.show_grid)
            imgui.same_line()
            if imgui.button("Axes"):
                view_config.update(show_axes=not view_config.show_axes)
            imgui.same_line()
            if imgui.button("Labels"):
                view_config.update(show_labels=not view_config.show_labels)

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
        self._render_menu_bar(state, dispatch, view_config, app)
        self._render_options_bar(state, dispatch, view_config)
