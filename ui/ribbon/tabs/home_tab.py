"""Home tab - common actions like undo/redo, clipboard, and theme."""

import imgui
from typing import Optional, Callable, Any

from ui.ribbon.ribbon_tab import RibbonTab
from ui.ribbon.ribbon_group import RibbonGroup
from ui.ribbon.ribbon_button import RibbonButton
from state.actions import Undo, Redo, ClearSelection, SetTheme


class HomeTab(RibbonTab):
    """Home tab with common actions."""

    def __init__(self):
        groups = [
            RibbonGroup("Clipboard", [
                RibbonButton("Paste", "Pst", tooltip="Paste from clipboard", enabled=False),
                RibbonButton("Copy", "Cpy", tooltip="Copy selection", enabled=False),
                RibbonButton("Cut", "Cut", tooltip="Cut selection", enabled=False),
            ]),
            RibbonGroup("History", [
                RibbonButton("Undo", "Undo", tooltip="Undo last action (Ctrl+Z)", action=Undo()),
                RibbonButton("Redo", "Redo", tooltip="Redo (Ctrl+Shift+Z)", action=Redo()),
            ]),
            RibbonGroup("Selection", [
                RibbonButton("Select\nAll", "All", tooltip="Select all objects", enabled=False),
                RibbonButton("Clear\nSel.", "Clr", tooltip="Clear selection", action=ClearSelection()),
            ]),
        ]
        super().__init__(groups)
        self._theme_items = ["Dark", "Light", "High Contrast"]
        self._theme_map = {
            "Dark": "dark",
            "Light": "light",
            "High Contrast": "high-contrast",
        }

    def render(
        self,
        state: Any,
        dispatch: Optional[Callable] = None,
        camera: Any = None,
        view_config: Any = None,
    ) -> None:
        """Render Home tab with history buttons and theme selector."""
        # Update button states based on history
        if state:
            # History group buttons
            history_group = self._groups[1]
            history_group.buttons[0].enabled = len(state.history) > 0  # Undo
            history_group.buttons[1].enabled = len(state.future) > 0  # Redo

            # Selection group
            sel_group = self._groups[2]
            sel_group.buttons[1].enabled = state.selected_id is not None  # Clear

        # Render button groups
        self._render_groups(state, dispatch)

        # Theme selector
        imgui.same_line()
        imgui.begin_group()
        imgui.spacing()

        imgui.text("Theme:")
        imgui.push_item_width(120)

        current_theme = "Dark"
        if state:
            for name, token in self._theme_map.items():
                if state.ui_theme == token:
                    current_theme = name
                    break

        if imgui.begin_combo("##theme", current_theme):
            for name in self._theme_items:
                if imgui.selectable(name, name == current_theme)[0]:
                    if dispatch:
                        dispatch(SetTheme(theme=self._theme_map[name]))
            imgui.end_combo()

        imgui.pop_item_width()
        imgui.spacing()
        imgui.text_disabled("Appearance")
        imgui.end_group()
