"""Theme management for CVLA UI."""

import imgui


def _apply_base(style):
    style.window_rounding = 2.0
    style.frame_rounding = 2.0
    style.scrollbar_rounding = 2.0
    style.grab_rounding = 2.0
    style.window_padding = (8, 6)
    style.frame_padding = (6, 3)
    style.item_spacing = (6, 4)
    style.item_inner_spacing = (4, 3)
    style.window_border_size = 1.0
    style.frame_border_size = 1.0


def apply_theme(theme_name: str) -> None:
    """Apply a named theme to ImGui."""
    style = imgui.get_style()
    _apply_base(style)

    if theme_name == "light":
        colors = style.colors
        colors[imgui.COLOR_TEXT] = (0.15, 0.15, 0.16, 1.0)
        colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.92, 0.92, 0.93, 1.0)
        colors[imgui.COLOR_CHILD_BACKGROUND] = (0.95, 0.95, 0.96, 1.0)
        colors[imgui.COLOR_FRAME_BACKGROUND] = (0.84, 0.84, 0.86, 1.0)
        colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.8, 0.8, 0.82, 1.0)
        colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = (0.76, 0.76, 0.78, 1.0)
        colors[imgui.COLOR_BUTTON] = (0.25, 0.4, 0.65, 0.9)
        colors[imgui.COLOR_BUTTON_HOVERED] = (0.27, 0.45, 0.72, 1.0)
        colors[imgui.COLOR_BUTTON_ACTIVE] = (0.22, 0.36, 0.6, 1.0)
        colors[imgui.COLOR_HEADER] = (0.3, 0.4, 0.6, 0.35)
        colors[imgui.COLOR_HEADER_HOVERED] = (0.3, 0.4, 0.6, 0.45)
        colors[imgui.COLOR_HEADER_ACTIVE] = (0.3, 0.4, 0.6, 0.55)
        colors[imgui.COLOR_SEPARATOR] = (0.7, 0.7, 0.72, 1.0)
        colors[imgui.COLOR_TAB] = (0.84, 0.84, 0.86, 1.0)
        colors[imgui.COLOR_TAB_ACTIVE] = (0.3, 0.4, 0.6, 1.0)
        colors[imgui.COLOR_TAB_HOVERED] = (0.35, 0.48, 0.7, 1.0)
        colors[imgui.COLOR_TITLE_BACKGROUND] = (0.88, 0.88, 0.9, 1.0)
        colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = (0.88, 0.88, 0.9, 1.0)
    elif theme_name == "high-contrast":
        colors = style.colors
        colors[imgui.COLOR_TEXT] = (1.0, 1.0, 1.0, 1.0)
        colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.0, 0.0, 0.0, 1.0)
        colors[imgui.COLOR_CHILD_BACKGROUND] = (0.06, 0.06, 0.06, 1.0)
        colors[imgui.COLOR_FRAME_BACKGROUND] = (0.15, 0.15, 0.15, 1.0)
        colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.25, 0.25, 0.25, 1.0)
        colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = (0.35, 0.35, 0.35, 1.0)
        colors[imgui.COLOR_BUTTON] = (1.0, 0.8, 0.2, 1.0)
        colors[imgui.COLOR_BUTTON_HOVERED] = (1.0, 0.9, 0.3, 1.0)
        colors[imgui.COLOR_BUTTON_ACTIVE] = (0.95, 0.75, 0.15, 1.0)
        colors[imgui.COLOR_HEADER] = (1.0, 0.8, 0.2, 1.0)
        colors[imgui.COLOR_HEADER_HOVERED] = (1.0, 0.9, 0.3, 1.0)
        colors[imgui.COLOR_HEADER_ACTIVE] = (0.95, 0.75, 0.15, 1.0)
        colors[imgui.COLOR_SEPARATOR] = (1.0, 1.0, 1.0, 1.0)
        colors[imgui.COLOR_TAB] = (0.08, 0.08, 0.08, 1.0)
        colors[imgui.COLOR_TAB_ACTIVE] = (1.0, 0.8, 0.2, 1.0)
        colors[imgui.COLOR_TAB_HOVERED] = (1.0, 0.9, 0.3, 1.0)
        colors[imgui.COLOR_TITLE_BACKGROUND] = (0.05, 0.05, 0.05, 1.0)
        colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = (0.05, 0.05, 0.05, 1.0)
    else:
        # Dark default
        colors = style.colors
        colors[imgui.COLOR_TEXT] = (0.88, 0.89, 0.9, 1.0)
        colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.12, 0.12, 0.13, 1.0)
        colors[imgui.COLOR_CHILD_BACKGROUND] = (0.16, 0.16, 0.17, 1.0)
        colors[imgui.COLOR_FRAME_BACKGROUND] = (0.21, 0.21, 0.22, 1.0)
        colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.26, 0.26, 0.28, 1.0)
        colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = (0.3, 0.3, 0.33, 1.0)
        colors[imgui.COLOR_BUTTON] = (0.24, 0.24, 0.26, 1.0)
        colors[imgui.COLOR_BUTTON_HOVERED] = (0.3, 0.3, 0.34, 1.0)
        colors[imgui.COLOR_BUTTON_ACTIVE] = (0.34, 0.36, 0.4, 1.0)
        colors[imgui.COLOR_HEADER] = (0.2, 0.24, 0.3, 0.8)
        colors[imgui.COLOR_HEADER_HOVERED] = (0.24, 0.28, 0.34, 0.9)
        colors[imgui.COLOR_HEADER_ACTIVE] = (0.26, 0.3, 0.36, 1.0)
        colors[imgui.COLOR_SEPARATOR] = (0.2, 0.2, 0.22, 1.0)
        colors[imgui.COLOR_TAB] = (0.18, 0.18, 0.2, 1.0)
        colors[imgui.COLOR_TAB_ACTIVE] = (0.3, 0.35, 0.42, 1.0)
        colors[imgui.COLOR_TAB_HOVERED] = (0.34, 0.4, 0.48, 1.0)
        colors[imgui.COLOR_TITLE_BACKGROUND] = (0.14, 0.14, 0.15, 1.0)
        colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = (0.14, 0.14, 0.15, 1.0)
