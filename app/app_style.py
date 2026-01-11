"""
ImGui style configuration.
"""

import imgui


def setup_imgui_style(self):
    style = imgui.get_style()
    style.window_rounding = 6.0
    style.frame_rounding = 4.0
    style.scrollbar_rounding = 6.0
    style.window_border_size = 1.0
    style.frame_border_size = 1.0
    style.item_spacing = (8.0, 6.0)
    style.window_padding = (10.0, 8.0)

    colors = style.colors
    colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.06, 0.06, 0.07, 0.98)
    colors[imgui.COLOR_FRAME_BACKGROUND] = (0.18, 0.18, 0.22, 1.00)
    colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.24, 0.24, 0.28, 1.00)
    colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = (0.28, 0.28, 0.32, 1.00)
    colors[imgui.COLOR_BUTTON] = (0.22, 0.45, 0.82, 0.9)
    colors[imgui.COLOR_BUTTON_HOVERED] = (0.26, 0.55, 0.95, 1.0)
    colors[imgui.COLOR_BUTTON_ACTIVE] = (0.18, 0.38, 0.72, 1.0)
    colors[imgui.COLOR_HEADER] = (0.20, 0.20, 0.24, 0.95)
    colors[imgui.COLOR_HEADER_HOVERED] = (0.26, 0.26, 0.30, 1.0)
    colors[imgui.COLOR_HEADER_ACTIVE] = (0.30, 0.30, 0.34, 1.0)
