"""Ribbon group component - a group of related buttons with a label."""

from dataclasses import dataclass
from typing import List, Optional, Callable, Any
import imgui

from ui.ribbon.ribbon_button import RibbonButton, render_ribbon_button


@dataclass
class RibbonGroup:
    """A group of related buttons in the ribbon with a group label."""

    name: str  # Group label shown below buttons
    buttons: List[RibbonButton] = None

    def __post_init__(self):
        if self.buttons is None:
            self.buttons = []


def render_ribbon_group(
    group: RibbonGroup,
    dispatch: Optional[Callable] = None,
    state: Any = None,
) -> None:
    """
    Render a ribbon group with its buttons and label.

    Args:
        group: The group configuration
        dispatch: Function to dispatch actions
        state: Current app state
    """
    imgui.begin_group()

    # Render buttons in a row
    for i, button in enumerate(group.buttons):
        render_ribbon_button(button, dispatch, state)
        if i < len(group.buttons) - 1:
            imgui.same_line()

    # Group label below buttons
    imgui.spacing()

    # Center the label
    text_width = imgui.calc_text_size(group.name).x
    total_width = sum(b.width for b in group.buttons) + (len(group.buttons) - 1) * 4
    offset = (total_width - text_width) / 2
    if offset > 0:
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + offset)

    imgui.text_disabled(group.name)

    imgui.end_group()


def render_ribbon_separator() -> None:
    """Render a vertical separator between ribbon groups."""
    draw_list = imgui.get_window_draw_list()
    pos = imgui.get_cursor_screen_pos()

    # Draw a vertical line
    color = imgui.get_color_u32_rgba(0.3, 0.3, 0.3, 1.0)
    draw_list.add_line(
        pos[0] + 8, pos[1],
        pos[0] + 8, pos[1] + 70,
        color,
        1.0
    )

    # Add spacing
    imgui.dummy(16, 1)
