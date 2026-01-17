"""Ribbon button component with icon, label, and optional dropdown."""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any, Callable
import imgui


@dataclass
class RibbonButton:
    """A single ribbon button with icon and label."""

    label: str  # Text label (can include newlines for multi-line)
    icon: str = ""  # Short text or symbol for icon area
    tooltip: str = ""
    width: int = 64
    height: int = 56
    action: Optional[Any] = None  # Action to dispatch on click
    dropdown_items: Optional[List[Tuple[str, Any]]] = None  # (label, action) pairs
    enabled: bool = True

    # For stateful buttons (toggles)
    is_toggle: bool = False
    is_active: bool = False


def render_ribbon_button(
    button: RibbonButton,
    dispatch: Optional[Callable] = None,
    state: Any = None,
) -> bool:
    """
    Render a ribbon button and return True if clicked.

    Args:
        button: The button configuration
        dispatch: Function to dispatch actions
        state: Current app state (for checking enabled conditions)

    Returns:
        True if the button was clicked
    """
    clicked = False
    button_id = f"##{button.label.replace(chr(10), '_')}"

    # Style for active/toggle state
    if button.is_toggle and button.is_active:
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.5, 0.7, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.35, 0.55, 0.75, 1.0)

    # Disabled state
    if not button.enabled:
        imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)

    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (4, 4))

    # Create button with icon and label
    display_text = button.icon
    if button.label:
        if button.icon:
            display_text += "\n" + button.label
        else:
            display_text = button.label

    if imgui.button(display_text + button_id, button.width, button.height):
        if button.enabled:
            if button.dropdown_items:
                imgui.open_popup(button_id + "_popup")
            elif button.action and dispatch:
                dispatch(button.action)
                clicked = True
            else:
                clicked = True

    imgui.pop_style_var()  # FRAME_PADDING

    if not button.enabled:
        imgui.pop_style_var()  # ALPHA

    if button.is_toggle and button.is_active:
        imgui.pop_style_color(2)

    # Dropdown popup
    if button.dropdown_items and imgui.begin_popup(button_id + "_popup"):
        for item_label, item_action in button.dropdown_items:
            if item_label == "-":
                imgui.separator()
            elif imgui.menu_item(item_label)[0]:
                if item_action and dispatch:
                    dispatch(item_action)
                clicked = True
        imgui.end_popup()

    # Tooltip
    if imgui.is_item_hovered() and button.tooltip:
        imgui.set_tooltip(button.tooltip)

    return clicked
