"""Base class for ribbon tabs."""

from dataclasses import dataclass
from typing import List, Optional, Callable, Any
import imgui

from ui.ribbon.ribbon_group import RibbonGroup, render_ribbon_group, render_ribbon_separator


class RibbonTab:
    """Base class for ribbon tabs containing groups of buttons."""

    def __init__(self, groups: Optional[List[RibbonGroup]] = None):
        self._groups = groups or []

    @property
    def groups(self) -> List[RibbonGroup]:
        return self._groups

    def render(
        self,
        state: Any,
        dispatch: Optional[Callable] = None,
        camera: Any = None,
        view_config: Any = None,
    ) -> None:
        """
        Render the tab content (all groups).

        Override this method in subclasses to customize rendering
        or dynamically update button states.
        """
        self._render_groups(state, dispatch)

    def _render_groups(
        self,
        state: Any,
        dispatch: Optional[Callable] = None,
    ) -> None:
        """Render all groups with separators between them."""
        for i, group in enumerate(self._groups):
            render_ribbon_group(group, dispatch, state)

            if i < len(self._groups) - 1:
                imgui.same_line()
                render_ribbon_separator()
                imgui.same_line()

    def update_button_states(self, state: Any) -> None:
        """
        Update button enabled/active states based on current state.

        Override in subclasses to update button states dynamically.
        """
        pass
