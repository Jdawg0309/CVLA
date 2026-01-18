"""
Tensor list widget for the input panel.

Displays all tensors in the scene with selection capability.
"""

import imgui
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.app_state import AppState

from state.actions.tensor_actions import (
    SelectTensor, DeselectTensor, DeleteTensor, UpdateTensor
)
from state.models.tensor_model import TensorDType


class TensorListWidget:
    """Widget displaying all tensors in the scene."""

    TYPE_ICONS = {
        'r1': '[R1]',
        'r2': '[R2]',
        'r3': '[R3+]',
    }

    TYPE_COLORS = {
        'r1': (0.4, 0.7, 1.0, 1.0),   # Rank-1
        'r2': (0.4, 1.0, 0.7, 1.0),   # Rank-2
        'r3': (0.7, 0.7, 0.8, 1.0),   # Rank-3+
    }

    def __init__(self):
        self._filter_type = "all"  # "all", "r1", "r2", "r3"
        self._search_text = ""

    def render(self, state: "AppState", dispatch, width: float, height: float):
        """Render the tensor list widget."""
        imgui.text("Tensors")

        # Filter controls
        imgui.same_line(width - 150)
        imgui.push_item_width(60)
        filter_items = ["All", "R1", "R2", "R3+"]
        filter_map = ["all", "r1", "r2", "r3"]
        current_idx = filter_map.index(self._filter_type) if self._filter_type in filter_map else 0
        changed, new_idx = imgui.combo("##filter", current_idx, filter_items)
        if changed:
            self._filter_type = filter_map[new_idx]
        imgui.pop_item_width()

        imgui.same_line()
        imgui.push_item_width(80)
        _, self._search_text = imgui.input_text("##search", self._search_text, 64)
        imgui.pop_item_width()

        imgui.spacing()
        selected_id = state.selected_tensor_id
        selected_tensor = None
        if selected_id:
            for t in state.tensors:
                if t.id == selected_id:
                    selected_tensor = t
                    break
        if selected_tensor is not None:
            shape_str = self._format_feedback_shape(selected_tensor)
            imgui.text_colored(
                f"Parsed as rank-{selected_tensor.rank} tensor, shape {shape_str}",
                0.5, 0.7, 0.5, 1.0
            )
            imgui.spacing()

        # Tensor list
        list_height = height - 60
        imgui.begin_child("tensor_list", width - 10, list_height, border=True)

        tensors = state.tensors
        selected_id = state.selected_tensor_id

        # Filter tensors
        filtered = self._filter_tensors(tensors)

        if not filtered:
            imgui.text_colored("No tensors", 0.5, 0.5, 0.5, 1.0)
        else:
            for tensor in filtered:
                self._render_tensor_item(tensor, selected_id, dispatch, width - 30)

        imgui.end_child()

        # Action buttons
        imgui.spacing()
        if selected_id:
            button_width = (width - 30) / 2
            if imgui.button("Deselect", button_width, 0):
                dispatch(DeselectTensor())
            imgui.same_line()
            if imgui.button("Delete", button_width, 0):
                dispatch(DeleteTensor(id=selected_id))

    def _filter_tensors(self, tensors):
        """Filter tensors based on type and search text."""
        result = []
        for t in tensors:
            # Type filter
            if self._filter_type != "all":
                if self._filter_type == "r1" and t.rank != 1:
                    continue
                if self._filter_type == "r2" and t.rank != 2:
                    continue
                if self._filter_type == "r3" and t.rank < 3:
                    continue

            # Search filter
            if self._search_text:
                search_lower = self._search_text.lower()
                if search_lower not in t.label.lower():
                    continue

            result.append(t)
        return result

    def _render_tensor_item(self, tensor, selected_id, dispatch, width):
        """Render a single tensor item in the list."""
        is_selected = tensor.id == selected_id
        if tensor.rank == 1:
            tensor_type = "r1"
        elif tensor.rank == 2:
            tensor_type = "r2"
        else:
            tensor_type = "r3"

        # Item styling
        icon = self.TYPE_ICONS.get(tensor_type, '[?]')
        color = self.TYPE_COLORS.get(tensor_type, (0.8, 0.8, 0.8, 1.0))

        # Selection background
        if is_selected:
            imgui.push_style_color(imgui.COLOR_HEADER, 0.3, 0.5, 0.7, 1.0)
            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0.35, 0.55, 0.75, 1.0)
            imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, 0.4, 0.6, 0.8, 1.0)

        # Render selectable item
        clicked, _ = imgui.selectable(
            f"##tensor_{tensor.id}",
            is_selected,
            width=width,
            height=24
        )

        if is_selected:
            imgui.pop_style_color(3)

        if clicked:
            if is_selected:
                dispatch(DeselectTensor())
            else:
                dispatch(SelectTensor(id=tensor.id))

        # Draw content on top
        imgui.same_line(10)
        imgui.text_colored(icon, *color)
        imgui.same_line()
        imgui.text(tensor.label)
        if tensor.dtype in (TensorDType.IMAGE_RGB, TensorDType.IMAGE_GRAYSCALE):
            imgui.same_line()
            imgui.text_colored(f"[{tensor.dtype.value}]", 0.6, 0.6, 0.6, 1.0)

        # Shape info
        imgui.same_line(width - 80)
        shape_str = self._format_shape(tensor)
        imgui.text_colored(shape_str, 0.5, 0.5, 0.5, 1.0)

        # Visibility toggle
        imgui.same_line(width - 20)
        visible_icon = "O" if tensor.visible else "-"
        if imgui.small_button(f"{visible_icon}##vis_{tensor.id}"):
            dispatch(UpdateTensor(id=tensor.id, visible=not tensor.visible))

    def _format_shape(self, tensor) -> str:
        """Format tensor shape for display."""
        if tensor.dtype in (TensorDType.IMAGE_RGB, TensorDType.IMAGE_GRAYSCALE):
            if len(tensor.shape) == 2:
                return f"{tensor.shape[0]}x{tensor.shape[1]}"
            return f"{tensor.shape[0]}x{tensor.shape[1]}x{tensor.shape[2]}"
        if tensor.rank == 1:
            return f"({len(tensor.data)},)"
        elif tensor.rank == 2:
            return f"({tensor.rows}x{tensor.cols})"
        return str(tensor.shape)

    def _format_feedback_shape(self, tensor) -> str:
        """Format shape for creation feedback."""
        if tensor.rank == 1:
            return f"({tensor.shape[0]},)"
        if tensor.rank == 2:
            return f"({tensor.shape[0]}x{tensor.shape[1]})"
        return str(tensor.shape)
