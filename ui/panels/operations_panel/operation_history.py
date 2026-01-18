"""
Operation history widget for the operations panel.

Shows a timeline of operations applied to the selected tensor.
"""

import imgui
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from state.app_state import AppState
    from state.models.tensor_model import TensorData


class OperationHistoryWidget:
    """Widget showing operation history for a tensor."""

    def __init__(self):
        self._selected_step_idx = -1
        self._filter_text = ""
        self._show_all_tensors = False

    def render(self, tensor: "TensorData", state: "AppState", dispatch, width: float):
        """Render operation history UI."""
        imgui.text("OPERATION HISTORY")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Options row
        _, self._show_all_tensors = imgui.checkbox(
            "Show all",
            self._show_all_tensors
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Show history for all tensors")

        imgui.spacing()

        # Filter input
        imgui.push_item_width(width - 20)
        _, self._filter_text = imgui.input_text_with_hint(
            "##history_filter",
            "Filter operations...",
            self._filter_text,
            256
        )
        imgui.pop_item_width()

        imgui.spacing()

        # History list
        history = self._get_history(tensor, state)

        if not history:
            imgui.text_colored("No operations yet", 0.5, 0.5, 0.5, 1.0)
            imgui.spacing()
            imgui.text_wrapped(
                "Operations will appear here as you apply them to tensors."
            )
            return

        # Filter history
        if self._filter_text:
            filter_lower = self._filter_text.lower()
            history = [h for h in history if filter_lower in h["name"].lower()]

        # History count
        imgui.text_disabled(f"{len(history)} operations")
        imgui.spacing()

        # Scrollable history list
        list_height = min(200, max(80, len(history) * 28))

        imgui.begin_child(
            "##history_list",
            width - 10,
            list_height,
            border=True,
            flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR
        )

        for i, entry in enumerate(history):
            self._render_history_entry(entry, i, width - 30, dispatch)

        imgui.end_child()

        imgui.spacing()

        # Summary section
        if tensor is not None and hasattr(tensor, 'history') and tensor.history:
            imgui.separator()
            imgui.spacing()
            imgui.text("Tensor History:")
            imgui.spacing()

            for i, op_name in enumerate(tensor.history):
                step_num = i + 1
                imgui.text_colored(f"{step_num}.", 0.4, 0.6, 0.8, 1.0)
                imgui.same_line()
                imgui.text(op_name)

    def _get_history(self, tensor: "TensorData", state: "AppState") -> List[dict]:
        """Get operation history from state."""
        history = []

        # Get global operation history
        global_history = getattr(state, 'operation_history', ())

        for record in global_history:
            # Filter by tensor if not showing all
            if not self._show_all_tensors and tensor is not None:
                input_ids = getattr(record, 'input_tensor_ids', ())
                output_id = getattr(record, 'output_tensor_id', None)
                if tensor.id not in input_ids and tensor.id != output_id:
                    continue

            history.append({
                "id": getattr(record, 'id', ''),
                "name": getattr(record, 'operation_name', 'Unknown'),
                "type": getattr(record, 'operation_type', ''),
                "description": getattr(record, 'description', ''),
                "timestamp": getattr(record, 'timestamp', 0),
                "reversible": getattr(record, 'reversible', True),
                "params": dict(getattr(record, 'parameters', ())),
            })

        # Also include tensor's own history (simple string list)
        if tensor is not None and hasattr(tensor, 'history'):
            for i, op_name in enumerate(tensor.history):
                # Only add if not already in global history
                already_exists = any(
                    h["name"] == op_name for h in history
                )
                if not already_exists:
                    history.append({
                        "id": f"tensor_{tensor.id}_{i}",
                        "name": op_name,
                        "type": "tensor_history",
                        "description": f"Applied to {tensor.label}",
                        "timestamp": 0,
                        "reversible": False,
                        "params": {},
                    })

        return history

    def _render_history_entry(self, entry: dict, index: int, width: float, dispatch):
        """Render a single history entry."""
        op_name = entry["name"]
        op_type = entry["type"]
        description = entry["description"]
        reversible = entry["reversible"]

        # Color based on operation type
        type_colors = {
            "vector_op": (0.2, 0.6, 0.8, 1.0),
            "matrix_op": (0.8, 0.6, 0.2, 1.0),
            "image_op": (0.6, 0.8, 0.2, 1.0),
            "transform": (0.8, 0.2, 0.6, 1.0),
            "tensor_history": (0.5, 0.5, 0.7, 1.0),
        }
        color = type_colors.get(op_type, (0.7, 0.7, 0.7, 1.0))

        # Selectable row
        is_selected = self._selected_step_idx == index
        clicked, _ = imgui.selectable(
            f"##history_{index}",
            is_selected,
            imgui.SELECTABLE_SPAN_ALL_COLUMNS,
            (width, 24)
        )

        if clicked:
            self._selected_step_idx = index

        # Draw content on same line
        imgui.same_line(10)

        # Step number
        imgui.text_colored(f"{index + 1:2d}", 0.5, 0.5, 0.5, 1.0)
        imgui.same_line()

        # Type indicator (colored dot)
        imgui.text_colored("●", *color)
        imgui.same_line()

        # Operation name
        imgui.text(op_name)

        # Tooltip with details
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.text(op_name)
            imgui.separator()
            if description:
                imgui.text_wrapped(description)
            if entry["params"]:
                imgui.spacing()
                imgui.text_disabled("Parameters:")
                for k, v in entry["params"].items():
                    imgui.text(f"  {k}: {v}")
            if not reversible:
                imgui.spacing()
                imgui.text_colored("Not reversible", 0.8, 0.4, 0.4, 1.0)
            imgui.end_tooltip()

        # Context menu on right-click
        if imgui.begin_popup_context_item(f"##history_ctx_{index}"):
            if reversible:
                if imgui.menu_item("Undo to here")[0]:
                    # Would dispatch undo action
                    pass
            if imgui.menu_item("Copy operation")[0]:
                # Copy to clipboard
                pass
            if imgui.menu_item("Re-apply")[0]:
                # Re-apply the same operation
                pass
            imgui.end_popup()
