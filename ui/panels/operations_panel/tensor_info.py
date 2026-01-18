"""
Tensor info widget for the operations panel.

Displays detailed information about the selected tensor.
"""

import imgui
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.app_state import AppState
    from state.models.tensor_model import TensorData

from state.actions.tensor_actions import UpdateTensor
from state.selectors import get_tensor_stats, get_tensor_magnitude, get_tensor_norm


class TensorInfoWidget:
    """Widget displaying selected tensor details."""

    def __init__(self):
        self._label_buffer = ""
        self._editing_label = False

    def render(self, tensor: "TensorData", state: "AppState", dispatch, width: float):
        """Render tensor information."""
        if tensor is None:
            imgui.text_colored("No tensor selected", 0.5, 0.5, 0.5, 1.0)
            imgui.spacing()
            imgui.text_colored("Select a tensor from the list", 0.5, 0.5, 0.5, 1.0)
            imgui.text_colored("to view details and operations.", 0.5, 0.5, 0.5, 1.0)
            return

        # Header with type icon
        tensor_type = tensor.tensor_type
        type_colors = {
            'vector': (0.4, 0.7, 1.0),
            'matrix': (0.4, 1.0, 0.7),
            'image': (1.0, 0.7, 0.4),
        }
        color = type_colors.get(tensor_type, (0.8, 0.8, 0.8))

        imgui.text_colored(tensor_type.upper(), *color, 1.0)
        imgui.same_line()

        # Editable label
        if self._editing_label:
            imgui.push_item_width(width - 100)
            changed, new_label = imgui.input_text(
                "##edit_label",
                self._label_buffer,
                256,
                imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
            )
            imgui.pop_item_width()
            if changed or imgui.is_key_pressed(imgui.KEY_ESCAPE):
                if changed and self._label_buffer.strip():
                    dispatch(UpdateTensor(id=tensor.id, label=self._label_buffer.strip()))
                self._editing_label = False
        else:
            imgui.text(tensor.label)
            imgui.same_line()
            if imgui.small_button("Edit"):
                self._label_buffer = tensor.label
                self._editing_label = True

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Basic properties
        imgui.text("Shape:")
        imgui.same_line(80)
        imgui.text(self._format_shape(tensor))

        imgui.text("Type:")
        imgui.same_line(80)
        imgui.text(str(tensor.dtype.value))

        imgui.text("Visible:")
        imgui.same_line(80)
        _, new_visible = imgui.checkbox("##visible", tensor.visible)
        if _ != tensor.visible:
            dispatch(UpdateTensor(id=tensor.id, visible=new_visible))

        # Color picker (for vectors/matrices)
        if tensor_type in ('vector', 'matrix'):
            imgui.text("Color:")
            imgui.same_line(80)
            _, new_color = imgui.color_edit3(
                "##color",
                tensor.color[0], tensor.color[1], tensor.color[2]
            )
            if new_color != tensor.color:
                dispatch(UpdateTensor(id=tensor.id, color=new_color))

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Type-specific info
        if tensor.is_vector:
            self._render_vector_info(tensor)
        elif tensor.is_matrix:
            self._render_matrix_info(tensor)
        elif tensor.is_image:
            self._render_image_info(tensor)

        # History
        if tensor.history:
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.text("History:")
            for op in tensor.history[-5:]:  # Show last 5 operations
                imgui.bullet_text(op)

    def _format_shape(self, tensor: "TensorData") -> str:
        """Format tensor shape for display."""
        return " x ".join(str(d) for d in tensor.shape)

    def _render_vector_info(self, tensor: "TensorData"):
        """Render vector-specific information."""
        imgui.text("Coordinates:")
        coords = tensor.coords
        for i, c in enumerate(coords):
            axis = ['x', 'y', 'z', 'w'][i] if i < 4 else f"[{i}]"
            imgui.text(f"  {axis}: {c:.4f}")

        imgui.spacing()
        magnitude = get_tensor_magnitude(tensor)
        imgui.text(f"Magnitude: {magnitude:.4f}")

    def _render_matrix_info(self, tensor: "TensorData"):
        """Render matrix-specific information."""
        imgui.text("Values:")

        # Show matrix values in a compact grid
        values = tensor.values
        rows = min(len(values), 5)  # Show at most 5 rows
        cols = min(len(values[0]) if values else 0, 5)  # Show at most 5 cols

        for r in range(rows):
            row_str = "  "
            for c in range(cols):
                row_str += f"{values[r][c]:8.3f}"
            if cols < len(values[0]):
                row_str += " ..."
            imgui.text(row_str)

        if rows < len(values):
            imgui.text("  ...")

        imgui.spacing()
        norm = get_tensor_norm(tensor)
        imgui.text(f"Frobenius Norm: {norm:.4f}")

    def _render_image_info(self, tensor: "TensorData"):
        """Render image-specific information."""
        imgui.text(f"Dimensions: {tensor.height} x {tensor.width}")
        imgui.text(f"Channels: {tensor.channels}")
        imgui.text(f"Grayscale: {'Yes' if tensor.is_grayscale else 'No'}")

        imgui.spacing()

        # Statistics
        stats = get_tensor_stats(tensor)
        imgui.text("Statistics:")
        imgui.text(f"  Min: {stats[0]:.4f}")
        imgui.text(f"  Max: {stats[1]:.4f}")
        imgui.text(f"  Mean: {stats[2]:.4f}")
        imgui.text(f"  Std: {stats[3]:.4f}")
