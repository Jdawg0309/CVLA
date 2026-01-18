"""
Operation preview widget for the operations panel.

Shows before/after visualization of pending operations.
"""

import imgui
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from state.app_state import AppState
    from state.models.tensor_model import TensorData

from state.actions.tensor_actions import CancelPreview, ConfirmPreview


class OperationPreviewWidget:
    """Widget showing operation preview with before/after views."""

    def __init__(self):
        self._show_diff = False
        self._split_view = True

    def render(self, tensor: "TensorData", state: "AppState", dispatch, width: float):
        """Render operation preview UI."""
        pending_op = getattr(state, 'pending_operation', None)
        preview_result = getattr(state, 'operation_preview_tensor', None)

        if not pending_op:
            imgui.text_colored("No pending operation", 0.5, 0.5, 0.5, 1.0)
            return

        imgui.text("OPERATION PREVIEW")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Operation name and parameters
        imgui.text(f"Operation: ")
        imgui.same_line()
        imgui.text_colored(pending_op, 0.4, 0.8, 0.4, 1.0)

        params = getattr(state, 'pending_operation_params', ())
        if params:
            imgui.spacing()
            imgui.text_disabled("Parameters:")
            for key, value in params:
                imgui.text(f"  {key}: {value}")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # View options
        _, self._split_view = imgui.checkbox("Split view", self._split_view)
        imgui.same_line()
        _, self._show_diff = imgui.checkbox("Show diff", self._show_diff)

        imgui.spacing()

        # Preview visualization
        self._render_preview_visualization(
            tensor, preview_result, state, width
        )

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Action buttons
        half_width = (width - 30) / 2

        imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.5, 0.2, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.6, 0.3, 1.0)
        if imgui.button("Apply", half_width, 30):
            dispatch(ConfirmPreview())
        imgui.pop_style_color(2)

        imgui.same_line()

        imgui.push_style_color(imgui.COLOR_BUTTON, 0.5, 0.2, 0.2, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.6, 0.3, 0.3, 1.0)
        if imgui.button("Cancel", half_width, 30):
            dispatch(CancelPreview())
        imgui.pop_style_color(2)

    def _render_preview_visualization(
        self,
        original: "TensorData",
        preview: Optional["TensorData"],
        state: "AppState",
        width: float
    ):
        """Render the before/after visualization."""
        preview_height = 150

        if original is None:
            imgui.text_colored("No tensor selected", 0.5, 0.5, 0.5, 1.0)
            return

        if preview is None:
            imgui.text_colored("Computing preview...", 0.7, 0.7, 0.3, 1.0)
            return

        if self._split_view:
            # Side by side view
            half = (width - 30) / 2

            imgui.begin_child("##preview_before", half, preview_height, border=True)
            imgui.text_colored("BEFORE", 0.7, 0.7, 0.7, 1.0)
            imgui.separator()
            self._render_tensor_preview(original, half - 10)
            imgui.end_child()

            imgui.same_line()

            imgui.begin_child("##preview_after", half, preview_height, border=True)
            imgui.text_colored("AFTER", 0.4, 0.8, 0.4, 1.0)
            imgui.separator()
            self._render_tensor_preview(preview, half - 10)
            imgui.end_child()
        else:
            # Single view (after only)
            imgui.begin_child("##preview_result", width - 20, preview_height, border=True)
            imgui.text_colored("RESULT", 0.4, 0.8, 0.4, 1.0)
            imgui.separator()
            self._render_tensor_preview(preview, width - 30)
            imgui.end_child()

        if self._show_diff and original is not None and preview is not None:
            imgui.spacing()
            self._render_diff_info(original, preview)

    def _render_tensor_preview(self, tensor: "TensorData", width: float):
        """Render a compact preview of a tensor."""
        if tensor is None:
            imgui.text_disabled("(no data)")
            return

        tensor_type = tensor.tensor_type

        if tensor_type == "vector":
            # Show vector coordinates
            coords = tensor.coords
            if len(coords) <= 4:
                coord_str = ", ".join(f"{c:.3f}" for c in coords)
                imgui.text(f"[{coord_str}]")
            else:
                coord_str = ", ".join(f"{c:.2f}" for c in coords[:3])
                imgui.text(f"[{coord_str}, ...]")

            # Show magnitude
            magnitude = sum(c * c for c in coords) ** 0.5
            imgui.text_colored(f"Magnitude: {magnitude:.4f}", 0.6, 0.6, 0.6, 1.0)

        elif tensor_type == "matrix":
            # Show matrix dimensions and sample values
            rows, cols = tensor.shape[0], tensor.shape[1]
            imgui.text(f"Shape: {rows}x{cols}")

            # Show first few values
            if rows > 0 and cols > 0:
                data = tensor.data
                for i in range(min(3, rows)):
                    row = data[i] if i < len(data) else ()
                    vals = ", ".join(f"{v:.2f}" for v in row[:min(4, cols)])
                    if cols > 4:
                        vals += ", ..."
                    imgui.text_colored(f"[{vals}]", 0.7, 0.7, 0.7, 1.0)
                if rows > 3:
                    imgui.text_colored("...", 0.5, 0.5, 0.5, 1.0)

        elif tensor_type == "image":
            # Show image dimensions and stats
            h, w = tensor.shape[0], tensor.shape[1]
            channels = tensor.shape[2] if len(tensor.shape) > 2 else 1
            imgui.text(f"Size: {w}x{h}")
            ch_str = "RGB" if channels == 3 else ("RGBA" if channels == 4 else "Gray")
            imgui.text(f"Format: {ch_str}")

            # Would show thumbnail here if we had image rendering
            imgui.text_colored("(image preview)", 0.5, 0.5, 0.5, 1.0)

    def _render_diff_info(self, original: "TensorData", preview: "TensorData"):
        """Show difference statistics between original and preview."""
        imgui.text("Changes:")

        # Shape change
        if original.shape != preview.shape:
            imgui.text_colored(
                f"Shape: {original.shape} -> {preview.shape}",
                0.8, 0.6, 0.2, 1.0
            )

        # For numeric tensors, compute difference stats
        if original.tensor_type == "vector" and preview.tensor_type == "vector":
            orig_coords = original.coords
            prev_coords = preview.coords
            if len(orig_coords) == len(prev_coords):
                diff = [abs(a - b) for a, b in zip(orig_coords, prev_coords)]
                max_diff = max(diff) if diff else 0
                avg_diff = sum(diff) / len(diff) if diff else 0
                imgui.text(f"Max change: {max_diff:.4f}")
                imgui.text(f"Avg change: {avg_diff:.4f}")
