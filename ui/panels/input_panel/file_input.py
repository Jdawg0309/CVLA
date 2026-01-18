"""
File input widget for the input panel.

Provides file path entry for matrix or image files.
"""

import imgui
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.app_state import AppState

from state.actions.input_panel_actions import (
    SetFilePath, ClearFilePath, CreateTensorFromFileInput
)
from state.actions.image_actions import LoadImage
from state.selectors import get_next_color


class FileInputWidget:
    """Widget for file-based tensor input."""

    def __init__(self):
        self._path_buffer = ""
        self._label_buffer = ""

    def render(self, state: "AppState", dispatch, width: float, file_type: str):
        """Render the file input widget for the selected file type."""
        labels = {
            "json": "JSON file path:",
            "csv": "CSV file path:",
            "excel": "Excel file path:",
            "image": "Image file path:",
        }
        button_labels = {
            "json": "Create Matrix",
            "csv": "Create Matrix",
            "excel": "Create Matrix",
            "image": "Load Image",
        }
        is_image = file_type == "image"

        imgui.text(labels.get(file_type, "File path:"))

        imgui.push_item_width(width - 20)
        changed, self._path_buffer = imgui.input_text(
            "##file_path",
            state.input_file_path or self._path_buffer,
            1024
        )
        imgui.pop_item_width()

        if changed:
            dispatch(SetFilePath(path=self._path_buffer))

        # File status
        file_path = state.input_file_path or self._path_buffer
        if file_path:
            if os.path.exists(file_path):
                imgui.text_colored("File found", 0.4, 0.8, 0.4, 1.0)
            else:
                imgui.text_colored("File not found", 0.8, 0.4, 0.4, 1.0)

        imgui.spacing()

        if not is_image:
            # Label input (matrix files)
            imgui.text("Label (optional):")
            imgui.same_line()
            imgui.push_item_width(width - 140)
            _, self._label_buffer = imgui.input_text(
                "##matrix_label",
                self._label_buffer,
                256
            )
            imgui.pop_item_width()
            imgui.spacing()

        # Load/create button
        can_load = file_path and os.path.exists(file_path)
        if not can_load:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)

        if imgui.button(button_labels.get(file_type, "Load"), width - 20, 30):
            if can_load:
                if is_image:
                    dispatch(LoadImage(path=file_path))
                else:
                    label = self._label_buffer.strip()
                    color, _ = get_next_color(state)
                    dispatch(CreateTensorFromFileInput(
                        file_type=file_type,
                        label=label,
                        color=color
                    ))
                self._path_buffer = ""
                self._label_buffer = ""
                dispatch(ClearFilePath())

        if not can_load:
            imgui.pop_style_var()

        # Supported formats
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        imgui.text_colored("Supported formats:", 0.6, 0.6, 0.6, 1.0)
        if file_type == "json":
            imgui.text_colored("  .json (array of rows)", 0.5, 0.5, 0.5, 1.0)
        elif file_type == "csv":
            imgui.text_colored("  .csv (numeric table)", 0.5, 0.5, 0.5, 1.0)
        elif file_type == "excel":
            imgui.text_colored("  .xlsx (first sheet)", 0.5, 0.5, 0.5, 1.0)
        else:
            imgui.text_colored("  PNG, JPG, BMP, TIFF", 0.5, 0.5, 0.5, 1.0)
