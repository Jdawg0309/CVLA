"""
File input widget for the input panel.

Provides file path entry for numeric or image tensor files.
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
            "json": "Create Tensor",
            "csv": "Create Tensor",
            "excel": "Create Tensor",
            "image": "Load Image Tensor",
        }
        is_image = file_type == "image"

        imgui.text(labels.get(file_type, "File path:"))

        input_width = max(120, width - 120)
        imgui.push_item_width(input_width)
        changed, self._path_buffer = imgui.input_text(
            "##file_path",
            state.input_file_path or self._path_buffer,
            1024
        )
        imgui.pop_item_width()
        clicked = imgui.is_item_clicked()

        imgui.same_line()
        if imgui.button("Browse", 80, 0):
            clicked = True

        if changed:
            dispatch(SetFilePath(path=self._path_buffer))
        if clicked:
            selected_path = self._open_file_dialog(file_type)
            if selected_path:
                self._path_buffer = selected_path
                dispatch(SetFilePath(path=selected_path))

        # File status
        file_path = state.input_file_path or self._path_buffer
        if file_path:
            if os.path.exists(file_path):
                imgui.text_colored("File found", 0.4, 0.8, 0.4, 1.0)
            else:
                imgui.text_colored("File not found", 0.8, 0.4, 0.4, 1.0)

        imgui.spacing()

        if not is_image:
            # Label input (numeric tensor files)
            imgui.text("Label (optional):")
            imgui.same_line()
            imgui.push_item_width(width - 140)
            _, self._label_buffer = imgui.input_text(
                "##tensor_label",
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
            imgui.text_colored("  [1,2,3] -> rank-1", 0.5, 0.5, 0.5, 1.0)
            imgui.text_colored("  [[1,2,3]] -> rank-2", 0.5, 0.5, 0.5, 1.0)
        elif file_type == "csv":
            imgui.text_colored("  .csv (numeric table)", 0.5, 0.5, 0.5, 1.0)
        elif file_type == "excel":
            imgui.text_colored("  .xlsx (first sheet)", 0.5, 0.5, 0.5, 1.0)
        else:
            imgui.text_colored("  PNG, JPG, BMP, TIFF", 0.5, 0.5, 0.5, 1.0)

    def _open_file_dialog(self, file_type: str) -> str:
        """Open a native file picker and return the selected path."""
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception:
            return ""

        filetypes = [("All files", "*.*")]
        if file_type == "json":
            filetypes = [("JSON files", "*.json"), ("All files", "*.*")]
        elif file_type == "csv":
            filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        elif file_type == "excel":
            filetypes = [
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        elif file_type == "image":
            filetypes = [
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        path = filedialog.askopenfilename(filetypes=filetypes)
        try:
            root.destroy()
        except Exception:
            pass
        return path or ""
