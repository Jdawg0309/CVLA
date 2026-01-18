"""
File input widget for the input panel.

Provides file path entry and sample image generation.
"""

import imgui
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.app_state import AppState

from state.actions.input_panel_actions import (
    SetFilePath, ClearFilePath, CreateTensorFromFileInput
)
from state.actions.tensor_actions import AddImageTensor


class FileInputWidget:
    """Widget for file-based tensor input."""

    SAMPLE_PATTERNS = [
        ("checkerboard", "Checkerboard pattern"),
        ("gradient", "Horizontal gradient"),
        ("circle", "Circle pattern"),
        ("noise", "Random noise"),
        ("stripes", "Vertical stripes"),
    ]

    SAMPLE_SIZES = [32, 64, 128, 256, 512]

    def __init__(self):
        self._path_buffer = ""
        self._label_buffer = ""
        self._selected_pattern = 0
        self._selected_size = 2  # Default to 128

    def render(self, state: "AppState", dispatch, width: float):
        """Render the file input widget."""
        # File path section
        imgui.text("Image file path:")

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

        # Label input
        imgui.text("Label (optional):")
        imgui.same_line()
        imgui.push_item_width(width - 140)
        _, self._label_buffer = imgui.input_text(
            "##image_label",
            self._label_buffer,
            256
        )
        imgui.pop_item_width()

        imgui.spacing()

        # Load button
        can_load = file_path and os.path.exists(file_path)
        if not can_load:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)

        if imgui.button("Load Image", width - 20, 30):
            if can_load:
                dispatch(CreateTensorFromFileInput(
                    label=self._label_buffer.strip()
                ))
                self._path_buffer = ""
                self._label_buffer = ""
                dispatch(ClearFilePath())

        if not can_load:
            imgui.pop_style_var()

        # Sample image section
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        imgui.text("Or create sample image:")
        imgui.spacing()

        # Pattern selector
        imgui.text("Pattern:")
        imgui.same_line(80)
        imgui.push_item_width(width - 100)
        pattern_names = [p[1] for p in self.SAMPLE_PATTERNS]
        _, self._selected_pattern = imgui.combo(
            "##pattern",
            self._selected_pattern,
            pattern_names
        )
        imgui.pop_item_width()

        # Size selector
        imgui.text("Size:")
        imgui.same_line(80)
        imgui.push_item_width(width - 100)
        size_names = [f"{s}x{s}" for s in self.SAMPLE_SIZES]
        _, self._selected_size = imgui.combo(
            "##size",
            self._selected_size,
            size_names
        )
        imgui.pop_item_width()

        imgui.spacing()

        # Create sample button
        if imgui.button("Create Sample", width - 20, 30):
            pattern = self.SAMPLE_PATTERNS[self._selected_pattern][0]
            size = self.SAMPLE_SIZES[self._selected_size]
            dispatch(AddImageTensor(
                source="sample",
                pattern=pattern,
                size=size,
                label=f"{pattern}_{size}"
            ))

        # Supported formats
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        imgui.text_colored("Supported formats:", 0.6, 0.6, 0.6, 1.0)
        imgui.text_colored("  PNG, JPG, BMP, TIFF", 0.5, 0.5, 0.5, 1.0)
        imgui.text_colored("  CSV (numeric data)", 0.5, 0.5, 0.5, 1.0)
        imgui.text_colored("  NPY (numpy arrays)", 0.5, 0.5, 0.5, 1.0)
