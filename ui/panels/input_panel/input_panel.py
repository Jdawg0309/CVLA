"""
Main input panel orchestrator.

Combines manual and file input widgets with the tensor list.
"""

import imgui
import os
import re
from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    from state.app_state import AppState

from state.actions.image_actions import LoadImage
from state.actions.input_panel_actions import (
    ClearFilePath,
    ClearTextInput,
    CreateTensorFromFileInput,
    CreateTensorFromTextInput,
    SetFilePath,
    SetInputMethod,
    SetTextInput,
)
from state.actions.tensor_actions import (
    DeleteTensor,
    DeselectTensor,
    SelectTensor,
    UpdateTensor,
)
from state.models.tensor_model import TensorDType
from state.selectors import get_next_color, get_tensor_count

# ImGui flag constants
_WINDOW_NO_TITLE_BAR = getattr(imgui, "WINDOW_NO_TITLE_BAR", 0)
_WINDOW_NO_RESIZE = getattr(imgui, "WINDOW_NO_RESIZE", 0)
_WINDOW_NO_MOVE = getattr(imgui, "WINDOW_NO_MOVE", 0)
_WINDOW_NO_COLLAPSE = getattr(imgui, "WINDOW_NO_COLLAPSE", 0)
_WINDOW_ALWAYS_VERTICAL_SCROLLBAR = getattr(
    imgui, "WINDOW_ALWAYS_VERTICAL_SCROLLBAR", 0
)


def parse_input(content: str) -> Tuple[str, Optional[Union[Tuple, Tuple[Tuple, ...]]]]:
    """
    Parse text input and determine its type.

    Args:
        content: Raw text input

    Returns:
        Tuple of (type_name, parsed_data) where type_name is
        "vector", "matrix", or "" (unparseable), and parsed_data
        is the parsed tuple data or None.
    """
    content = content.strip()
    if not content:
        return ("", None)

    # Rank-1 tensors require explicit brackets: [1, 2, 3]
    if _is_rank1_bracketed(content):
        data = parse_vector(content)
        if data is not None:
            return ("vector", data)
        return ("", None)

    # Everything else parses as rank-2 if valid
    data = parse_matrix(content)
    if data is not None:
        return ("matrix", data)

    return ("", None)


def parse_vector(content: str) -> Optional[Tuple[float, ...]]:
    """
    Parse text as a vector.

    Supported formats:
    - "[1, 2, 3]"

    Returns:
        Tuple of floats or None if parsing fails.
    """
    content = content.strip()

    # Require single bracket pair
    if not _is_rank1_bracketed(content):
        return None

    # Remove brackets
    content = content.strip()[1:-1].strip()

    # Split by comma or whitespace
    if "," in content:
        parts = content.split(",")
    else:
        parts = content.split()

    try:
        values = tuple(float(p.strip()) for p in parts if p.strip())
        return values if values else None
    except ValueError:
        return None


def parse_matrix(content: str) -> Optional[Tuple[Tuple[float, ...], ...]]:
    """
    Parse text as a matrix.

    Supported formats:
    - "1, 2; 3, 4" (semicolon-separated rows)
    - "[[1, 2], [3, 4]]" (Python-style nested lists)
    - Multi-line input (each line is a row)

    Returns:
        Nested tuple of floats or None if parsing fails.
    """
    content = content.strip()

    # Handle Python-style nested lists
    if content.startswith("[["):
        return _parse_python_matrix(content)

    # Handle semicolon-separated rows
    if ";" in content:
        return _parse_semicolon_matrix(content)

    # Handle newline-separated rows
    if "\n" in content:
        return _parse_newline_matrix(content)

    # Handle single-row rank-2 input
    return _parse_single_row_matrix(content)


def _parse_python_matrix(content: str) -> Optional[Tuple[Tuple[float, ...], ...]]:
    """Parse Python-style nested list format."""
    try:
        # Replace brackets for tuple parsing
        content = content.replace("[", "(").replace("]", ")")
        # Safety check: only allow numbers, parens, commas, whitespace, minus, dots, e
        if not re.match(r'^[\d\s,.\-()eE+]+$', content):
            return None
        result = eval(content)
        if isinstance(result, tuple) and all(isinstance(r, tuple) for r in result):
            return tuple(tuple(float(v) for v in row) for row in result)
    except Exception:
        pass
    return None


def _parse_semicolon_matrix(content: str) -> Optional[Tuple[Tuple[float, ...], ...]]:
    """Parse semicolon-separated matrix format."""
    rows = content.split(";")
    try:
        result = []
        for row in rows:
            row = row.strip().strip("[]()").strip()
            if not row:
                continue
            if "," in row:
                values = [float(v.strip()) for v in row.split(",") if v.strip()]
            else:
                values = [float(v.strip()) for v in row.split() if v.strip()]
            if values:
                result.append(tuple(values))
        return tuple(result) if result else None
    except ValueError:
        return None


def _parse_newline_matrix(content: str) -> Optional[Tuple[Tuple[float, ...], ...]]:
    """Parse newline-separated matrix format."""
    rows = content.strip().split("\n")
    try:
        result = []
        for row in rows:
            row = row.strip().strip("[]()").strip()
            if not row:
                continue
            if "," in row:
                values = [float(v.strip()) for v in row.split(",") if v.strip()]
            else:
                values = [float(v.strip()) for v in row.split() if v.strip()]
            if values:
                result.append(tuple(values))
        return tuple(result) if result else None
    except ValueError:
        return None


def _parse_single_row_matrix(content: str) -> Optional[Tuple[Tuple[float, ...], ...]]:
    """Parse a single-row rank-2 tensor."""
    row = content.strip().strip("[]()").strip()
    if not row:
        return None
    try:
        if "," in row:
            values = [float(v.strip()) for v in row.split(",") if v.strip()]
        else:
            values = [float(v.strip()) for v in row.split() if v.strip()]
        if not values:
            return None
        return (tuple(values),)
    except ValueError:
        return None


def _is_rank1_bracketed(content: str) -> bool:
    """Detect explicit rank-1 syntax: single bracket pair with no nesting."""
    text = content.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return False
    if text.startswith("[["):
        return False
    if text.count("[") != 1 or text.count("]") != 1:
        return False
    if ";" in text or "\n" in text:
        return False
    return True


def format_vector(coords: Tuple[float, ...], precision: int = 3) -> str:
    """Format vector as string."""
    formatted = [f"{v:.{precision}g}" for v in coords]
    return f"[{', '.join(formatted)}]"


def format_matrix(values: Tuple[Tuple[float, ...], ...], precision: int = 3) -> str:
    """Format matrix as string."""
    rows = []
    for row in values:
        formatted = [f"{v:.{precision}g}" for v in row]
        rows.append(f"  [{', '.join(formatted)}]")
    return "[\n" + ",\n".join(rows) + "\n]"


def get_type_description(type_name: str) -> str:
    """Get human-readable description of type."""
    descriptions = {
        "vector": "Rank-1 Tensor",
        "matrix": "Rank-2 Tensor",
        "": "Unknown",
    }
    return descriptions.get(type_name, type_name)


def get_shape_string(type_name: str, data) -> str:
    """Get shape string for parsed data."""
    if type_name == "vector" and data:
        return f"({len(data)},)"
    if type_name == "matrix" and data:
        rows = len(data)
        cols = len(data[0]) if data else 0
        return f"({rows}, {cols})"
    return ""


class TextInputWidget:
    """Widget for text-based tensor input."""

    def __init__(self):
        self._buffer = ""
        self._label_buffer = ""

    def render(self, state: "AppState", dispatch, width: float, matrix_only: bool = False):
        """Render the text input widget."""
        imgui.text("Enter tensor data:")

        # Text input area
        imgui.push_item_width(width - 20)
        changed, new_text = imgui.input_text_multiline(
            "##text_input",
            state.input_text_content,
            2048,  # buffer size
            width - 20,
            120
        )
        imgui.pop_item_width()

        if changed:
            dispatch(SetTextInput(content=new_text))

        # Show parsed type and shape
        parsed_type = state.input_text_parsed_type
        if parsed_type:
            if matrix_only and parsed_type != "matrix":
                type_desc = "Invalid"
                shape_str = ""
            else:
                type_desc = get_type_description(parsed_type)
                parsed = parse_matrix(state.input_text_content) if matrix_only else parse_input(state.input_text_content)
                shape_str = get_shape_string(parsed_type, parsed if matrix_only else parsed[1])
            imgui.text_colored(
                f"Detected: {type_desc} {shape_str}",
                0.4, 0.8, 0.4, 1.0
            )
        elif state.input_text_content.strip():
            imgui.text_colored(
                "Could not parse input",
                0.8, 0.4, 0.4, 1.0
            )

        imgui.spacing()

        # Label input
        imgui.text("Label:")
        imgui.same_line()
        imgui.push_item_width(width - 80)
        changed, self._label_buffer = imgui.input_text(
            "##tensor_label",
            self._label_buffer,
            256
        )
        imgui.pop_item_width()

        imgui.spacing()

        # Action buttons
        can_create = parsed_type in ("vector", "matrix")
        if matrix_only and parsed_type not in ("vector", "matrix"):
            can_create = False

        if not can_create:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)

        button_label = "Create Tensor"
        if parsed_type in ("vector", "matrix"):
            type_desc = get_type_description(parsed_type)
            button_label = f"Create {type_desc}"
        if imgui.button(button_label, width - 20, 30):
            if can_create:
                label = self._label_buffer.strip() or self._generate_label(state, parsed_type)
                color, _ = get_next_color(state)
                dispatch(CreateTensorFromTextInput(
                    label=label,
                    color=color
                ))
                self._label_buffer = ""

        if not can_create:
            imgui.pop_style_var()

        imgui.same_line()
        if imgui.button("Clear"):
            dispatch(ClearTextInput())
            self._label_buffer = ""

        # Help text
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        imgui.text_colored("Input formats:", 0.6, 0.6, 0.6, 1.0)
        if not matrix_only:
            imgui.text_colored("  Rank-1: [1, 2, 3]", 0.5, 0.5, 0.5, 1.0)
            imgui.text_colored("  Rank-2 (1x3): 1, 2, 3", 0.5, 0.5, 0.5, 1.0)
        imgui.text_colored("  Rank-2: 1, 2; 3, 4", 0.5, 0.5, 0.5, 1.0)
        imgui.text_colored("  Or multi-line entries", 0.5, 0.5, 0.5, 1.0)

    def _generate_label(self, state: "AppState", tensor_type: str) -> str:
        """Generate an automatic label for the tensor."""
        count = get_tensor_count(state)
        return f"T{count + 1}"


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
        selection_id = state.selected_tensor_id
        selected_tensor = None
        if selection_id:
            for t in state.tensors:
                if t.id == selection_id:
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
        selection_id = state.selected_tensor_id

        # Filter tensors
        filtered = self._filter_tensors(tensors)

        if not filtered:
            imgui.text_colored("No tensors", 0.5, 0.5, 0.5, 1.0)
        else:
            for tensor in filtered:
                self._render_tensor_item(tensor, selection_id, dispatch, width - 30)

        imgui.end_child()

        # Action buttons
        imgui.spacing()
        if selection_id:
            button_width = (width - 30) / 2
            if imgui.button("Deselect", button_width, 0):
                dispatch(DeselectTensor())
            imgui.same_line()
            if imgui.button("Delete", button_width, 0):
                dispatch(DeleteTensor(id=selection_id))

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

    def _render_tensor_item(self, tensor, selection_id, dispatch, width):
        """Render a single tensor item in the list."""
        is_selected = tensor.id == selection_id
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


class InputPanel:
    """
    Left input panel for creating tensors.

    Contains:
    - Input selector for manual or file-based input
    - Active input widget
    - Tensor list showing all tensors in scene
    """

    INPUT_METHODS = [
        ("matrix", "Manual", "Enter tensor data manually"),
        ("json", "JSON", "Load a numeric tensor from JSON"),
        ("csv", "CSV", "Load a numeric tensor from CSV"),
        ("excel", "Excel", "Load a numeric tensor from Excel"),
        ("image", "Image", "Load an image tensor from file"),
    ]

    def __init__(self):
        self.text_widget = TextInputWidget()
        self.file_widget = FileInputWidget()
        self.tensor_list = TensorListWidget()
        self._last_mode = None

    def render(self, rect, state: "AppState", dispatch):
        """
        Render the input panel.

        Args:
            rect: (x, y, width, height) tuple
            state: Current app state
            dispatch: Action dispatch function
        """
        x, y, width, height = rect

        flags = (
            _WINDOW_NO_TITLE_BAR |
            _WINDOW_NO_RESIZE |
            _WINDOW_NO_MOVE |
            _WINDOW_NO_COLLAPSE
        )

        imgui.set_next_window_position(x, y)
        imgui.set_next_window_size(width, height)

        if imgui.begin("Input Panel", flags=flags):
            self._render_header(state, dispatch, width)
            imgui.separator()
            imgui.spacing()

            # Get active mode
            active_mode = state.active_mode if state else "vectors"
            self._last_mode = active_mode

            if imgui.begin_child(
                "##input_scroll",
                0,
                0,
                border=False,
                flags=_WINDOW_ALWAYS_VERTICAL_SCROLLBAR
            ):
                # Render different content based on mode
                if active_mode == "visualize":
                    # View mode - show view info
                    self._render_view_mode_content(state, dispatch, width, height)
                else:
                    # Algebra or Vision mode - show normal input
                    self._render_standard_content(state, dispatch, width, height)

            imgui.end_child()

        imgui.end()

    def _render_header(self, state: "AppState", dispatch, width: float):
        """Render panel header."""
        # Show mode-specific header
        active_mode = state.active_mode if state else "vectors"
        mode_labels = {
            "vectors": "TENSORS",
            "visualize": "VIEW",
        }
        header = mode_labels.get(active_mode, "TENSORS")
        imgui.text(header)
        imgui.same_line(width - 60)
        # Could add settings button here

    def _render_input_selector(self, state: "AppState", dispatch, width: float):
        """Render input method selector."""
        active_method = state.active_input_method
        method_ids = [m[0] for m in self.INPUT_METHODS]
        method_labels = [m[1] for m in self.INPUT_METHODS]
        method_tooltips = {m[0]: m[2] for m in self.INPUT_METHODS}

        try:
            current_index = method_ids.index(active_method)
        except ValueError:
            current_index = 0

        imgui.text("Input Source:")
        imgui.same_line()
        imgui.push_item_width(width - 110)
        changed, new_index = imgui.combo(
            "##input_type",
            current_index,
            method_labels
        )
        imgui.pop_item_width()

        if changed:
            dispatch(SetInputMethod(method=method_ids[new_index]))

        if imgui.is_item_hovered() and active_method in method_tooltips:
            imgui.set_tooltip(method_tooltips[active_method])

    def _render_active_input(self, state: "AppState", dispatch, width: float):
        """Render the currently active input widget."""
        active_method = state.active_input_method

        if active_method == "matrix":
            self.text_widget.render(state, dispatch, width, matrix_only=False)
        elif active_method in ("json", "csv", "excel", "image"):
            self.file_widget.render(state, dispatch, width, file_type=active_method)
        else:
            self.text_widget.render(state, dispatch, width, matrix_only=False)

    def _render_standard_content(self, state: "AppState", dispatch, width: float, height: float):
        """Render standard input content (tensor mode)."""
        # Input selector and widget
        self._render_input_selector(state, dispatch, width)
        imgui.spacing()

        expanded, _ = imgui.collapsing_header(
            "Create",
            imgui.TREE_NODE_DEFAULT_OPEN
        )
        if expanded:
            self._render_active_input(state, dispatch, width - 20)

        imgui.spacing()

        expanded, _ = imgui.collapsing_header(
            "Tensors",
            imgui.TREE_NODE_DEFAULT_OPEN
        )
        if expanded:
            list_height = max(220, height * 0.35)
            self.tensor_list.render(state, dispatch, width - 10, list_height)

        imgui.spacing()

    def _render_view_mode_content(self, state: "AppState", dispatch, width: float, height: float):
        """Render View mode content - simplified, just tensor list."""
        # Info text
        imgui.text_colored("View Settings", 0.6, 0.8, 0.6, 1.0)
        imgui.spacing()
        imgui.text_wrapped("Use the Operations panel on the right to adjust view settings.")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Tensor list for reference
        expanded, _ = imgui.collapsing_header(
            "Tensors",
            imgui.TREE_NODE_DEFAULT_OPEN
        )
        if expanded:
            list_height = max(300, height * 0.5)
            self.tensor_list.render(state, dispatch, width - 10, list_height)

        imgui.spacing()
