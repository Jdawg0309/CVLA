"""
Reducer for input panel actions.

Handles state changes for the text, file, and grid input widgets.
"""

from dataclasses import replace
from typing import Optional, Callable, TYPE_CHECKING
import re

if TYPE_CHECKING:
    from state.app_state import AppState

from state.actions import Action
from state.actions.input_panel_actions import (
    SetInputMethod, SetTextInput, ClearTextInput, ParseTextInput,
    SetFilePath, LoadFile, ClearFilePath,
    SetGridSize, SetGridCell, SetGridRow, SetGridColumn,
    ClearGrid, ApplyGridTemplate, TransposeGrid,
    CreateTensorFromTextInput, CreateTensorFromFileInput, CreateTensorFromGridInput,
)
from state.models.tensor_model import TensorData


def reduce_input_panel(
    state: "AppState",
    action: Action,
    with_history: Callable[["AppState"], "AppState"]
) -> Optional["AppState"]:
    """
    Reduce input panel actions.

    Returns:
        New state if action was handled, None otherwise.
    """
    # Input method switching
    if isinstance(action, SetInputMethod):
        return replace(state, active_input_method=action.method)

    # Text input actions
    if isinstance(action, SetTextInput):
        parsed_type = _parse_input_type(action.content)
        return replace(
            state,
            input_text_content=action.content,
            input_text_parsed_type=parsed_type
        )

    if isinstance(action, ClearTextInput):
        return replace(
            state,
            input_text_content="",
            input_text_parsed_type=""
        )

    if isinstance(action, ParseTextInput):
        parsed_type = _parse_input_type(state.input_text_content)
        return replace(state, input_text_parsed_type=parsed_type)

    # File input actions
    if isinstance(action, SetFilePath):
        return replace(state, input_file_path=action.path)

    if isinstance(action, ClearFilePath):
        return replace(state, input_file_path="")

    if isinstance(action, LoadFile):
        # File loading is handled elsewhere (triggers image loading)
        return None

    # Grid input actions
    if isinstance(action, SetGridSize):
        new_cells = _resize_grid(
            state.input_grid_cells,
            state.input_grid_rows,
            state.input_grid_cols,
            action.rows,
            action.cols
        )
        return replace(
            state,
            input_grid_rows=action.rows,
            input_grid_cols=action.cols,
            input_grid_cells=new_cells
        )

    if isinstance(action, SetGridCell):
        new_cells = _set_grid_cell(
            state.input_grid_cells,
            action.row,
            action.col,
            action.value
        )
        return replace(state, input_grid_cells=new_cells)

    if isinstance(action, SetGridRow):
        new_cells = _set_grid_row(
            state.input_grid_cells,
            action.row,
            action.values
        )
        return replace(state, input_grid_cells=new_cells)

    if isinstance(action, SetGridColumn):
        new_cells = _set_grid_column(
            state.input_grid_cells,
            action.col,
            action.values
        )
        return replace(state, input_grid_cells=new_cells)

    if isinstance(action, ClearGrid):
        new_cells = _create_zero_grid(state.input_grid_rows, state.input_grid_cols)
        return replace(state, input_grid_cells=new_cells)

    if isinstance(action, ApplyGridTemplate):
        new_cells = _apply_template(
            action.template,
            state.input_grid_rows,
            state.input_grid_cols
        )
        return replace(state, input_grid_cells=new_cells)

    if isinstance(action, TransposeGrid):
        new_cells = _transpose_grid(state.input_grid_cells)
        new_rows = state.input_grid_cols
        new_cols = state.input_grid_rows
        return replace(
            state,
            input_grid_rows=new_rows,
            input_grid_cols=new_cols,
            input_grid_cells=new_cells
        )

    # Tensor creation actions
    if isinstance(action, CreateTensorFromTextInput):
        return _create_tensor_from_text(state, action, with_history)

    if isinstance(action, CreateTensorFromFileInput):
        return _create_tensor_from_file(state, action, with_history)

    if isinstance(action, CreateTensorFromGridInput):
        return _create_tensor_from_grid(state, action, with_history)

    return None


def _parse_input_type(content: str) -> str:
    """
    Parse text input to determine type.

    Returns:
        "vector" if 1D data
        "matrix" if 2D data
        "" if unparseable
    """
    content = content.strip()
    if not content:
        return ""

    # Try to parse as vector: "1, 2, 3" or "[1, 2, 3]" or "1 2 3"
    # Try to parse as matrix: "1, 2; 3, 4" or "[[1, 2], [3, 4]]" or multi-line

    # Check for matrix indicators
    if ";" in content or "\n" in content or "[[" in content:
        if _try_parse_matrix(content) is not None:
            return "matrix"

    # Try as vector
    if _try_parse_vector(content) is not None:
        return "vector"

    # Try as matrix (might be single row)
    if _try_parse_matrix(content) is not None:
        return "matrix"

    return ""


def _try_parse_vector(content: str) -> Optional[tuple]:
    """Try to parse content as a vector."""
    content = content.strip()

    # Remove brackets
    content = content.strip("[]")

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


def _try_parse_matrix(content: str) -> Optional[tuple]:
    """Try to parse content as a matrix."""
    content = content.strip()

    # Handle Python-style nested lists
    if content.startswith("[["):
        try:
            # Simple eval-like parsing (safe subset)
            content = content.replace("[", "(").replace("]", ")")
            # Only allow numbers, parens, commas, whitespace, minus, dots
            if re.match(r'^[\d\s,.\-()eE+]+$', content):
                result = eval(content)
                if isinstance(result, tuple) and all(isinstance(r, tuple) for r in result):
                    return tuple(tuple(float(v) for v in row) for row in result)
        except Exception:
            pass

    # Handle semicolon-separated rows
    if ";" in content:
        rows = content.split(";")
        try:
            result = []
            for row in rows:
                row = row.strip().strip("[]")
                if "," in row:
                    values = [float(v.strip()) for v in row.split(",") if v.strip()]
                else:
                    values = [float(v.strip()) for v in row.split() if v.strip()]
                if values:
                    result.append(tuple(values))
            return tuple(result) if result else None
        except ValueError:
            return None

    # Handle newline-separated rows
    if "\n" in content:
        rows = content.strip().split("\n")
        try:
            result = []
            for row in rows:
                row = row.strip().strip("[]")
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

    return None


def _resize_grid(
    old_cells: tuple,
    old_rows: int,
    old_cols: int,
    new_rows: int,
    new_cols: int
) -> tuple:
    """Resize grid, preserving existing values where possible."""
    new_cells = []
    for r in range(new_rows):
        row = []
        for c in range(new_cols):
            if r < old_rows and c < old_cols and r < len(old_cells) and c < len(old_cells[r]):
                row.append(old_cells[r][c])
            else:
                row.append(0.0)
        new_cells.append(tuple(row))
    return tuple(new_cells)


def _set_grid_cell(cells: tuple, row: int, col: int, value: float) -> tuple:
    """Set a single cell in the grid."""
    cells_list = [list(r) for r in cells]
    if 0 <= row < len(cells_list) and 0 <= col < len(cells_list[row]):
        cells_list[row][col] = value
    return tuple(tuple(r) for r in cells_list)


def _set_grid_row(cells: tuple, row: int, values: tuple) -> tuple:
    """Set an entire row in the grid."""
    cells_list = [list(r) for r in cells]
    if 0 <= row < len(cells_list):
        for i, v in enumerate(values):
            if i < len(cells_list[row]):
                cells_list[row][i] = v
    return tuple(tuple(r) for r in cells_list)


def _set_grid_column(cells: tuple, col: int, values: tuple) -> tuple:
    """Set an entire column in the grid."""
    cells_list = [list(r) for r in cells]
    for i, v in enumerate(values):
        if i < len(cells_list) and col < len(cells_list[i]):
            cells_list[i][col] = v
    return tuple(tuple(r) for r in cells_list)


def _create_zero_grid(rows: int, cols: int) -> tuple:
    """Create a grid of zeros."""
    return tuple(tuple(0.0 for _ in range(cols)) for _ in range(rows))


def _apply_template(template: str, rows: int, cols: int) -> tuple:
    """Apply a template to create grid values."""
    import random

    if template == "identity":
        return tuple(
            tuple(1.0 if i == j else 0.0 for j in range(cols))
            for i in range(rows)
        )

    if template == "zeros":
        return _create_zero_grid(rows, cols)

    if template == "ones":
        return tuple(tuple(1.0 for _ in range(cols)) for _ in range(rows))

    if template == "random":
        return tuple(
            tuple(round(random.uniform(-1, 1), 2) for _ in range(cols))
            for _ in range(rows)
        )

    if template == "diagonal":
        return tuple(
            tuple(float(i + 1) if i == j else 0.0 for j in range(cols))
            for i in range(rows)
        )

    return _create_zero_grid(rows, cols)


def _transpose_grid(cells: tuple) -> tuple:
    """Transpose the grid."""
    if not cells or not cells[0]:
        return cells
    rows = len(cells)
    cols = len(cells[0])
    return tuple(
        tuple(cells[r][c] for r in range(rows))
        for c in range(cols)
    )


def _create_tensor_from_text(
    state: "AppState",
    action: "CreateTensorFromTextInput",
    with_history: Callable
) -> Optional["AppState"]:
    """Create a tensor from text input."""
    content = state.input_text_content
    parsed_type = state.input_text_parsed_type

    if parsed_type == "vector":
        data = _try_parse_vector(content)
        if data is None:
            return state
        tensor = TensorData.create_vector(
            coords=data,
            label=action.label,
            color=action.color
        )
    elif parsed_type == "matrix":
        data = _try_parse_matrix(content)
        if data is None:
            return state
        tensor = TensorData.create_matrix(
            values=data,
            label=action.label,
            color=action.color
        )
    else:
        return state

    new_state = replace(
        state,
        tensors=state.tensors + (tensor,),
        selected_tensor_id=tensor.id,
        input_text_content="",
        input_text_parsed_type=""
    )
    return with_history(new_state)


def _create_tensor_from_file(
    state: "AppState",
    action: "CreateTensorFromFileInput",
    with_history: Callable
) -> Optional["AppState"]:
    """Create a tensor from file input."""
    path = state.input_file_path
    if not path:
        return state

    try:
        from domain.images.io.image_loader import load_image
        pixels = load_image(path)
        label = action.label if action.label else path.split("/")[-1].split("\\")[-1]
        if hasattr(pixels, "data"):
            pixels = pixels.data
        tensor = TensorData.create_image(pixels=pixels, name=label)

        new_state = replace(
            state,
            tensors=state.tensors + (tensor,),
            selected_tensor_id=tensor.id,
            input_file_path=""
        )
        return with_history(new_state)
    except Exception:
        return state


def _create_tensor_from_grid(
    state: "AppState",
    action: "CreateTensorFromGridInput",
    with_history: Callable
) -> Optional["AppState"]:
    """Create a tensor from grid input."""
    cells = state.input_grid_cells

    if action.tensor_type == "vector":
        # Use first row as vector
        if cells and cells[0]:
            data = cells[0]
            tensor = TensorData.create_vector(
                coords=data,
                label=action.label,
                color=action.color
            )
        else:
            return state
    else:
        # Use full grid as matrix
        tensor = TensorData.create_matrix(
            values=cells,
            label=action.label,
            color=action.color
        )

    new_state = replace(
        state,
        tensors=state.tensors + (tensor,),
        selected_tensor_id=tensor.id
    )
    return with_history(new_state)
