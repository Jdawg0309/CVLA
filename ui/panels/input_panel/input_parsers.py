"""
Input parsing utilities for the input panel.

Provides functions to parse user text input into rank-1 or rank-2 tensors.
"""

import re
from typing import Optional, Tuple, Union


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
