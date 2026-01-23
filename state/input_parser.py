"""
Shared parser used by both the UI and reducers.

This module focuses on reading raw text, extracting numeric rows, and
inferring tensor rank from the actual data instead of the notation.
"""

import re
from typing import Optional, Tuple, Union

VectorData = Tuple[float, ...]
MatrixData = Tuple[Tuple[float, ...], ...]
TensorParseResult = Tuple[str, Optional[Union[VectorData, MatrixData]]]


def parse_input(content: str, matrix_only: bool = False) -> TensorParseResult:
    """
    Parse text input and return the detected type plus raw numeric values.

    Rank inference is based on the underlying structure, not the delimiters.
    """
    content = content.strip()
    if not content:
        return "", None

    rows = parse_matrix(content)
    if not rows:
        return "", None

    if not matrix_only and _looks_like_flat_vector(rows):
        return "vector", rows[0]

    return "matrix", rows


def parse_matrix(content: str) -> Optional[MatrixData]:
    """Parse content into nested rows irrespective of rank."""
    content = content.strip()
    if not content:
        return None

    if content.startswith("[["):
        return _parse_python_matrix(content)
    if ";" in content:
        return _parse_semicolon_matrix(content)
    if "\n" in content:
        return _parse_newline_matrix(content)
    return _parse_single_row_matrix(content)


def _normalize_value_row(row: str) -> Optional[Tuple[float, ...]]:
    row = row.strip().strip("[]()").strip()
    if not row:
        return None
    separator = "," if "," in row else None
    if separator:
        parts = [p.strip() for p in row.split(",")]
    else:
        parts = row.split()

    try:
        values = tuple(float(p) for p in parts if p)
        return values if values else None
    except ValueError:
        return None


def _parse_python_matrix(content: str) -> Optional[MatrixData]:
    try:
        sanitized = content.replace("[", "(").replace("]", ")")
        if not re.match(r'^[\d\s,.\-()eE+]+$', sanitized):
            return None
        result = eval(sanitized)
        if isinstance(result, tuple) and result and all(isinstance(row, tuple) for row in result):
            parsed_rows = tuple(tuple(float(v) for v in row) for row in result)
            return parsed_rows if parsed_rows else None
    except Exception:
        pass
    return None


def _parse_semicolon_matrix(content: str) -> Optional[MatrixData]:
    rows = content.split(";")
    parsed = []
    for row in rows:
        values = _normalize_value_row(row)
        if values:
            parsed.append(values)
    return tuple(parsed) if parsed else None


def _parse_newline_matrix(content: str) -> Optional[MatrixData]:
    rows = content.strip().split("\n")
    parsed = []
    for row in rows:
        values = _normalize_value_row(row)
        if values:
            parsed.append(values)
    return tuple(parsed) if parsed else None


def _parse_single_row_matrix(content: str) -> Optional[MatrixData]:
    values = _normalize_value_row(content)
    if values:
        return (values,)
    return None


def _looks_like_flat_vector(rows: MatrixData) -> bool:
    return len(rows) == 1 and len(rows[0]) > 0
