"""
Structured tensor parser shared by UI and reducers.

This module exposes a single source of truth for interpreting textual tensor
input. It converts the input into a canonical nested tuple structure with
explicit metadata (order, shape, kind) so the rest of the system can trust the
result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple, Union

Number = Union[int, float]
NestedData = Union[Number, Sequence["NestedData"]]


class TensorKind(Enum):
    """Semantic kind derived from tensor order."""

    SCALAR = "scalar"
    VECTOR = "vector"
    MATRIX = "matrix"
    TENSOR = "tensor"


@dataclass(frozen=True)
class ParsedTensor:
    """Canonical representation produced by the parser."""

    data: Union[float, Tuple["ParsedTensorData", ...]]
    order: int
    shape: Tuple[int, ...]
    kind: TensorKind


ParsedTensorData = Union[float, Tuple["ParsedTensorData", ...]]


_NUMBER_PATTERN = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)


def parse_tensor(content: str) -> ParsedTensor:
    """
    Parse the given text into a ParsedTensor.

    Raises:
        ValueError: when the input cannot be parsed as a numeric tensor.
    """
    content = content.strip()
    if not content:
        raise ValueError("Input is empty.")

    try:
        if content[0] in "[(":
            structure = _BracketParser(content).parse()
        else:
            structure = _parse_plain_text(content)
    except ValueError as exc:
        raise ValueError(f"Invalid tensor input: {exc}") from exc

    return build_parsed_tensor(structure)


def try_parse_tensor(content: str) -> Optional[ParsedTensor]:
    """Return parsed tensor or None when parsing fails."""
    try:
        return parse_tensor(content)
    except ValueError:
        return None


def build_parsed_tensor(data: NestedData) -> ParsedTensor:
    """
    Convert a nested structure of numbers/lists into a ParsedTensor.
    """
    normalized, shape = _normalize_structure(data)
    order = len(shape)
    kind = _kind_from_order(order)
    return ParsedTensor(data=normalized, order=order, shape=shape, kind=kind)


def _kind_from_order(order: int) -> TensorKind:
    if order == 0:
        return TensorKind.SCALAR
    if order == 1:
        return TensorKind.VECTOR
    if order == 2:
        return TensorKind.MATRIX
    return TensorKind.TENSOR


def _normalize_structure(value: NestedData) -> Tuple[ParsedTensorData, Tuple[int, ...]]:
    if isinstance(value, (int, float)):
        return float(value), ()
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Tensor dimension cannot be empty.")
        normalized_items = []
        child_shapes = []
        for item in value:
            normalized_child, shape = _normalize_structure(item)
            normalized_items.append(normalized_child)
            child_shapes.append(shape)
        first_shape = child_shapes[0]
        for shape in child_shapes:
            if shape != first_shape:
                raise ValueError("Tensor dimensions are not uniform.")
        return tuple(normalized_items), (len(normalized_items),) + first_shape
    raise ValueError("Unsupported tensor element.")


def _parse_plain_text(content: str) -> NestedData:
    rows = [row.strip() for row in re.split(r"[;\n]+", content) if row.strip()]
    if not rows:
        raise ValueError("No numeric values found.")

    if len(rows) == 1:
        values = _parse_number_list(rows[0])
        if len(values) == 1:
            return float(values[0])
        return tuple(values)

    matrix: list[tuple[float, ...]] = []
    expected_cols: Optional[int] = None
    for row in rows:
        values = _parse_number_list(row)
        if expected_cols is None:
            expected_cols = len(values)
        elif len(values) != expected_cols:
            raise ValueError("Rows must have the same number of columns.")
        matrix.append(tuple(values))

    return tuple(matrix)


def _parse_number_list(row: str) -> Sequence[float]:
    matches = _NUMBER_PATTERN.findall(row)
    if not matches:
        raise ValueError("Row contains no numeric values.")
    return tuple(float(num) for num in matches)


class _BracketParser:
    """Low-level parser for bracketed tensor literals."""

    _OPEN_TO_CLOSE = {"[": "]", "(": ")"}

    def __init__(self, text: str):
        self._text = text
        self._pos = 0
        self._length = len(text)

    def parse(self) -> NestedData:
        value = self._parse_value()
        self._skip_whitespace()
        if not self._at_end():
            raise ValueError("Unexpected trailing characters.")
        return value

    def _parse_value(self) -> NestedData:
        self._skip_whitespace()
        if self._at_end():
            raise ValueError("Unexpected end of input.")
        char = self._current()
        if char in self._OPEN_TO_CLOSE:
            return self._parse_list(char)
        return self._parse_number()

    def _parse_list(self, opener: str) -> NestedData:
        closer = self._OPEN_TO_CLOSE[opener]
        self._advance()
        items = []
        while True:
            self._skip_whitespace()
            if self._at_end():
                raise ValueError("Unterminated bracket.")
            if self._current() == closer:
                self._advance()
                break
            items.append(self._parse_value())
            self._skip_whitespace()
            if self._at_end():
                raise ValueError("Unterminated bracket.")
            if self._current() in ",;":
                self._advance()
        if not items:
            raise ValueError("Tensor bracket must contain at least one element.")
        return tuple(items)

    def _parse_number(self) -> float:
        match = _NUMBER_PATTERN.match(self._text[self._pos :])
        if not match:
            raise ValueError(f"Invalid token at position {self._pos}.")
        token = match.group()
        self._pos += len(token)
        return float(token)

    def _skip_whitespace(self):
        while not self._at_end() and self._current().isspace():
            self._advance()

    def _advance(self):
        self._pos += 1

    def _current(self) -> str:
        return self._text[self._pos]

    def _at_end(self) -> bool:
        return self._pos >= self._length
