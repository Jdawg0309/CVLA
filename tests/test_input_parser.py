import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from state.input_parser import parse_input


@pytest.mark.parametrize(
    "content",
    [
        "[1 2 3]",
        "1 2 3",
        "1,2,3",
    ],
)
def test_flat_list_parses_as_vector(content):
    """Flat tokens should always infer rank-1 when matrix_only is False."""
    parsed_type, parsed_data = parse_input(content, matrix_only=False)
    assert parsed_type == "vector"
    assert parsed_data == (1.0, 2.0, 3.0)


def test_semicolon_input_is_matrix():
    """Semicolons force a matrix rank."""
    parsed_type, parsed_data = parse_input("[1;2;3]")
    assert parsed_type == "matrix"
    assert parsed_data == ((1.0,), (2.0,), (3.0,))


def test_matrix_only_mode_prefers_matrix():
    """Matrix-only mode should not demote single rows to vectors."""
    parsed_type, parsed_data = parse_input("1 2 3", matrix_only=True)
    assert parsed_type == "matrix"
    assert parsed_data == ((1.0, 2.0, 3.0),)
