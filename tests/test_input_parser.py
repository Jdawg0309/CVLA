import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from state.input_parser import TensorKind, parse_tensor
from state.models.tensor_model import TensorData
from state.reducers.reducer_tensors import _op_transpose


@pytest.mark.parametrize(
    "content",
    [
        "1 2 3",
        "[1 2 3]",
    ],
)
def test_flat_input_parses_as_vector(content):
    """Flat vector input should always yield order-1 tensors."""
    parsed = parse_tensor(content)
    assert parsed.kind == TensorKind.VECTOR
    assert parsed.order == 1
    assert parsed.shape == (3,)


def test_matrix_input_parses_as_order_two():
    """Row-separated input should produce a rank-2 tensor."""
    parsed = parse_tensor("1 2; 3 4")
    assert parsed.kind == TensorKind.MATRIX
    assert parsed.order == 2
    assert parsed.shape == (2, 2)


def test_scalar_input_parses_as_order_zero():
    """Single numeric value is treated as a scalar (rank 0)."""
    parsed = parse_tensor("5")
    assert parsed.kind == TensorKind.SCALAR
    assert parsed.order == 0
    assert parsed.shape == ()


def test_ragged_input_raises():
    """Ragged rows (different column counts) should fail cleanly."""
    with pytest.raises(ValueError):
        parse_tensor("1 2; 3")


def test_transpose_preserves_order():
    """Transposing a rank-2 tensor must keep its order (rank)."""
    parsed = parse_tensor("1 2; 3 4")
    tensor = TensorData.from_parsed(parsed=parsed, label="T", color=(0.5, 0.5, 0.5))
    results = _op_transpose([tensor], {}, True)
    assert results
    assert results[0].rank == 2
