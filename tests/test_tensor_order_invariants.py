import pytest
from dataclasses import replace

from state.input_parser import parse_tensor
from state.models.tensor_model import TensorData
from state.reducers.reducer_tensors import (
    _op_transpose,
    _op_outer_product,
    _op_determinant,
    OperationError,
)


def _make_tensor(content: str, label: str = "tensor") -> TensorData:
    parsed = parse_tensor(content)
    return TensorData.from_parsed(parsed, label=label)


def _as_column(tensor: TensorData) -> TensorData:
    """Simulate a column representation while keeping the same order."""
    column_data = tuple((float(value),) for value in tensor.data)
    return replace(
        tensor,
        data=column_data,
        shape=(len(column_data), 1),
        order=tensor.order
    )


def _as_row(tensor: TensorData) -> TensorData:
    """Simulate a row representation while keeping the same order."""
    row_data = (tuple(float(value) for value in tensor.data),)
    return replace(
        tensor,
        data=row_data,
        shape=(1, len(row_data[0])),
        order=tensor.order
    )


def test_flat_vector_order():
    tensor = _make_tensor("1 2 3", label="v")
    assert tensor.order == 1


def test_representation_helpers_preserve_order():
    v = _make_tensor("1 2 3", label="v")
    assert _as_column(v).order == 1
    assert _as_row(v).order == 1


def test_transpose_vector_preserves_order():
    v = _make_tensor("1 2 3", label="v")
    result = _op_transpose([v], {}, True)
    assert result
    assert result[0].order == 1


def test_outer_product_increases_order():
    v = _make_tensor("1 2 3", label="v")
    results = _op_outer_product([v, v], {}, True)
    assert results
    assert results[0].order == 2


def test_det_raises_for_vector():
    v = _make_tensor("1 2 3", label="v")
    with pytest.raises(OperationError):
        _op_determinant([v], {}, True)


def test_rank_unbounded_with_nesting():
    depth = 6
    nested = "[" * depth + "1" + "]" * depth
    tensor = _make_tensor(nested, label="deep")
    assert tensor.order == depth
