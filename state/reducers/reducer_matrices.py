"""
Matrix action reducers.

Writes to unified tensor store only.
"""

from dataclasses import replace
from uuid import uuid4
import numpy as np

from state.actions import (
    AddMatrix, DeleteMatrix, UpdateMatrixCell, UpdateMatrix, SelectMatrix,
    ApplyMatrixToSelected, ApplyMatrixToAll, ToggleMatrixPlot,
)
from state.models import TensorData


def reduce_matrices(state, action, with_history):
    if isinstance(action, AddMatrix):
        label = action.label or f"M{state.next_matrix_id}"
        matrix_id = f"mat_{uuid4().hex[:8]}"

        # Create TensorData (unified storage)
        new_tensor = TensorData.create_matrix(action.values, label)
        new_tensor = replace(new_tensor, id=matrix_id)

        new_state = replace(state,
            tensors=state.tensors + (new_tensor,),
            selected_tensor_id=matrix_id,
            next_matrix_id=state.next_matrix_id + 1,
            show_matrix_editor=False,
        )
        return with_history(new_state)

    if isinstance(action, DeleteMatrix):
        new_tensors = tuple(t for t in state.tensors if t.id != action.id)
        new_selected_tensor_id = None if state.selected_tensor_id == action.id else state.selected_tensor_id
        new_state = replace(state,
            tensors=new_tensors,
            selected_tensor_id=new_selected_tensor_id,
        )
        return with_history(new_state)

    if isinstance(action, UpdateMatrixCell):
        def update_tensor_cell(t):
            if t.id != action.id or t.rank != 2 or t.is_image_dtype:
                return t
            # Convert data to mutable list, update cell, convert back
            rows = [list(row) for row in t.data]
            if 0 <= action.row < len(rows) and 0 <= action.col < len(rows[0]):
                rows[action.row][action.col] = float(action.value)
            new_data = tuple(tuple(row) for row in rows)
            return replace(t, data=new_data)

        new_tensors = tuple(update_tensor_cell(t) for t in state.tensors)
        new_state = replace(state, tensors=new_tensors)
        return with_history(new_state)

    if isinstance(action, UpdateMatrix):
        def update_tensor(t):
            if t.id != action.id or t.rank != 2 or t.is_image_dtype:
                return t
            new_data = action.values if action.values is not None else t.data
            new_shape = (len(new_data), len(new_data[0])) if action.values is not None else t.shape
            new_label = action.label if action.label is not None else t.label
            return replace(t, data=new_data, shape=new_shape, label=new_label)

        new_tensors = tuple(update_tensor(t) for t in state.tensors)
        new_state = replace(state, tensors=new_tensors)
        return with_history(new_state)

    if isinstance(action, SelectMatrix):
        return replace(state, selected_tensor_id=action.id)

    if isinstance(action, ApplyMatrixToSelected):
        # Find matrix tensor
        matrix_tensor = None
        for t in state.tensors:
            if t.id == action.matrix_id and t.rank == 2 and not t.is_image_dtype:
                matrix_tensor = t
                break
        if matrix_tensor is None:
            return state

        # Check if selected tensor is a vector
        selected_tensor = None
        if state.selected_tensor_id:
            for t in state.tensors:
                if t.id == state.selected_tensor_id and t.rank == 1:
                    selected_tensor = t
                    break
        if selected_tensor is None:
            return state

        mat_np = np.array(matrix_tensor.data, dtype=np.float32)
        vec_np = np.array(selected_tensor.data, dtype=np.float32)
        if mat_np.shape[1] != vec_np.shape[0]:
            return state

        result = mat_np @ vec_np
        new_coords = tuple(float(val) for val in result.tolist())
        transformed_id = selected_tensor.id

        new_tensors = tuple(
            replace(t, data=new_coords, shape=(len(new_coords),))
            if t.id == transformed_id and t.rank == 1 else t
            for t in state.tensors
        )
        new_state = replace(state, tensors=new_tensors)
        return with_history(new_state)

    if isinstance(action, ApplyMatrixToAll):
        # Find matrix tensor
        matrix_tensor = None
        for t in state.tensors:
            if t.id == action.matrix_id and t.rank == 2 and not t.is_image_dtype:
                matrix_tensor = t
                break
        if matrix_tensor is None:
            return state

        mat_np = np.array(matrix_tensor.data, dtype=np.float32)
        transformed_coords = {}  # id -> new_coords

        # Find all visible vector tensors and transform them
        for t in state.tensors:
            if t.rank == 1 and t.visible:
                vec_np = np.array(t.data, dtype=np.float32)
                if mat_np.shape[1] == vec_np.shape[0]:
                    result = mat_np @ vec_np
                    new_coords = tuple(float(val) for val in result.tolist())
                    transformed_coords[t.id] = new_coords

        def update_tensor(t):
            if t.id in transformed_coords and t.rank == 1:
                coords = transformed_coords[t.id]
                return replace(t, data=coords, shape=(len(coords),))
            return t

        new_tensors = tuple(update_tensor(t) for t in state.tensors)
        new_state = replace(state, tensors=new_tensors)
        return with_history(new_state)

    if isinstance(action, ToggleMatrixPlot):
        return replace(state, matrix_plot_enabled=not state.matrix_plot_enabled)

    return None
