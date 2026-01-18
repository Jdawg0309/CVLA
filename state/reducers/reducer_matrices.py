"""
Matrix action reducers.

NOTE: This reducer implements dual-write to both legacy (matrices) and new (tensors) stores
during the migration period. Once migration is complete, this file can be removed.
"""

from dataclasses import replace
from uuid import uuid4

from state.actions import (
    AddMatrix, DeleteMatrix, UpdateMatrixCell, UpdateMatrix, SelectMatrix,
    ApplyMatrixToSelected, ApplyMatrixToAll, ToggleMatrixPlot,
)
from state.models import MatrixData, TensorData


def reduce_matrices(state, action, with_history):
    if isinstance(action, AddMatrix):
        label = action.label or f"M{state.next_matrix_id}"
        matrix_id = str(uuid4())
        new_matrix = MatrixData(
            id=matrix_id,
            values=action.values,
            label=label,
            visible=True
        )

        # DUAL-WRITE: Also create TensorData with same ID
        new_tensor = TensorData.create_matrix(action.values, label)
        new_tensor = replace(new_tensor, id=matrix_id)

        new_state = replace(state,
            matrices=state.matrices + (new_matrix,),
            tensors=state.tensors + (new_tensor,),  # DUAL-WRITE
            selected_id=new_matrix.id,
            selected_type='matrix',
            selected_tensor_id=matrix_id,  # DUAL-WRITE
            next_matrix_id=state.next_matrix_id + 1,
            show_matrix_editor=False,
        )
        return with_history(new_state)

    if isinstance(action, DeleteMatrix):
        new_matrices = tuple(m for m in state.matrices if m.id != action.id)
        # DUAL-WRITE: Also remove from tensors
        new_tensors = tuple(t for t in state.tensors if t.id != action.id)
        new_selected_id = None if state.selected_id == action.id else state.selected_id
        new_selected_type = None if state.selected_id == action.id else state.selected_type
        new_selected_tensor_id = None if state.selected_tensor_id == action.id else state.selected_tensor_id
        new_state = replace(state,
            matrices=new_matrices,
            tensors=new_tensors,  # DUAL-WRITE
            selected_id=new_selected_id,
            selected_type=new_selected_type,
            selected_tensor_id=new_selected_tensor_id,  # DUAL-WRITE
        )
        return with_history(new_state)

    if isinstance(action, UpdateMatrixCell):
        new_matrices = tuple(
            m.with_cell(action.row, action.col, action.value) if m.id == action.id else m
            for m in state.matrices
        )
        # DUAL-WRITE: Also update in tensors
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
        new_state = replace(state, matrices=new_matrices, tensors=new_tensors)
        return with_history(new_state)

    if isinstance(action, UpdateMatrix):
        new_matrices = tuple(
            replace(
                m,
                values=action.values if action.values is not None else m.values,
                label=action.label if action.label is not None else m.label,
            ) if m.id == action.id else m
            for m in state.matrices
        )
        # DUAL-WRITE: Also update in tensors
        def update_tensor(t):
            if t.id != action.id or t.rank != 2 or t.is_image_dtype:
                return t
            new_data = action.values if action.values is not None else t.data
            new_shape = (len(new_data), len(new_data[0])) if action.values is not None else t.shape
            new_label = action.label if action.label is not None else t.label
            return replace(t, data=new_data, shape=new_shape, label=new_label)

        new_tensors = tuple(update_tensor(t) for t in state.tensors)
        new_state = replace(state, matrices=new_matrices, tensors=new_tensors)
        return with_history(new_state)

    if isinstance(action, SelectMatrix):
        # DUAL-WRITE: Set both legacy and tensor selection
        return replace(state, selected_id=action.id, selected_type='matrix', selected_tensor_id=action.id)

    if isinstance(action, ApplyMatrixToSelected):
        matrix = None
        for m in state.matrices:
            if m.id == action.matrix_id:
                matrix = m
                break
        if matrix is None or state.selected_type != 'vector':
            return state

        mat_np = matrix.to_numpy()
        new_vectors = []
        transformed_id = None
        new_coords = None
        for v in state.vectors:
            if v.id == state.selected_id:
                vec_np = v.to_numpy()
                if mat_np.shape[1] != vec_np.shape[0]:
                    return state
                result = mat_np @ vec_np
                new_coords = tuple(float(val) for val in result.tolist())
                transformed_id = v.id
                new_vectors.append(replace(v, coords=new_coords))
            else:
                new_vectors.append(v)

        # DUAL-WRITE: Also update corresponding tensor
        new_tensors = tuple(
            replace(t, data=new_coords, shape=(len(new_coords),))
            if t.id == transformed_id and t.rank == 1 else t
            for t in state.tensors
        )
        new_state = replace(state, vectors=tuple(new_vectors), tensors=new_tensors)
        return with_history(new_state)

    if isinstance(action, ApplyMatrixToAll):
        matrix = None
        for m in state.matrices:
            if m.id == action.matrix_id:
                matrix = m
                break
        if matrix is None:
            return state

        mat_np = matrix.to_numpy()
        new_vectors = []
        transformed_coords = {}  # id -> new_coords
        for v in state.vectors:
            if v.visible:
                vec_np = v.to_numpy()
                if mat_np.shape[1] != vec_np.shape[0]:
                    new_vectors.append(v)
                else:
                    result = mat_np @ vec_np
                    new_coords = tuple(float(val) for val in result.tolist())
                    transformed_coords[v.id] = new_coords
                    new_vectors.append(replace(v, coords=new_coords))
            else:
                new_vectors.append(v)

        # DUAL-WRITE: Also update corresponding tensors
        def update_tensor(t):
            if t.id in transformed_coords and t.rank == 1:
                coords = transformed_coords[t.id]
                return replace(t, data=coords, shape=(len(coords),))
            return t

        new_tensors = tuple(update_tensor(t) for t in state.tensors)
        new_state = replace(state, vectors=tuple(new_vectors), tensors=new_tensors)
        return with_history(new_state)

    if isinstance(action, ToggleMatrixPlot):
        return replace(state, matrix_plot_enabled=not state.matrix_plot_enabled)

    return None
