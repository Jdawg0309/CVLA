"""
Matrix action reducers.
"""

from dataclasses import replace

from state.actions import (
    AddMatrix, DeleteMatrix, UpdateMatrixCell, UpdateMatrix, SelectMatrix,
    ApplyMatrixToSelected, ApplyMatrixToAll,
)
from state.models import MatrixData


def reduce_matrices(state, action, with_history):
    if isinstance(action, AddMatrix):
        label = action.label or f"M{state.next_matrix_id}"
        new_matrix = MatrixData(
            id=str(__import__('uuid').uuid4()),
            values=action.values,
            label=label,
            visible=True
        )
        new_state = replace(state,
            matrices=state.matrices + (new_matrix,),
            selected_id=new_matrix.id,
            selected_type='matrix',
            next_matrix_id=state.next_matrix_id + 1,
            show_matrix_editor=False,
        )
        return with_history(new_state)

    if isinstance(action, DeleteMatrix):
        new_matrices = tuple(m for m in state.matrices if m.id != action.id)
        new_selected_id = None if state.selected_id == action.id else state.selected_id
        new_selected_type = None if state.selected_id == action.id else state.selected_type
        new_state = replace(state,
            matrices=new_matrices,
            selected_id=new_selected_id,
            selected_type=new_selected_type,
        )
        return with_history(new_state)

    if isinstance(action, UpdateMatrixCell):
        new_matrices = tuple(
            m.with_cell(action.row, action.col, action.value) if m.id == action.id else m
            for m in state.matrices
        )
        new_state = replace(state, matrices=new_matrices)
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
        new_state = replace(state, matrices=new_matrices)
        return with_history(new_state)

    if isinstance(action, SelectMatrix):
        return replace(state, selected_id=action.id, selected_type='matrix')

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
        for v in state.vectors:
            if v.id == state.selected_id:
                vec_np = v.to_numpy()
                result = mat_np @ vec_np
                new_coords = (float(result[0]), float(result[1]), float(result[2]))
                new_vectors.append(replace(v, coords=new_coords))
            else:
                new_vectors.append(v)

        new_state = replace(state, vectors=tuple(new_vectors))
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
        for v in state.vectors:
            if v.visible:
                vec_np = v.to_numpy()
                result = mat_np @ vec_np
                new_coords = (float(result[0]), float(result[1]), float(result[2]))
                new_vectors.append(replace(v, coords=new_coords))
            else:
                new_vectors.append(v)

        new_state = replace(state, vectors=tuple(new_vectors))
        return with_history(new_state)

    return None
