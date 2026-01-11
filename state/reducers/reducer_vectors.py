"""
Vector action reducers.
"""

from dataclasses import replace

from state.selectors import get_next_color, COLOR_PALETTE
from state.actions import (
    AddVector, DeleteVector, UpdateVector, SelectVector,
    ClearAllVectors, DuplicateVector, DeselectVector,
)
from state.models import VectorData


def reduce_vectors(state, action, with_history):
    if isinstance(action, AddVector):
        color, new_color_idx = get_next_color(state)
        actual_color = action.color if action.color != (0.8, 0.2, 0.2) else color
        label = action.label or f"v{state.next_vector_id}"
        next_input_color = COLOR_PALETTE[new_color_idx % len(COLOR_PALETTE)]

        new_vector = VectorData.create(action.coords, actual_color, label)
        new_state = replace(state,
            vectors=state.vectors + (new_vector,),
            selected_id=new_vector.id,
            selected_type='vector',
            next_vector_id=state.next_vector_id + 1,
            next_color_index=new_color_idx,
            input_vector_label="",
            input_vector_coords=(1.0, 0.0, 0.0),
            input_vector_color=next_input_color,
        )
        return with_history(new_state)

    if isinstance(action, DeleteVector):
        new_vectors = tuple(v for v in state.vectors if v.id != action.id)
        new_selected_id = None if state.selected_id == action.id else state.selected_id
        new_selected_type = None if state.selected_id == action.id else state.selected_type
        new_state = replace(state,
            vectors=new_vectors,
            selected_id=new_selected_id,
            selected_type=new_selected_type,
        )
        return with_history(new_state)

    if isinstance(action, UpdateVector):
        new_vectors = tuple(
            replace(v,
                coords=action.coords if action.coords is not None else v.coords,
                color=action.color if action.color is not None else v.color,
                label=action.label if action.label is not None else v.label,
                visible=action.visible if action.visible is not None else v.visible,
            ) if v.id == action.id else v
            for v in state.vectors
        )
        new_state = replace(state, vectors=new_vectors)
        return with_history(new_state)

    if isinstance(action, SelectVector):
        return replace(state, selected_id=action.id, selected_type='vector')

    if isinstance(action, DeselectVector):
        if state.selected_type == 'vector':
            return replace(state, selected_id=None, selected_type=None)
        return state

    if isinstance(action, ClearAllVectors):
        new_selected_id = None if state.selected_type == 'vector' else state.selected_id
        new_selected_type = None if state.selected_type == 'vector' else state.selected_type
        new_state = replace(state,
            vectors=(),
            selected_id=new_selected_id,
            selected_type=new_selected_type,
        )
        return with_history(new_state)

    if isinstance(action, DuplicateVector):
        source = None
        for v in state.vectors:
            if v.id == action.id:
                source = v
                break
        if source is None:
            return state

        new_label = f"{source.label}_copy"
        new_vector = VectorData.create(source.coords, source.color, new_label)
        new_state = replace(state,
            vectors=state.vectors + (new_vector,),
            selected_id=new_vector.id,
            selected_type='vector',
        )
        return with_history(new_state)

    return None
