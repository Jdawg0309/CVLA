"""
Vector action reducers.
"""

from dataclasses import replace

from runtime.state_queries import get_next_color
from state.actions import AddVector, DeleteVector, UpdateVector, SelectVector
from state.models import VectorData


def reduce_vectors(state, action, with_history):
    if isinstance(action, AddVector):
        color, new_color_idx = get_next_color(state)
        actual_color = action.color if action.color != (0.8, 0.2, 0.2) else color
        label = action.label or f"v{state.next_vector_id}"

        new_vector = VectorData.create(action.coords, actual_color, label)
        new_state = replace(state,
            vectors=state.vectors + (new_vector,),
            selected_id=new_vector.id,
            selected_type='vector',
            next_vector_id=state.next_vector_id + 1,
            next_color_index=new_color_idx,
            input_vector_label="",
            input_vector_coords=(1.0, 0.0, 0.0),
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

    return None
