"""
Vector action reducers.

NOTE: This reducer implements dual-write to both legacy (vectors) and new (tensors) stores
during the migration period. Once migration is complete, this file can be removed.
"""

from dataclasses import replace

from state.selectors import get_next_color, COLOR_PALETTE
from state.actions import (
    AddVector, DeleteVector, UpdateVector, SelectVector,
    ClearAllVectors, DuplicateVector, DeselectVector,
)
from state.models import VectorData, TensorData


def reduce_vectors(state, action, with_history):
    if isinstance(action, AddVector):
        color, new_color_idx = get_next_color(state)
        actual_color = action.color if action.color != (0.8, 0.2, 0.2) else color
        label = action.label or f"v{state.next_vector_id}"
        next_input_color = COLOR_PALETTE[new_color_idx % len(COLOR_PALETTE)]

        # Create legacy VectorData
        new_vector = VectorData.create(action.coords, actual_color, label)

        # DUAL-WRITE: Also create TensorData with same ID for consistency
        new_tensor = TensorData.create_vector(action.coords, label, actual_color)
        # Use same ID as legacy vector for consistency during transition
        new_tensor = replace(new_tensor, id=new_vector.id)

        new_state = replace(state,
            vectors=state.vectors + (new_vector,),
            tensors=state.tensors + (new_tensor,),  # DUAL-WRITE
            selected_id=new_vector.id,
            selected_type='vector',
            selected_tensor_id=new_vector.id,  # DUAL-WRITE
            next_vector_id=state.next_vector_id + 1,
            next_color_index=new_color_idx,
            input_vector_label="",
            input_vector_coords=(1.0, 0.0, 0.0),
            input_vector_color=next_input_color,
        )
        return with_history(new_state)

    if isinstance(action, DeleteVector):
        new_vectors = tuple(v for v in state.vectors if v.id != action.id)
        # DUAL-WRITE: Also remove from tensors
        new_tensors = tuple(t for t in state.tensors if t.id != action.id)
        new_selected_id = None if state.selected_id == action.id else state.selected_id
        new_selected_type = None if state.selected_id == action.id else state.selected_type
        new_selected_tensor_id = None if state.selected_tensor_id == action.id else state.selected_tensor_id
        new_state = replace(state,
            vectors=new_vectors,
            tensors=new_tensors,  # DUAL-WRITE
            selected_id=new_selected_id,
            selected_type=new_selected_type,
            selected_tensor_id=new_selected_tensor_id,  # DUAL-WRITE
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
        # DUAL-WRITE: Also update in tensors
        new_tensors = tuple(
            replace(t,
                data=tuple(action.coords) if action.coords is not None else t.data,
                shape=(len(action.coords),) if action.coords is not None else t.shape,
                color=action.color if action.color is not None else t.color,
                label=action.label if action.label is not None else t.label,
                visible=action.visible if action.visible is not None else t.visible,
            ) if t.id == action.id and t.rank == 1 else t
            for t in state.tensors
        )
        new_state = replace(state, vectors=new_vectors, tensors=new_tensors)
        return with_history(new_state)

    if isinstance(action, SelectVector):
        # DUAL-WRITE: Set both legacy and tensor selection
        return replace(state, selected_id=action.id, selected_type='vector', selected_tensor_id=action.id)

    if isinstance(action, DeselectVector):
        if state.selected_type == 'vector':
            # DUAL-WRITE: Clear both legacy and tensor selection
            return replace(state, selected_id=None, selected_type=None, selected_tensor_id=None)
        return state

    if isinstance(action, ClearAllVectors):
        new_selected_id = None if state.selected_type == 'vector' else state.selected_id
        new_selected_type = None if state.selected_type == 'vector' else state.selected_type
        # DUAL-WRITE: Also clear rank-1 tensors
        new_tensors = tuple(t for t in state.tensors if t.rank != 1)
        # Clear tensor selection if it was a vector
        current_tensor = None
        if state.selected_tensor_id:
            for t in state.tensors:
                if t.id == state.selected_tensor_id:
                    current_tensor = t
                    break
        new_selected_tensor_id = None if (current_tensor and current_tensor.rank == 1) else state.selected_tensor_id
        new_state = replace(state,
            vectors=(),
            tensors=new_tensors,  # DUAL-WRITE
            selected_id=new_selected_id,
            selected_type=new_selected_type,
            selected_tensor_id=new_selected_tensor_id,  # DUAL-WRITE
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

        # DUAL-WRITE: Also create TensorData with same ID
        new_tensor = TensorData.create_vector(source.coords, new_label, source.color)
        new_tensor = replace(new_tensor, id=new_vector.id)

        new_state = replace(state,
            vectors=state.vectors + (new_vector,),
            tensors=state.tensors + (new_tensor,),  # DUAL-WRITE
            selected_id=new_vector.id,
            selected_type='vector',
            selected_tensor_id=new_vector.id,  # DUAL-WRITE
        )
        return with_history(new_state)

    return None
