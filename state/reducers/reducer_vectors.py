"""
Vector action reducers.

Writes to unified tensor store only.
"""

from dataclasses import replace
from uuid import uuid4

from state.selectors import get_next_color, COLOR_PALETTE
from state.actions import (
    AddVector, DeleteVector, UpdateVector, SelectVector,
    ClearAllVectors, DuplicateVector, DeselectVector,
)
from state.models import TensorData


def reduce_vectors(state, action, with_history):
    if isinstance(action, AddVector):
        color, new_color_idx = get_next_color(state)
        actual_color = action.color if action.color != (0.8, 0.2, 0.2) else color
        label = action.label or f"v{state.next_vector_id}"
        next_input_color = COLOR_PALETTE[new_color_idx % len(COLOR_PALETTE)]

        # Create TensorData (unified storage)
        new_tensor = TensorData.create_vector(action.coords, label, actual_color)
        vector_id = f"vec_{uuid4().hex[:8]}"
        new_tensor = replace(new_tensor, id=vector_id)

        new_state = replace(state,
            tensors=state.tensors + (new_tensor,),
            selected_tensor_id=vector_id,
            next_vector_id=state.next_vector_id + 1,
            next_color_index=new_color_idx,
            input_vector_label="",
            input_vector_coords=(1.0, 0.0, 0.0),
            input_vector_color=next_input_color,
        )
        return with_history(new_state)

    if isinstance(action, DeleteVector):
        new_tensors = tuple(t for t in state.tensors if t.id != action.id)
        new_selected_tensor_id = None if state.selected_tensor_id == action.id else state.selected_tensor_id
        new_state = replace(state,
            tensors=new_tensors,
            selected_tensor_id=new_selected_tensor_id,
        )
        return with_history(new_state)

    if isinstance(action, UpdateVector):
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
        new_state = replace(state, tensors=new_tensors)
        return with_history(new_state)

    if isinstance(action, SelectVector):
        return replace(state, selected_tensor_id=action.id)

    if isinstance(action, DeselectVector):
        # Check if current selection is a vector (rank-1 tensor)
        current_tensor = None
        if state.selected_tensor_id:
            for t in state.tensors:
                if t.id == state.selected_tensor_id:
                    current_tensor = t
                    break
        if current_tensor and current_tensor.rank == 1:
            return replace(state, selected_tensor_id=None)
        return state

    if isinstance(action, ClearAllVectors):
        # Clear rank-1 tensors
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
            tensors=new_tensors,
            selected_tensor_id=new_selected_tensor_id,
        )
        return with_history(new_state)

    if isinstance(action, DuplicateVector):
        source = None
        for t in state.tensors:
            if t.id == action.id and t.rank == 1:
                source = t
                break
        if source is None:
            return state

        new_label = f"{source.label}_copy"
        new_tensor = TensorData.create_vector(source.data, new_label, source.color)
        new_id = f"vec_{uuid4().hex[:8]}"
        new_tensor = replace(new_tensor, id=new_id)

        new_state = replace(state,
            tensors=state.tensors + (new_tensor,),
            selected_tensor_id=new_id,
        )
        return with_history(new_state)

    return None
