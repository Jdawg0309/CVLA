"""
Pipeline action reducers.
"""

from dataclasses import replace
from typing import Optional

from state.actions import StepForward, StepBackward, JumpToStep, ResetPipeline, SetPipeline
from state.models.tensor_model import TensorData, TensorDType
from domain.operations.operation_registry import registry


def _sync_matrix_vector_result(state) -> Optional["AppState"]:
    if not state.operations or not state.operations.steps:
        return state
    if state.operations.current_operation != "matrix_vector_multiply":
        return state
    if not state.operation_history:
        return state

    record = state.operation_history[-1]
    if record.operation_name != "matrix_multiply":
        return state
    if len(record.target_ids) < 2:
        return state
    if not record.result_ids:
        return state

    result_id = record.result_ids[0]
    step_index = state.operations.step_index
    last_index = len(state.operations.steps) - 1
    exists = any(t.id == result_id for t in state.tensors)

    if step_index == last_index:
        if exists:
            return state
        tensor_by_id = {t.id: t for t in state.tensors}
        matrix = tensor_by_id.get(record.target_ids[0])
        vector = tensor_by_id.get(record.target_ids[1])
        if matrix is None or vector is None:
            return state
        op = registry.get("matrix_vector_multiply")
        if op is None:
            return state
        try:
            result_value = op.compute({"matrix": matrix, "vector": vector}, dict(record.parameters))
        except Exception:
            return state
        new_data = tuple(float(x) for x in result_value)
        result_tensor = TensorData(
            id=result_id,
            data=new_data,
            shape=(matrix.rows,),
            dtype=TensorDType.NUMERIC,
            label=f"{matrix.label}*{vector.label}",
            color=vector.color,
            visible=True,
            history=vector.history + ("matrix_multiply",)
        )
        new_tensors = state.tensors + (result_tensor,)
        return replace(state, tensors=new_tensors, selected_tensor_id=result_id)

    if not exists:
        return state

    new_tensors = tuple(t for t in state.tensors if t.id != result_id)
    selected_id = state.selected_tensor_id
    if selected_id == result_id:
        vector_id = record.target_ids[1]
        if any(t.id == vector_id for t in new_tensors):
            selected_id = vector_id
        else:
            selected_id = None
    return replace(state, tensors=new_tensors, selected_tensor_id=selected_id)


def reduce_pipeline(state, action, with_history):
    if isinstance(action, StepForward):
        if state.operations and state.operations.steps:
            max_idx = len(state.operations.steps) - 1
            new_idx = min(state.operations.step_index + 1, max_idx)
            new_state = replace(state, operations=replace(state.operations, step_index=new_idx))
            return _sync_matrix_vector_result(new_state)
        max_idx = len(state.pipeline_steps) - 1
        new_idx = min(state.pipeline_step_index + 1, max_idx)
        return replace(state, pipeline_step_index=new_idx)

    if isinstance(action, StepBackward):
        if state.operations and state.operations.steps:
            new_idx = max(state.operations.step_index - 1, 0)
            new_state = replace(state, operations=replace(state.operations, step_index=new_idx))
            return _sync_matrix_vector_result(new_state)
        new_idx = max(state.pipeline_step_index - 1, 0)
        return replace(state, pipeline_step_index=new_idx)

    if isinstance(action, JumpToStep):
        if state.operations and state.operations.steps:
            if 0 <= action.index < len(state.operations.steps):
                new_state = replace(state, operations=replace(state.operations, step_index=action.index))
                return _sync_matrix_vector_result(new_state)
            return state
        if 0 <= action.index < len(state.pipeline_steps):
            return replace(state, pipeline_step_index=action.index)
        return state

    if isinstance(action, ResetPipeline):
        new_state = replace(state,
            current_image=None,
            processed_image=None,
            pipeline_steps=(),
            pipeline_step_index=0,
        )
        return with_history(new_state)

    if isinstance(action, SetPipeline):
        if not action.steps:
            return replace(state, pipeline_steps=(), pipeline_step_index=0)
        idx = max(0, min(action.index, len(action.steps) - 1))
        return replace(state, pipeline_steps=action.steps, pipeline_step_index=idx)

    return None
