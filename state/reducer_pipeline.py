"""
Pipeline action reducers.
"""

from dataclasses import replace

from state.actions import StepForward, StepBackward, JumpToStep, ResetPipeline


def reduce_pipeline(state, action, with_history):
    if isinstance(action, StepForward):
        max_idx = len(state.pipeline_steps) - 1
        new_idx = min(state.pipeline_step_index + 1, max_idx)
        return replace(state, pipeline_step_index=new_idx)

    if isinstance(action, StepBackward):
        new_idx = max(state.pipeline_step_index - 1, 0)
        return replace(state, pipeline_step_index=new_idx)

    if isinstance(action, JumpToStep):
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

    return None
