"""
Basic image state reducers.
"""

from dataclasses import replace

from state.actions import UseResultAsInput, ClearImage


def reduce_image_basic(state, action):
    if isinstance(action, UseResultAsInput):
        if state.processed_image is None:
            return state
        return replace(state,
            current_image=state.processed_image,
            processed_image=None,
        )

    if isinstance(action, ClearImage):
        return replace(state,
            current_image=None,
            processed_image=None,
            pipeline_steps=(),
            pipeline_step_index=0,
        )

    return None
