"""
Basic image state reducers.
"""

from dataclasses import replace

from state.actions import UseResultAsInput, ClearImage
from state.reducers.image_cache import compute_image_stats, compute_preview_matrix


def reduce_image_basic(state, action):
    if isinstance(action, UseResultAsInput):
        if state.processed_image is None:
            return state
        stats = compute_image_stats(state.processed_image)
        preview = compute_preview_matrix(state.processed_image)
        return replace(state,
            current_image=state.processed_image,
            processed_image=None,
            current_image_stats=stats,
            processed_image_stats=None,
            current_image_preview=preview,
            processed_image_preview=None,
        )

    if isinstance(action, ClearImage):
        return replace(state,
            current_image=None,
            processed_image=None,
            pipeline_steps=(),
            pipeline_step_index=0,
            current_image_stats=None,
            processed_image_stats=None,
            current_image_preview=None,
            processed_image_preview=None,
        )

    return None
