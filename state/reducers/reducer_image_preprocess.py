"""
Image preprocessing reducers.
"""

from dataclasses import replace
from typing import Optional

from domain.images import normalize_image
from domain.images.image_matrix import ImageMatrix

from state.actions import NormalizeImage
from state.models import ImageData, EducationalStep
from state.reducers.image_cache import compute_image_stats, compute_preview_matrix


def _as_image_matrix(image_data: ImageData) -> ImageMatrix:
    return ImageMatrix(image_data.pixels, image_data.name)


def reduce_image_preprocess(state, action, with_history):
    if not isinstance(action, NormalizeImage):
        return None

    if state.current_image is None:
        return state

    mean = action.mean
    std = action.std
    try:
        matrix = _as_image_matrix(state.current_image)
        normalized = normalize_image(matrix, mean=mean, std=std)
        result_data = ImageData.create(
            normalized.data,
            normalized.name,
            state.current_image.history + ("normalize",)
        )
        params = normalized.history[-1][1] if normalized.history else {}
        mean_tag = params.get('mean', mean)
        std_tag = params.get('std', std)
        step = EducationalStep.create(
            title="Normalize Image",
            explanation="Subtract mean and divide by standard deviation.",
            operation="normalize",
            input_data=state.current_image,
            output_data=result_data,
            transform_matrix={'mean': mean_tag, 'std': std_tag},
        )

        stats = compute_image_stats(result_data)
        preview = compute_preview_matrix(result_data)
        new_state = replace(state,
            processed_image=result_data,
            pipeline_steps=state.pipeline_steps + (step,),
            pipeline_step_index=len(state.pipeline_steps),
            processed_image_stats=stats,
            processed_image_preview=preview,
        )
        return with_history(new_state)
    except Exception as exc:
        return replace(state,
            image_status=f"Normalize failed: {exc}",
            image_status_level="error",
        )
