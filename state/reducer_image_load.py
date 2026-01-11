"""
Image loading reducers.
"""

from dataclasses import replace

from state.actions import LoadImage, CreateSampleImage
from state.models import ImageData, EducationalStep


def _auto_fit_scale(image_data, grid_size=20.0, margin=0.9):
    """Compute a render scale that fits the image within the grid bounds."""
    if image_data is None:
        return 1.0
    max_dim = max(float(image_data.width), float(image_data.height), 1.0)
    fit = (2.0 * grid_size * margin) / max_dim
    return min(1.0, max(0.05, fit))


def reduce_image_load(state, action, with_history):
    if isinstance(action, LoadImage):
        max_size = action.max_size
        if max_size is None and state.image_downsample_enabled:
            max_size = (state.image_preview_resolution, state.image_preview_resolution)
        try:
            from vision import load_image as vision_load_image
            img = vision_load_image(action.path, max_size=max_size, grayscale=True)
            if img is None:
                return replace(state,
                    image_status="Failed to load image. Check the path and Pillow install.",
                    image_status_level="error",
                )
            image_data = ImageData.create(img.data, img.name)
            render_scale = state.image_render_scale
            if state.image_auto_fit:
                render_scale = _auto_fit_scale(image_data)
            new_state = replace(state,
                current_image=image_data,
                processed_image=None,
                pipeline_steps=(),
                pipeline_step_index=0,
                image_status=f"Loaded image '{image_data.name}' ({image_data.width}x{image_data.height})",
                image_status_level="info",
                image_render_scale=render_scale,
                show_image_on_grid=True,
                selected_pixel=(0, 0),
            )
            return with_history(new_state)
        except Exception as e:
            return replace(state,
                image_status=f"Failed to load image: {e}",
                image_status_level="error",
            )

    if isinstance(action, CreateSampleImage):
        try:
            from vision import create_sample_image
            img = create_sample_image(action.size, action.pattern)
            image_data = ImageData.create(img.data, img.name)

            step = EducationalStep.create(
                title="Load Image",
                explanation=f"Created {action.size}x{action.size} {action.pattern} image. Each pixel is a number 0-1.",
                operation="load",
                output_data=image_data,
            )

            render_scale = state.image_render_scale
            if state.image_auto_fit:
                render_scale = _auto_fit_scale(image_data)

            new_state = replace(state,
                current_image=image_data,
                processed_image=None,
                pipeline_steps=(step,),
                pipeline_step_index=0,
                image_status=f"Created sample image '{image_data.name}' ({image_data.width}x{image_data.height})",
                image_status_level="info",
                image_render_scale=render_scale,
                show_image_on_grid=True,
                selected_pixel=(0, 0),
            )
            return with_history(new_state)
        except Exception:
            return replace(state,
                image_status="Failed to create sample image.",
                image_status_level="error",
            )

    return None
