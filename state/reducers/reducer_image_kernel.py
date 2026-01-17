"""
Image kernel reducers.
"""

from dataclasses import replace

from state.actions import ApplyKernel
from engine.image_adapter import ImageDataAdapter
from state.models import ImageData, EducationalStep


def reduce_image_kernel(state, action, with_history):
    if not isinstance(action, ApplyKernel):
        return None
    if state.current_image is None:
        return state
    try:
        from domain.images import apply_kernel, get_kernel_by_name

        adapter = ImageDataAdapter(state.current_image)
        result = apply_kernel(adapter, action.kernel_name, normalize_output=True)

        result_data = ImageData.create(
            result.data,
            f"{state.current_image.name}_{action.kernel_name}",
            state.current_image.history + (f"kernel:{action.kernel_name}",)
        )

        kernel = get_kernel_by_name(action.kernel_name)

        step = EducationalStep.create(
            title=f"Apply {action.kernel_name.replace('_', ' ').title()}",
            explanation="Convolution: slide kernel over image, compute weighted sum at each position.",
            operation="convolution",
            input_data=state.current_image,
            output_data=result_data,
            kernel_name=action.kernel_name,
            kernel_values=kernel,
        )

        new_state = replace(state,
            processed_image=result_data,
            pipeline_steps=state.pipeline_steps + (step,),
            pipeline_step_index=len(state.pipeline_steps),
        )
        return with_history(new_state)
    except Exception as e:
        return replace(state,
            image_status=f"Kernel failed: {e}",
            image_status_level="error",
        )
