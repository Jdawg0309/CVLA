"""
Image kernel reducers.
"""

from dataclasses import replace

from state.actions import ApplyKernel
from state.models import ImageData, EducationalStep


def reduce_image_kernel(state, action, with_history):
    if not isinstance(action, ApplyKernel):
        return None
    if state.current_image is None:
        return state
    try:
        from vision import apply_kernel, get_kernel_by_name

        class TempImageMatrix:
            def __init__(self, data, name):
                self.data = data
                self.name = name
            def as_matrix(self):
                if len(self.data.shape) == 2:
                    return self.data
                return (0.299 * self.data[:, :, 0] +
                        0.587 * self.data[:, :, 1] +
                        0.114 * self.data[:, :, 2])
            @property
            def is_grayscale(self):
                return len(self.data.shape) == 2
            @property
            def height(self):
                return self.data.shape[0]
            @property
            def width(self):
                return self.data.shape[1]
            @property
            def history(self):
                return []

        temp_img = TempImageMatrix(state.current_image.pixels, state.current_image.name)
        result = apply_kernel(temp_img, action.kernel_name, normalize_output=True)

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
        print(f"ApplyKernel error: {e}")
        return state
