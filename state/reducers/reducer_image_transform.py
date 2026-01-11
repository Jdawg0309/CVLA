"""
Image transform reducers.
"""

from dataclasses import replace

from state.actions import ApplyTransform, FlipImageHorizontal
from state.models import ImageData, EducationalStep


def reduce_image_transform(state, action, with_history):
    if isinstance(action, ApplyTransform):
        if state.current_image is None:
            return state
        try:
            from domain.images import AffineTransform, apply_affine_transform

            class TempImageMatrix:
                def __init__(self, data, name):
                    self.data = data
                    self.name = name
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
            h, w = temp_img.height, temp_img.width
            center = (w / 2, h / 2)

            transform = AffineTransform()
            transform.rotate(action.rotation, center)
            transform.scale(action.scale, center=center)

            result = apply_affine_transform(temp_img, transform)

            result_data = ImageData.create(
                result.data,
                f"{state.current_image.name}_transformed",
                state.current_image.history + (f"transform:rot={action.rotation},scale={action.scale}",)
            )

            step = EducationalStep.create(
                title="Affine Transform",
                explanation=f"Rotation {action.rotation:.1f} deg, Scale {action.scale:.2f}x via matrix multiplication.",
                operation="transform",
                input_data=state.current_image,
                output_data=result_data,
                transform_matrix=transform.matrix,
            )

            new_state = replace(state,
                processed_image=result_data,
                pipeline_steps=state.pipeline_steps + (step,),
                pipeline_step_index=len(state.pipeline_steps),
            )
            return with_history(new_state)
        except Exception as e:
            print(f"ApplyTransform error: {e}")
            return state

    if isinstance(action, FlipImageHorizontal):
        if state.current_image is None:
            return state
        try:
            from domain.images import AffineTransform, apply_affine_transform

            class TempImageMatrix:
                def __init__(self, data, name):
                    self.data = data
                    self.name = name
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
            transform = AffineTransform()
            transform.flip_horizontal(temp_img.width)

            result = apply_affine_transform(temp_img, transform)

            result_data = ImageData.create(
                result.data,
                f"{state.current_image.name}_flipped",
                state.current_image.history + ("flip:horizontal",)
            )

            new_state = replace(state,
                processed_image=result_data,
                pipeline_steps=state.pipeline_steps + (EducationalStep.create(
                    title="Flip Horizontal",
                    explanation="Mirror image by negating x coordinates.",
                    operation="transform",
                    input_data=state.current_image,
                    output_data=result_data,
                ),),
                pipeline_step_index=len(state.pipeline_steps),
            )
            return with_history(new_state)
        except Exception:
            return state

    return None
