"""
Image action reducers (consolidated).
"""

from dataclasses import replace

from domain.images import normalize_image
from domain.images.image_matrix import ImageMatrix

from engine.image_adapter import ImageDataAdapter
from state.actions import (
    LoadImage,
    CreateSampleImage,
    ApplyKernel,
    ApplyTransform,
    FlipImageHorizontal,
    NormalizeImage,
    UseResultAsInput,
    ClearImage,
)
from state.models import ImageData, EducationalStep
from state.models.tensor_model import TensorData
from state.reducers.image_cache import compute_image_stats, compute_preview_matrix


def _auto_fit_scale(image_data, grid_size=20.0, margin=0.9):
    """Compute a render scale that fits the image within the grid bounds."""
    if image_data is None:
        return 1.0
    max_dim = max(float(image_data.width), float(image_data.height), 1.0)
    fit = (2.0 * grid_size * margin) / max_dim
    return min(1.0, max(0.05, fit))


class _TempImageMatrix:
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


def _as_image_matrix(image_data: ImageData) -> ImageMatrix:
    return ImageMatrix(image_data.pixels, image_data.name)


def reduce_image_load(state, action, with_history):
    if isinstance(action, LoadImage):
        max_size = action.max_size
        if max_size is None and state.image_downsample_enabled:
            max_size = (state.image_preview_resolution, state.image_preview_resolution)
        try:
            from domain.images import load_image as vision_load_image
            img = vision_load_image(action.path, max_size=max_size, grayscale=False)
            if img is None:
                return replace(state,
                    image_status="Failed to load image. Check the path and Pillow install.",
                    image_status_level="error",
                )
            image_data = ImageData.create(img.data, img.name)
            render_scale = state.image_render_scale
            if state.image_auto_fit:
                render_scale = _auto_fit_scale(image_data)
            stats = compute_image_stats(image_data)
            preview = compute_preview_matrix(image_data)
            tensor = TensorData.create_image(image_data.pixels, image_data.name)

            new_state = replace(state,
                current_image=image_data,
                processed_image=None,
                tensors=state.tensors + (tensor,),
                selected_tensor_id=tensor.id,
                pipeline_steps=(),
                pipeline_step_index=0,
                image_status=f"Loaded image '{image_data.name}' ({image_data.width}x{image_data.height})",
                image_status_level="info",
                image_render_scale=render_scale,
                show_image_on_grid=True,
                selected_pixel=(0, 0),
                current_image_stats=stats,
                processed_image_stats=None,
                current_image_preview=preview,
                processed_image_preview=None,
            )
            return with_history(new_state)
        except Exception as e:
            return replace(state,
                image_status=f"Failed to load image: {e}",
                image_status_level="error",
            )

    if isinstance(action, CreateSampleImage):
        try:
            from domain.images import create_sample_image
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

            stats = compute_image_stats(image_data)
            preview = compute_preview_matrix(image_data)
            tensor = TensorData.create_image(image_data.pixels, image_data.name)

            new_state = replace(state,
                current_image=image_data,
                processed_image=None,
                tensors=state.tensors + (tensor,),
                selected_tensor_id=tensor.id,
                pipeline_steps=(step,),
                pipeline_step_index=0,
                image_status=f"Created sample image '{image_data.name}' ({image_data.width}x{image_data.height})",
                image_status_level="info",
                image_render_scale=render_scale,
                show_image_on_grid=True,
                selected_pixel=(0, 0),
                current_image_stats=stats,
                processed_image_stats=None,
                current_image_preview=preview,
                processed_image_preview=None,
            )
            return with_history(new_state)
        except Exception:
            return replace(state,
                image_status="Failed to create sample image.",
                image_status_level="error",
            )

    return None


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
    except Exception as e:
        return replace(state,
            image_status=f"Kernel failed: {e}",
            image_status_level="error",
        )


def reduce_image_transform(state, action, with_history):
    if isinstance(action, ApplyTransform):
        if state.current_image is None:
            return state
        try:
            from domain.images import AffineTransform, apply_affine_transform

            temp_img = _TempImageMatrix(state.current_image.pixels, state.current_image.name)
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
        except Exception as e:
            return replace(state,
                image_status=f"Transform failed: {e}",
                image_status_level="error",
            )

    if isinstance(action, FlipImageHorizontal):
        if state.current_image is None:
            return state
        try:
            from domain.images import AffineTransform, apply_affine_transform

            temp_img = _TempImageMatrix(state.current_image.pixels, state.current_image.name)
            transform = AffineTransform()
            transform.flip_horizontal(temp_img.width)

            result = apply_affine_transform(temp_img, transform)

            result_data = ImageData.create(
                result.data,
                f"{state.current_image.name}_flipped",
                state.current_image.history + ("flip:horizontal",)
            )

            stats = compute_image_stats(result_data)
            preview = compute_preview_matrix(result_data)
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
                processed_image_stats=stats,
                processed_image_preview=preview,
            )
            return with_history(new_state)
        except Exception as e:
            return replace(state,
                image_status=f"Transform failed: {e}",
                image_status_level="error",
            )

    return None


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


def reduce_images(state, action, with_history):
    result = reduce_image_load(state, action, with_history)
    if result is not None:
        return result

    result = reduce_image_kernel(state, action, with_history)
    if result is not None:
        return result

    result = reduce_image_transform(state, action, with_history)
    if result is not None:
        return result

    result = reduce_image_preprocess(state, action, with_history)
    if result is not None:
        return result

    result = reduce_image_basic(state, action)
    if result is not None:
        return result

    return None
