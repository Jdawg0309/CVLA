"""
Reducer - The ONLY place where state changes happen

The reducer is a pure function: (state, action) -> new_state

RULES:
1. NEVER mutate the input state
2. ALWAYS return a new AppState via dataclasses.replace()
3. Handle ALL action types
4. Unknown actions return state unchanged
"""

from dataclasses import replace
from typing import Callable
import numpy as np

from .app_state import AppState, MAX_HISTORY, get_next_color
from .models import VectorData, MatrixData, ImageData, EducationalStep
from .actions import (
    Action,
    # Vector actions
    AddVector, DeleteVector, UpdateVector, SelectVector,
    # Matrix actions
    AddMatrix, DeleteMatrix, UpdateMatrixCell, SelectMatrix,
    ApplyMatrixToSelected, ApplyMatrixToAll,
    # Image actions
    LoadImage, CreateSampleImage, ApplyKernel, ApplyTransform,
    FlipImageHorizontal, UseResultAsInput, ClearImage,
    # Pipeline actions
    StepForward, StepBackward, JumpToStep, ResetPipeline,
    # UI input actions
    SetInputVector, SetInputMatrixCell, SetInputMatrixSize, SetInputMatrixLabel,
    SetImagePath, SetSamplePattern, SetSampleSize,
    SetTransformRotation, SetTransformScale, SetSelectedKernel,
    SetImageRenderScale, SetImageRenderMode, SetImageColorMode, ToggleImageGridOverlay,
    ToggleImageDownsample, SetImagePreviewResolution,
    # Navigation actions
    SetActiveTab, ToggleMatrixEditor, ToggleMatrixValues, ToggleImageOnGrid, TogglePreview,
    ClearSelection,
    # History actions
    Undo, Redo,
)


def _auto_fit_scale(image_data, grid_size=20.0, margin=0.9):
    """Compute a render scale that fits the image within the grid bounds."""
    if image_data is None:
        return 1.0
    max_dim = max(float(image_data.width), float(image_data.height), 1.0)
    fit = (2.0 * grid_size * margin) / max_dim
    return min(1.0, max(0.05, fit))

def reduce(state: AppState, action: Action) -> AppState:
    """
    Pure reducer function.

    Takes current state and an action, returns new state.
    NEVER mutates the input state.
    """

    # =========================================================================
    # HISTORY ACTIONS (special handling - no history push)
    # =========================================================================

    if isinstance(action, Undo):
        if not state.history:
            return state  # Nothing to undo
        # Pop from history, push current to future
        previous = state.history[-1]
        new_history = state.history[:-1]
        new_future = (state,) + state.future
        # Restore previous state but keep history/future updated
        return replace(previous, history=new_history, future=new_future)

    if isinstance(action, Redo):
        if not state.future:
            return state  # Nothing to redo
        # Pop from future, push current to history
        next_state = state.future[0]
        new_future = state.future[1:]
        new_history = state.history + (state,)
        # Restore next state but keep history/future updated
        return replace(next_state, history=new_history, future=new_future)

    # =========================================================================
    # For all other actions, push current state to history (for undo)
    # Clear future (new action invalidates redo stack)
    # =========================================================================

    def with_history(new_state: AppState) -> AppState:
        """Helper to add current state to history."""
        # Don't store input-only changes in history
        if isinstance(action, (SetInputVector, SetInputMatrixCell, SetInputMatrixSize,
                               SetInputMatrixLabel, SetImagePath, SetSamplePattern,
                               SetSampleSize, SetTransformRotation, SetTransformScale,
                               SetSelectedKernel, SetActiveTab, ToggleMatrixEditor,
                               ToggleMatrixValues, TogglePreview, ClearSelection,
                               StepForward, StepBackward, JumpToStep)):
            return new_state  # No history for UI-only changes

        new_history = (state.history + (state,))[-MAX_HISTORY:]
        return replace(new_state, history=new_history, future=())

    # =========================================================================
    # VECTOR ACTIONS
    # =========================================================================

    if isinstance(action, AddVector):
        color, new_color_idx = get_next_color(state)
        # Use provided color or auto-color
        actual_color = action.color if action.color != (0.8, 0.2, 0.2) else color
        label = action.label or f"v{state.next_vector_id}"

        new_vector = VectorData.create(action.coords, actual_color, label)
        new_state = replace(state,
            vectors=state.vectors + (new_vector,),
            selected_id=new_vector.id,
            selected_type='vector',
            next_vector_id=state.next_vector_id + 1,
            next_color_index=new_color_idx,
            # Reset input fields
            input_vector_label="",
            input_vector_coords=(1.0, 0.0, 0.0),
        )
        return with_history(new_state)

    if isinstance(action, DeleteVector):
        new_vectors = tuple(v for v in state.vectors if v.id != action.id)
        # Clear selection if deleted item was selected
        new_selected_id = None if state.selected_id == action.id else state.selected_id
        new_selected_type = None if state.selected_id == action.id else state.selected_type
        new_state = replace(state,
            vectors=new_vectors,
            selected_id=new_selected_id,
            selected_type=new_selected_type,
        )
        return with_history(new_state)

    if isinstance(action, UpdateVector):
        new_vectors = tuple(
            replace(v,
                coords=action.coords if action.coords is not None else v.coords,
                color=action.color if action.color is not None else v.color,
                label=action.label if action.label is not None else v.label,
                visible=action.visible if action.visible is not None else v.visible,
            ) if v.id == action.id else v
            for v in state.vectors
        )
        new_state = replace(state, vectors=new_vectors)
        return with_history(new_state)

    if isinstance(action, SelectVector):
        return replace(state, selected_id=action.id, selected_type='vector')

    # =========================================================================
    # MATRIX ACTIONS
    # =========================================================================

    if isinstance(action, AddMatrix):
        label = action.label or f"M{state.next_matrix_id}"
        new_matrix = MatrixData(
            id=str(__import__('uuid').uuid4()),
            values=action.values,
            label=label,
            visible=True
        )
        new_state = replace(state,
            matrices=state.matrices + (new_matrix,),
            selected_id=new_matrix.id,
            selected_type='matrix',
            next_matrix_id=state.next_matrix_id + 1,
            show_matrix_editor=False,
        )
        return with_history(new_state)

    if isinstance(action, DeleteMatrix):
        new_matrices = tuple(m for m in state.matrices if m.id != action.id)
        new_selected_id = None if state.selected_id == action.id else state.selected_id
        new_selected_type = None if state.selected_id == action.id else state.selected_type
        new_state = replace(state,
            matrices=new_matrices,
            selected_id=new_selected_id,
            selected_type=new_selected_type,
        )
        return with_history(new_state)

    if isinstance(action, UpdateMatrixCell):
        new_matrices = tuple(
            m.with_cell(action.row, action.col, action.value) if m.id == action.id else m
            for m in state.matrices
        )
        new_state = replace(state, matrices=new_matrices)
        return with_history(new_state)

    if isinstance(action, SelectMatrix):
        return replace(state, selected_id=action.id, selected_type='matrix')

    if isinstance(action, ApplyMatrixToSelected):
        # Find the matrix and selected vector
        matrix = None
        for m in state.matrices:
            if m.id == action.matrix_id:
                matrix = m
                break
        if matrix is None or state.selected_type != 'vector':
            return state

        mat_np = matrix.to_numpy()
        new_vectors = []
        for v in state.vectors:
            if v.id == state.selected_id:
                vec_np = v.to_numpy()
                result = mat_np @ vec_np
                new_coords = (float(result[0]), float(result[1]), float(result[2]))
                new_vectors.append(replace(v, coords=new_coords))
            else:
                new_vectors.append(v)

        new_state = replace(state, vectors=tuple(new_vectors))
        return with_history(new_state)

    if isinstance(action, ApplyMatrixToAll):
        matrix = None
        for m in state.matrices:
            if m.id == action.matrix_id:
                matrix = m
                break
        if matrix is None:
            return state

        mat_np = matrix.to_numpy()
        new_vectors = []
        for v in state.vectors:
            if v.visible:
                vec_np = v.to_numpy()
                result = mat_np @ vec_np
                new_coords = (float(result[0]), float(result[1]), float(result[2]))
                new_vectors.append(replace(v, coords=new_coords))
            else:
                new_vectors.append(v)

        new_state = replace(state, vectors=tuple(new_vectors))
        return with_history(new_state)

    # =========================================================================
    # IMAGE ACTIONS
    # =========================================================================

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

            # Create educational step
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

    if isinstance(action, ApplyKernel):
        if state.current_image is None:
            return state
        try:
            from vision import apply_kernel, get_kernel_by_name

            # Create ImageMatrix wrapper for vision module
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

            # Get kernel values for educational step
            kernel = get_kernel_by_name(action.kernel_name)
            kernel_values = tuple(tuple(row) for row in kernel)

            step = EducationalStep.create(
                title=f"Apply {action.kernel_name.replace('_', ' ').title()}",
                explanation=f"Convolution: slide kernel over image, compute weighted sum at each position.",
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

    if isinstance(action, ApplyTransform):
        if state.current_image is None:
            return state
        try:
            from vision import AffineTransform, apply_affine_transform

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
                title=f"Affine Transform",
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
            from vision import AffineTransform, apply_affine_transform

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

    # =========================================================================
    # PIPELINE ACTIONS
    # =========================================================================

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

    # =========================================================================
    # UI INPUT ACTIONS (no history)
    # =========================================================================

    if isinstance(action, SetInputVector):
        return replace(state,
            input_vector_coords=action.coords if action.coords is not None else state.input_vector_coords,
            input_vector_label=action.label if action.label is not None else state.input_vector_label,
            input_vector_color=action.color if action.color is not None else state.input_vector_color,
        )

    if isinstance(action, SetInputMatrixCell):
        # Update the input matrix (for the editor form)
        new_matrix = list(list(row) for row in state.input_matrix)
        # Ensure matrix is large enough
        while len(new_matrix) <= action.row:
            new_matrix.append([0.0] * state.input_matrix_size)
        while len(new_matrix[action.row]) <= action.col:
            new_matrix[action.row].append(0.0)
        new_matrix[action.row][action.col] = action.value
        return replace(state, input_matrix=tuple(tuple(row) for row in new_matrix))

    if isinstance(action, SetInputMatrixSize):
        # Resize the input matrix
        old = state.input_matrix
        new_size = action.size
        new_matrix = []
        for i in range(new_size):
            row = []
            for j in range(new_size):
                if i < len(old) and j < len(old[i]):
                    row.append(old[i][j])
                else:
                    row.append(1.0 if i == j else 0.0)
            new_matrix.append(tuple(row))
        return replace(state, input_matrix=tuple(new_matrix), input_matrix_size=new_size)

    if isinstance(action, SetInputMatrixLabel):
        return replace(state, input_matrix_label=action.label)

    if isinstance(action, SetImagePath):
        return replace(state, input_image_path=action.path)

    if isinstance(action, SetSamplePattern):
        return replace(state, input_sample_pattern=action.pattern)

    if isinstance(action, SetSampleSize):
        return replace(state, input_sample_size=action.size)

    if isinstance(action, SetTransformRotation):
        return replace(state, input_transform_rotation=action.rotation)

    if isinstance(action, SetTransformScale):
        return replace(state, input_transform_scale=action.scale)

    if isinstance(action, SetSelectedKernel):
        return replace(state, selected_kernel=action.kernel_name)

    # =========================================================================
    # NAVIGATION ACTIONS (no history)
    # =========================================================================

    if isinstance(action, SetActiveTab):
        return replace(state, active_tab=action.tab)

    if isinstance(action, ToggleMatrixEditor):
        return replace(state, show_matrix_editor=not state.show_matrix_editor)

    if isinstance(action, ToggleMatrixValues):
        return replace(state, show_matrix_values=not state.show_matrix_values)

    if isinstance(action, TogglePreview):
        return replace(state, preview_enabled=not state.preview_enabled)

    if isinstance(action, SetImageRenderScale):
        return replace(state, image_render_scale=max(0.05, float(action.scale)))

    if isinstance(action, SetImageRenderMode):
        mode = action.mode if action.mode in ("plane", "height-field") else "plane"
        return replace(state, image_render_mode=mode)

    if isinstance(action, SetImageColorMode):
        mode = action.mode if action.mode in ("grayscale", "heatmap") else "grayscale"
        return replace(state, image_color_mode=mode)

    if isinstance(action, ToggleImageGridOverlay):
        return replace(state, show_image_grid_overlay=not state.show_image_grid_overlay)

    if isinstance(action, ToggleImageDownsample):
        return replace(state, image_downsample_enabled=not state.image_downsample_enabled)

    if isinstance(action, SetImagePreviewResolution):
        size = max(16, min(int(action.size), 1024))
        return replace(state, image_preview_resolution=size)

    if isinstance(action, ToggleImageOnGrid):
        return replace(state, show_image_on_grid=not state.show_image_on_grid)

    if isinstance(action, ClearSelection):
        return replace(state, selected_id=None, selected_type=None)

    # =========================================================================
    # UNKNOWN ACTION - return unchanged
    # =========================================================================
    return state


# =============================================================================
# DISPATCH HELPER
# =============================================================================

class Store:
    """
    Simple store that holds state and processes actions.

    Usage:
        store = Store(create_initial_state())
        store.dispatch(AddVector(...))
        current_state = store.get_state()
    """

    def __init__(self, initial_state: AppState):
        self._state = initial_state
        self._listeners: list = []

    def get_state(self) -> AppState:
        """Get current state (read-only)."""
        return self._state

    def dispatch(self, action: Action) -> None:
        """Dispatch an action to update state."""
        self._state = reduce(self._state, action)
        # Notify listeners
        for listener in self._listeners:
            listener(self._state)

    def subscribe(self, listener: Callable[[AppState], None]) -> Callable[[], None]:
        """Subscribe to state changes. Returns unsubscribe function."""
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)
