"""
Reducer for tensor actions.

Handles CRUD operations on the unified tensor model.
"""

from dataclasses import replace
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from state.app_state import AppState

from state.actions import Action
from state.actions.tensor_actions import (
    AddTensor, DeleteTensor, UpdateTensor, SelectTensor, DeselectTensor,
    ApplyOperation, PreviewOperation, CancelPreview, ConfirmPreview,
    ClearAllTensors, DuplicateTensor,
    AddVectorTensor, AddMatrixTensor, AddImageTensor,
)
from state.models.tensor_model import TensorData, TensorDType
from state.models.operation_record import OperationRecord


def reduce_tensors(
    state: "AppState",
    action: Action,
    with_history: Callable[["AppState"], "AppState"]
) -> Optional["AppState"]:
    """
    Reduce tensor-related actions.

    Returns:
        New state if action was handled, None otherwise.
    """
    if isinstance(action, AddTensor):
        return _handle_add_tensor(state, action, with_history)

    if isinstance(action, DeleteTensor):
        return _handle_delete_tensor(state, action, with_history)

    if isinstance(action, UpdateTensor):
        return _handle_update_tensor(state, action, with_history)

    if isinstance(action, SelectTensor):
        return _handle_select_tensor(state, action)

    if isinstance(action, DeselectTensor):
        return _handle_deselect_tensor(state)

    if isinstance(action, ClearAllTensors):
        return _handle_clear_all_tensors(state, with_history)

    if isinstance(action, DuplicateTensor):
        return _handle_duplicate_tensor(state, action, with_history)

    if isinstance(action, AddVectorTensor):
        return _handle_add_vector_tensor(state, action, with_history)

    if isinstance(action, AddMatrixTensor):
        return _handle_add_matrix_tensor(state, action, with_history)

    if isinstance(action, AddImageTensor):
        return _handle_add_image_tensor(state, action, with_history)

    if isinstance(action, ApplyOperation):
        return _handle_apply_operation(state, action, with_history)

    if isinstance(action, PreviewOperation):
        return _handle_preview_operation(state, action)

    if isinstance(action, CancelPreview):
        return _handle_cancel_preview(state)

    if isinstance(action, ConfirmPreview):
        return _handle_confirm_preview(state, with_history)

    return None


def _handle_add_tensor(
    state: "AppState",
    action: AddTensor,
    with_history: Callable
) -> "AppState":
    """Add a new tensor."""
    dtype_map = {
        "numeric": TensorDType.NUMERIC,
        "image_rgb": TensorDType.IMAGE_RGB,
        "image_grayscale": TensorDType.IMAGE_GRAYSCALE,
    }
    dtype = dtype_map.get(action.dtype, TensorDType.NUMERIC)

    tensor = TensorData(
        id=_generate_id(),
        data=action.data,
        shape=action.shape,
        dtype=dtype,
        label=action.label,
        color=action.color,
        visible=True,
        history=()
    )

    new_state = replace(
        state,
        tensors=state.tensors + (tensor,),
        selected_tensor_id=tensor.id
    )
    return with_history(new_state)


def _handle_delete_tensor(
    state: "AppState",
    action: DeleteTensor,
    with_history: Callable
) -> "AppState":
    """Delete a tensor by ID."""
    new_tensors = tuple(t for t in state.tensors if t.id != action.id)
    new_selected = state.selected_tensor_id if state.selected_tensor_id != action.id else None

    new_state = replace(
        state,
        tensors=new_tensors,
        selected_tensor_id=new_selected
    )
    return with_history(new_state)


def _handle_update_tensor(
    state: "AppState",
    action: UpdateTensor,
    with_history: Callable
) -> "AppState":
    """Update tensor properties."""
    new_tensors = []
    for t in state.tensors:
        if t.id == action.id:
            updated = t
            if action.data is not None:
                shape = action.shape if action.shape is not None else t.shape
                updated = replace(updated, data=action.data, shape=shape)
            if action.label is not None:
                updated = replace(updated, label=action.label)
            if action.color is not None:
                updated = replace(updated, color=action.color)
            if action.visible is not None:
                updated = replace(updated, visible=action.visible)
            new_tensors.append(updated)
        else:
            new_tensors.append(t)

    new_state = replace(state, tensors=tuple(new_tensors))
    return with_history(new_state)


def _handle_select_tensor(state: "AppState", action: SelectTensor) -> "AppState":
    """Select a tensor by ID."""
    # Verify tensor exists
    exists = any(t.id == action.id for t in state.tensors)
    if not exists:
        return state
    return replace(state, selected_tensor_id=action.id)


def _handle_deselect_tensor(state: "AppState") -> "AppState":
    """Clear tensor selection."""
    return replace(state, selected_tensor_id=None)


def _handle_clear_all_tensors(
    state: "AppState",
    with_history: Callable
) -> "AppState":
    """Clear all tensors."""
    new_state = replace(
        state,
        tensors=(),
        selected_tensor_id=None
    )
    return with_history(new_state)


def _handle_duplicate_tensor(
    state: "AppState",
    action: DuplicateTensor,
    with_history: Callable
) -> "AppState":
    """Duplicate an existing tensor."""
    source = None
    for t in state.tensors:
        if t.id == action.id:
            source = t
            break

    if source is None:
        return state

    label = action.new_label if action.new_label else f"{source.label}_copy"
    duplicate = TensorData(
        id=_generate_id(),
        data=source.data,
        shape=source.shape,
        dtype=source.dtype,
        label=label,
        color=source.color,
        visible=source.visible,
        history=source.history
    )

    new_state = replace(
        state,
        tensors=state.tensors + (duplicate,),
        selected_tensor_id=duplicate.id
    )
    return with_history(new_state)


def _handle_add_vector_tensor(
    state: "AppState",
    action: AddVectorTensor,
    with_history: Callable
) -> "AppState":
    """Add a vector tensor (convenience handler)."""
    tensor = TensorData.create_vector(
        coords=action.coords,
        label=action.label,
        color=action.color
    )

    new_state = replace(
        state,
        tensors=state.tensors + (tensor,),
        selected_tensor_id=tensor.id
    )
    return with_history(new_state)


def _handle_add_matrix_tensor(
    state: "AppState",
    action: AddMatrixTensor,
    with_history: Callable
) -> "AppState":
    """Add a matrix tensor (convenience handler)."""
    tensor = TensorData.create_matrix(
        values=action.values,
        label=action.label,
        color=action.color
    )

    new_state = replace(
        state,
        tensors=state.tensors + (tensor,),
        selected_tensor_id=tensor.id
    )
    return with_history(new_state)


def _handle_add_image_tensor(
    state: "AppState",
    action: AddImageTensor,
    with_history: Callable
) -> "AppState":
    """Add an image tensor from file or sample."""
    import numpy as np

    if action.source == "file":
        # Load image from file
        try:
            from domain.images.io.image_loader import load_image
            pixels = load_image(action.path)
            label = action.label if action.label else action.path.split("/")[-1]
        except Exception:
            return state
    else:
        # Create sample pattern
        from domain.images.io.image_loader import create_sample_image
        pixels = create_sample_image(action.size, action.pattern)
        label = action.label if action.label else f"{action.pattern}_{action.size}"

    tensor = TensorData.create_image(pixels=pixels, name=label)

    new_state = replace(
        state,
        tensors=state.tensors + (tensor,),
        selected_tensor_id=tensor.id
    )
    return with_history(new_state)


def _handle_apply_operation(
    state: "AppState",
    action: ApplyOperation,
    with_history: Callable
) -> "AppState":
    """Apply an operation to tensors."""
    # Find target tensors
    targets = [t for t in state.tensors if t.id in action.target_ids]
    if not targets:
        return state

    # Apply operation (placeholder - actual implementation depends on operation)
    result_tensors = _execute_operation(
        action.operation_name,
        action.parameters,
        targets,
        action.create_new
    )

    if result_tensors is None:
        return state

    # Create operation record
    record = OperationRecord.create(
        operation_name=action.operation_name,
        parameters=action.parameters,
        target_ids=action.target_ids,
        result_ids=tuple(t.id for t in result_tensors if action.create_new),
    )

    # Update state
    if action.create_new:
        new_tensors = state.tensors + tuple(result_tensors)
    else:
        # Replace existing tensors
        new_tensors = []
        result_map = {t.id: t for t in result_tensors}
        for t in state.tensors:
            if t.id in result_map:
                new_tensors.append(result_map[t.id])
            else:
                new_tensors.append(t)
        new_tensors = tuple(new_tensors)

    new_state = replace(
        state,
        tensors=new_tensors,
        operation_history=state.operation_history + (record,)
    )
    return with_history(new_state)


def _handle_preview_operation(state: "AppState", action: PreviewOperation) -> "AppState":
    """Preview an operation result."""
    target = None
    for t in state.tensors:
        if t.id == action.target_id:
            target = t
            break

    if target is None:
        return state

    # Execute operation to get preview
    results = _execute_operation(
        action.operation_name,
        action.parameters,
        [target],
        create_new=True
    )

    if results is None or not results:
        return state

    return replace(
        state,
        pending_operation=action.operation_name,
        pending_operation_params=action.parameters,
        operation_preview_tensor=results[0]
    )


def _handle_cancel_preview(state: "AppState") -> "AppState":
    """Cancel operation preview."""
    return replace(
        state,
        pending_operation=None,
        pending_operation_params=(),
        operation_preview_tensor=None
    )


def _handle_confirm_preview(
    state: "AppState",
    with_history: Callable
) -> "AppState":
    """Confirm and apply previewed operation."""
    if state.operation_preview_tensor is None:
        return state

    # Add the previewed tensor
    new_tensor = state.operation_preview_tensor

    # Create operation record
    record = OperationRecord.create(
        operation_name=state.pending_operation or "unknown",
        parameters=state.pending_operation_params,
        target_ids=(),
        result_ids=(new_tensor.id,),
    )

    new_state = replace(
        state,
        tensors=state.tensors + (new_tensor,),
        selected_tensor_id=new_tensor.id,
        operation_history=state.operation_history + (record,),
        pending_operation=None,
        pending_operation_params=(),
        operation_preview_tensor=None
    )
    return with_history(new_state)


def _execute_operation(
    operation_name: str,
    parameters: tuple,
    targets: list,
    create_new: bool
) -> Optional[list]:
    """
    Execute an operation on tensors.

    This is a dispatcher that routes to specific operation implementations.
    """
    # Convert parameters to dict for easier access
    params = dict(parameters)

    # Vector operations
    if operation_name == "normalize":
        return _op_normalize(targets, params, create_new)
    if operation_name == "scale":
        return _op_scale(targets, params, create_new)

    # Matrix operations
    if operation_name == "transpose":
        return _op_transpose(targets, params, create_new)
    if operation_name == "inverse":
        return _op_inverse(targets, params, create_new)

    # Image operations
    if operation_name == "apply_kernel":
        return _op_apply_kernel(targets, params, create_new)

    return None


def _op_normalize(targets: list, params: dict, create_new: bool) -> list:
    """Normalize vectors."""
    import numpy as np
    results = []
    for t in targets:
        if not t.is_vector:
            continue
        arr = t.to_numpy()
        norm = np.linalg.norm(arr)
        if norm > 1e-10:
            arr = arr / norm
        new_data = tuple(float(x) for x in arr)
        if create_new:
            new_t = TensorData(
                id=_generate_id(),
                data=new_data,
                shape=t.shape,
                dtype=t.dtype,
                label=f"{t.label}_norm",
                color=t.color,
                visible=True,
                history=t.history + ("normalize",)
            )
        else:
            new_t = replace(t, data=new_data, history=t.history + ("normalize",))
        results.append(new_t)
    return results


def _op_scale(targets: list, params: dict, create_new: bool) -> list:
    """Scale vectors/matrices."""
    import numpy as np
    factor = float(params.get("factor", 1.0))
    results = []
    for t in targets:
        arr = t.to_numpy() * factor
        new_data = _numpy_to_tuples(arr)
        if create_new:
            new_t = TensorData(
                id=_generate_id(),
                data=new_data,
                shape=t.shape,
                dtype=t.dtype,
                label=f"{t.label}_scaled",
                color=t.color,
                visible=True,
                history=t.history + (f"scale({factor})",)
            )
        else:
            new_t = replace(t, data=new_data, history=t.history + (f"scale({factor})",))
        results.append(new_t)
    return results


def _op_transpose(targets: list, params: dict, create_new: bool) -> list:
    """Transpose matrices."""
    import numpy as np
    results = []
    for t in targets:
        if not t.is_matrix:
            continue
        arr = t.to_numpy().T
        new_data = _numpy_to_tuples(arr)
        new_shape = (t.shape[1], t.shape[0])
        if create_new:
            new_t = TensorData(
                id=_generate_id(),
                data=new_data,
                shape=new_shape,
                dtype=t.dtype,
                label=f"{t.label}^T",
                color=t.color,
                visible=True,
                history=t.history + ("transpose",)
            )
        else:
            new_t = replace(t, data=new_data, shape=new_shape, history=t.history + ("transpose",))
        results.append(new_t)
    return results


def _op_inverse(targets: list, params: dict, create_new: bool) -> list:
    """Compute matrix inverse."""
    import numpy as np
    results = []
    for t in targets:
        if not t.is_matrix:
            continue
        try:
            arr = np.linalg.inv(t.to_numpy())
            new_data = _numpy_to_tuples(arr)
            if create_new:
                new_t = TensorData(
                    id=_generate_id(),
                    data=new_data,
                    shape=t.shape,
                    dtype=t.dtype,
                    label=f"{t.label}^(-1)",
                    color=t.color,
                    visible=True,
                    history=t.history + ("inverse",)
                )
            else:
                new_t = replace(t, data=new_data, history=t.history + ("inverse",))
            results.append(new_t)
        except np.linalg.LinAlgError:
            pass  # Matrix not invertible
    return results


def _op_apply_kernel(targets: list, params: dict, create_new: bool) -> list:
    """Apply convolution kernel to images."""
    kernel_name = params.get("kernel", "identity")
    results = []
    for t in targets:
        if not t.is_image:
            continue
        try:
            from domain.images.convolution.convolution import apply_kernel
            from domain.images.kernels import get_kernel_by_name
            kernel = get_kernel_by_name(kernel_name)
            pixels = t.to_numpy()
            result_pixels = apply_kernel(pixels, kernel)
            new_t = TensorData.create_image(
                pixels=result_pixels,
                name=f"{t.label}_{kernel_name}" if create_new else t.label,
                history=t.history + (f"kernel:{kernel_name}",)
            )
            if not create_new:
                new_t = replace(new_t, id=t.id)
            results.append(new_t)
        except Exception:
            pass
    return results


def _generate_id() -> str:
    """Generate a unique ID."""
    from uuid import uuid4
    return str(uuid4())


def _numpy_to_tuples(arr) -> tuple:
    """Convert numpy array to nested tuples."""
    import numpy as np
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return tuple(float(x) for x in arr)
    return tuple(_numpy_to_tuples(row) for row in arr)
