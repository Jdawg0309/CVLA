"""
Reducer for tensor actions.

Handles CRUD operations on the unified tensor model.
"""

from dataclasses import replace
from typing import Optional, Callable, TYPE_CHECKING
import numpy as np

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
from domain.operations.operation_registry import registry


def _is_rank1(t: TensorData) -> bool:
    return t.rank == 1


def _is_rank2(t: TensorData) -> bool:
    return t.rank == 2


def _is_image_dtype(t: TensorData) -> bool:
    return t.dtype in (TensorDType.IMAGE_RGB, TensorDType.IMAGE_GRAYSCALE)


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
    removed = next((t for t in state.tensors if t.id == action.id), None)
    new_tensors = tuple(t for t in state.tensors if t.id != action.id)
    new_selected = state.selected_tensor_id if state.selected_tensor_id != action.id else None

    # If an image tensor is deleted, also clear current/processed images so it disappears from view.
    if removed and _is_image_dtype(removed):
        new_state = replace(
            state,
            tensors=new_tensors,
            selected_tensor_id=new_selected,
            current_image=None,
            processed_image=None,
            current_image_preview=None,
            processed_image_preview=None,
            current_image_stats=None,
            processed_image_stats=None,
        )
        return with_history(new_state)

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

    # Normalize to numpy array if ImageMatrix
    if hasattr(pixels, "data"):
        try:
            label = action.label or getattr(pixels, "name", label)
        except Exception:
            pass
        pixels = pixels.data

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
    # Find target tensors (preserve action order)
    tensor_by_id = {t.id: t for t in state.tensors}
    targets = [tensor_by_id[tid] for tid in action.target_ids if tid in tensor_by_id]
    if not targets:
        return state

    if action.operation_name == "dot":
        op = registry.get("dot")
        if op is None:
            return state
        if len(targets) < 2:
            return state
        a, b = targets[0], targets[1]
        if not (_is_rank1(a) and _is_rank1(b)):
            return state
        try:
            result_value = op.compute({"vectors": (a, b)}, dict(action.parameters))
            steps = tuple(op.steps({"vectors": (a, b)}, dict(action.parameters), result_value))
        except Exception as e:
            return replace(state,
                error_message=str(e),
                show_error_modal=True,
            )

        new_data = (float(result_value),)
        if action.create_new:
            result_tensor = TensorData(
                id=_generate_id(),
                data=new_data,
                shape=(1,),
                dtype=a.dtype,
                label=f"{a.label}dot{b.label}",
                color=a.color,
                visible=True,
                history=a.history + ("dot",)
            )
        else:
            result_tensor = replace(a, data=new_data, shape=(1,), history=a.history + ("dot",))

        record = OperationRecord.create(
            operation_name=action.operation_name,
            parameters=action.parameters,
            target_ids=action.target_ids,
            result_ids=(result_tensor.id,) if action.create_new else (),
        )

        if action.create_new:
            new_tensors = state.tensors + (result_tensor,)
        else:
            new_tensors = tuple(result_tensor if t.id == result_tensor.id else t for t in state.tensors)

        new_state = replace(
            state,
            tensors=new_tensors,
            operation_history=state.operation_history + (record,),
            selected_tensor_id=result_tensor.id,
            operations=replace(
                state.operations,
                current_operation="dot",
                steps=steps,
                step_index=0,
            ),
        )
        return with_history(new_state)

    if action.operation_name == "matrix_multiply":
        if len(targets) >= 2 and _is_rank2(targets[0]) and _is_rank1(targets[1]):
            op = registry.get("matrix_vector_multiply")
            if op is None:
                return state
            matrix = targets[0]
            vector = targets[1]
            try:
                result_value = op.compute({"matrix": matrix, "vector": vector}, dict(action.parameters))
                steps = tuple(op.steps({"matrix": matrix, "vector": vector}, dict(action.parameters), result_value))
            except Exception as e:
                return replace(state,
                    error_message=str(e),
                    show_error_modal=True,
                )

            result_id = _generate_id()
            new_data = tuple(float(x) for x in result_value)

            record = OperationRecord.create(
                operation_name=action.operation_name,
                parameters=action.parameters,
                target_ids=action.target_ids,
                result_ids=(result_id,),
            )

            new_tensors = state.tensors
            selected_id = vector.id
            if len(steps) == 1:
                result_tensor = TensorData(
                    id=result_id,
                    data=new_data,
                    shape=(matrix.rows,),
                    dtype=TensorDType.NUMERIC,
                    label=f"{matrix.label}*{vector.label}",
                    color=vector.color,
                    visible=True,
                    history=vector.history + ("matrix_multiply",)
                )
                new_tensors = state.tensors + (result_tensor,)
                selected_id = result_id

            new_state = replace(
                state,
                tensors=new_tensors,
                operation_history=state.operation_history + (record,),
                selected_tensor_id=selected_id,
                operations=replace(
                    state.operations,
                    current_operation="matrix_vector_multiply",
                    steps=steps,
                    step_index=0,
                ),
            )
            return with_history(new_state)

    # Apply operation (placeholder - actual implementation depends on operation)
    try:
        result_tensors = _execute_operation(
            action.operation_name,
            action.parameters,
            targets,
            action.create_new
        )
    except OperationError as e:
        # Operation failed with a user-facing error - show error modal
        return replace(state,
            error_message=e.message,
            show_error_modal=True,
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

    # If any result is an image tensor, sync to processed_image for rendering.
    processed_image = state.processed_image
    current_image = state.current_image
    if any(_is_image_dtype(t) for t in result_tensors):
        first_image = next(t for t in result_tensors if _is_image_dtype(t))
        import numpy as np
        from state.models.image_model import ImageData
        img_np = np.array(first_image.data)
        processed_image = ImageData.create(img_np, first_image.label)
        current_image = current_image or processed_image

    selected_id = result_tensors[-1].id if result_tensors else state.selected_tensor_id

    new_state = replace(
        state,
        tensors=new_tensors,
        operation_history=state.operation_history + (record,),
        processed_image=processed_image,
        current_image=current_image,
        active_image_tab="processed" if processed_image is not None else state.active_image_tab,
        image_status=(
            f"Applied {action.operation_name}"
            if processed_image is not None else state.image_status
        ),
        image_status_level="info" if processed_image is not None else state.image_status_level,
        selected_tensor_id=selected_id,
        show_image_on_grid=True if processed_image is not None else state.show_image_on_grid,
        operations=replace(
            state.operations,
            current_operation=None,
            steps=(),
            step_index=0,
        ),
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
    has_image = any(_is_image_dtype(t) for t in targets)

    # Image-first handlers for shared names
    if operation_name == "normalize" and has_image:
        return _op_normalize_image(targets, params, create_new)

    # Vector operations
    if operation_name == "normalize":
        return _op_normalize(targets, params, create_new)
    if operation_name == "negate":
        return _op_negate(targets, params, create_new)
    if operation_name == "scale":
        return _op_scale(targets, params, create_new)
    if operation_name == "add":
        return _op_add(targets, params, create_new)
    if operation_name == "subtract":
        return _op_subtract(targets, params, create_new)
    if operation_name == "dot":
        return _op_dot(targets, params, create_new)
    if operation_name == "cross":
        return _op_cross(targets, params, create_new)
    if operation_name == "project":
        return _op_project(targets, params, create_new)
    if operation_name == "to_origin":
        return _op_to_origin(targets, params, create_new)

    # Matrix operations
    if operation_name == "transpose":
        return _op_transpose(targets, params, create_new)
    if operation_name == "inverse":
        return _op_inverse(targets, params, create_new)
    if operation_name == "determinant":
        return _op_determinant(targets, params, create_new)
    if operation_name == "trace":
        return _op_trace(targets, params, create_new)
    if operation_name == "matrix_multiply":
        return _op_matrix_multiply(targets, params, create_new)
    if operation_name == "eigen":
        return _op_eigen(targets, params, create_new)
    if operation_name == "svd":
        return _op_svd(targets, params, create_new)
    if operation_name == "qr":
        return _op_qr(targets, params, create_new)
    if operation_name == "lu":
        return _op_lu(targets, params, create_new)

    # Image operations
    if operation_name == "apply_kernel":
        return _op_apply_kernel(targets, params, create_new)
    if operation_name == "rotate":
        return _op_rotate_image(targets, params, create_new)
    if operation_name == "scale_image":
        return _op_scale_image(targets, params, create_new)
    if operation_name == "flip_horizontal":
        return _op_flip_image(targets, params, axis=1, create_new=create_new)
    if operation_name == "flip_vertical":
        return _op_flip_image(targets, params, axis=0, create_new=create_new)
    if operation_name == "normalize":
        return _op_normalize_image(targets, params, create_new)
    if operation_name == "to_grayscale":
        return _op_to_grayscale(targets, params, create_new)
    if operation_name == "invert":
        return _op_invert_image(targets, params, create_new)
    if operation_name == "to_matrix":
        return _op_image_to_matrix(targets, params, create_new)
    if operation_name == "reset_image":
        return _op_reset_image(targets, params, create_new)

    return None


def _op_normalize(targets: list, params: dict, create_new: bool) -> list:
    """Normalize vectors."""
    import numpy as np
    results = []
    for t in targets:
        if not _is_rank1(t):
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


def _op_negate(targets: list, params: dict, create_new: bool) -> list:
    """Negate vectors."""
    results = []
    for t in targets:
        if not _is_rank1(t):
            continue
        arr = -t.to_numpy()
        new_data = tuple(float(x) for x in arr)
        if create_new:
            new_t = TensorData(
                id=_generate_id(),
                data=new_data,
                shape=t.shape,
                dtype=t.dtype,
                label=f"-{t.label}",
                color=t.color,
                visible=True,
                history=t.history + ("negate",)
            )
        else:
            new_t = replace(t, data=new_data, history=t.history + ("negate",))
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


def _op_add(targets: list, params: dict, create_new: bool) -> list:
    """Add two vectors."""
    if len(targets) < 2:
        return []
    a, b = targets[0], targets[1]
    if not (_is_rank1(a) and _is_rank1(b)):
        return []
    if len(a.coords) != len(b.coords):
        return []
    arr = a.to_numpy() + b.to_numpy()
    new_data = tuple(float(x) for x in arr)
    if create_new:
        new_t = TensorData(
            id=_generate_id(),
            data=new_data,
            shape=a.shape,
            dtype=a.dtype,
            label=f"{a.label}+{b.label}",
            color=a.color,
            visible=True,
            history=a.history + ("add",)
        )
        return [new_t]
    new_t = replace(a, data=new_data, history=a.history + ("add",))
    return [new_t]


def _op_subtract(targets: list, params: dict, create_new: bool) -> list:
    """Subtract two vectors."""
    if len(targets) < 2:
        return []
    a, b = targets[0], targets[1]
    if not (_is_rank1(a) and _is_rank1(b)):
        return []
    if len(a.coords) != len(b.coords):
        return []
    arr = a.to_numpy() - b.to_numpy()
    new_data = tuple(float(x) for x in arr)
    if create_new:
        new_t = TensorData(
            id=_generate_id(),
            data=new_data,
            shape=a.shape,
            dtype=a.dtype,
            label=f"{a.label}-{b.label}",
            color=a.color,
            visible=True,
            history=a.history + ("subtract",)
        )
        return [new_t]
    new_t = replace(a, data=new_data, history=a.history + ("subtract",))
    return [new_t]


def _op_dot(targets: list, params: dict, create_new: bool) -> list:
    """Dot product of two vectors."""
    if len(targets) < 2:
        return []
    a, b = targets[0], targets[1]
    if not (_is_rank1(a) and _is_rank1(b)):
        return []
    if len(a.coords) != len(b.coords):
        return []
    value = float(a.to_numpy().dot(b.to_numpy()))
    new_data = (value,)
    if create_new:
        new_t = TensorData(
            id=_generate_id(),
            data=new_data,
            shape=(1,),
            dtype=a.dtype,
            label=f"{a.label}dot{b.label}",
            color=a.color,
            visible=True,
            history=a.history + ("dot",)
        )
        return [new_t]
    new_t = replace(a, data=new_data, shape=(1,), history=a.history + ("dot",))
    return [new_t]


def _op_cross(targets: list, params: dict, create_new: bool) -> list:
    """Cross product of two 3D vectors."""
    if len(targets) < 2:
        return []
    a, b = targets[0], targets[1]
    if not (_is_rank1(a) and _is_rank1(b)):
        return []
    if len(a.coords) != 3 or len(b.coords) != 3:
        return []
    arr = np.cross(a.to_numpy(), b.to_numpy())
    new_data = tuple(float(x) for x in arr)
    if create_new:
        new_t = TensorData(
            id=_generate_id(),
            data=new_data,
            shape=(3,),
            dtype=a.dtype,
            label=f"{a.label}x{b.label}",
            color=a.color,
            visible=True,
            history=a.history + ("cross",)
        )
        return [new_t]
    new_t = replace(a, data=new_data, shape=(3,), history=a.history + ("cross",))
    return [new_t]


def _op_project(targets: list, params: dict, create_new: bool) -> list:
    """Project one vector onto another."""
    if len(targets) < 2:
        return []
    v, onto = targets[0], targets[1]
    if not (_is_rank1(v) and _is_rank1(onto)):
        return []
    if len(v.coords) != len(onto.coords):
        return []
    onto_arr = onto.to_numpy()
    denom = float(np.dot(onto_arr, onto_arr))
    if denom < 1e-10:
        return []
    scalar = float(np.dot(v.to_numpy(), onto_arr)) / denom
    arr = onto_arr * scalar
    new_data = tuple(float(x) for x in arr)
    if create_new:
        new_t = TensorData(
            id=_generate_id(),
            data=new_data,
            shape=v.shape,
            dtype=v.dtype,
            label=f"proj_{onto.label}({v.label})",
            color=v.color,
            visible=True,
            history=v.history + ("project",)
        )
        return [new_t]
    new_t = replace(v, data=new_data, history=v.history + ("project",))
    return [new_t]


def _op_to_origin(targets: list, params: dict, create_new: bool) -> list:
    """Move vector tail to origin (0,0,0).

    Creates a fresh copy of the vector positioned from the origin.
    In the current model, vectors are already from origin, so this
    serves as a confirmation/reset operation.
    """
    results = []
    for t in targets:
        if not _is_rank1(t):
            continue
        # Vector data already represents position from origin
        # This operation confirms/resets the vector to origin-based
        new_data = t.data  # Keep same coordinates (direction from origin)
        if create_new:
            new_t = TensorData(
                id=_generate_id(),
                data=new_data,
                shape=t.shape,
                dtype=t.dtype,
                label=f"{t.label}_origin",
                color=t.color,
                visible=True,
                history=t.history + ("to_origin",)
            )
        else:
            new_t = replace(t, history=t.history + ("to_origin",))
        results.append(new_t)
    return results


def _op_transpose(targets: list, params: dict, create_new: bool) -> list:
    """Transpose matrices."""
    import numpy as np
    results = []
    for t in targets:
        if not _is_rank2(t):
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


class OperationError(Exception):
    """Exception raised when an operation fails with a user-facing error."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def _op_inverse(targets: list, params: dict, create_new: bool) -> list:
    """Compute matrix inverse."""
    import numpy as np
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        # Check if matrix is square
        if t.rows != t.cols:
            raise OperationError(f"Matrix '{t.label}' is not square ({t.rows}x{t.cols}). Only square matrices can be inverted.")
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
            raise OperationError(f"Matrix '{t.label}' is singular (determinant = 0) and cannot be inverted.")
    return results


def _op_determinant(targets: list, params: dict, create_new: bool) -> list:
    """Compute matrix determinant."""
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        if t.rows != t.cols:
            continue
        value = float(np.linalg.det(t.to_numpy()))
        new_data = ((value,),)
        if create_new:
            new_t = TensorData(
                id=_generate_id(),
                data=new_data,
                shape=(1, 1),
                dtype=t.dtype,
                label=f"det({t.label})",
                color=t.color,
                visible=True,
                history=t.history + ("determinant",)
            )
        else:
            new_t = replace(t, data=new_data, shape=(1, 1), history=t.history + ("determinant",))
        results.append(new_t)
    return results


def _op_trace(targets: list, params: dict, create_new: bool) -> list:
    """Compute matrix trace."""
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        if t.rows != t.cols:
            continue
        value = float(np.trace(t.to_numpy()))
        new_data = ((value,),)
        if create_new:
            new_t = TensorData(
                id=_generate_id(),
                data=new_data,
                shape=(1, 1),
                dtype=t.dtype,
                label=f"tr({t.label})",
                color=t.color,
                visible=True,
                history=t.history + ("trace",)
            )
        else:
            new_t = replace(t, data=new_data, shape=(1, 1), history=t.history + ("trace",))
        results.append(new_t)
    return results


def _op_matrix_multiply(targets: list, params: dict, create_new: bool) -> list:
    """Matrix multiplication (matrix-matrix or matrix-vector)."""
    if len(targets) < 2:
        return []
    a, b = targets[0], targets[1]
    if _is_rank2(a) and _is_rank1(b):
        if a.cols != len(b.coords):
            return []
        result = a.to_numpy() @ b.to_numpy()
        new_data = tuple(float(x) for x in result)
        if create_new:
            new_t = TensorData(
                id=_generate_id(),
                data=new_data,
                shape=(a.rows,),
                dtype=TensorDType.NUMERIC,
                label=f"{a.label}*{b.label}",
                color=b.color,
                visible=True,
                history=b.history + ("matrix_multiply",)
            )
        else:
            new_t = replace(b, data=new_data, shape=(a.rows,), history=b.history + ("matrix_multiply",))
        return [new_t]

    if _is_rank2(a) and _is_rank2(b):
        if a.cols != b.rows:
            return []
        result = a.to_numpy() @ b.to_numpy()
        new_data = _numpy_to_tuples(result)
        new_shape = (a.rows, b.cols)
        if create_new:
            new_t = TensorData(
                id=_generate_id(),
                data=new_data,
                shape=new_shape,
                dtype=a.dtype,
                label=f"{a.label}*{b.label}",
                color=a.color,
                visible=True,
                history=a.history + ("matrix_multiply",)
            )
        else:
            new_t = replace(a, data=new_data, shape=new_shape, history=a.history + ("matrix_multiply",))
        return [new_t]

    return []


def _op_eigen(targets: list, params: dict, create_new: bool) -> list:
    """Eigendecomposition: returns eigenvalues as vector and eigenvectors as matrix."""
    results = []
    for t in targets:
        if not _is_rank2(t) or t.rows != t.cols:
            continue
        try:
            vals, vecs = np.linalg.eig(t.to_numpy())
        except np.linalg.LinAlgError:
            continue
        eigvals = tuple(float(v) for v in vals)
        eigvecs = _numpy_to_tuples(vecs)
        val_tensor = TensorData(
            id=_generate_id() if create_new else t.id,
            data=eigvals,
            shape=(len(eigvals),),
            dtype=t.dtype,
            label=f"eigvals({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("eigen",)
        )
        vec_tensor = TensorData(
            id=_generate_id(),
            data=eigvecs,
            shape=vecs.shape,
            dtype=t.dtype,
            label=f"eigvecs({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("eigen",)
        )
        results.extend([val_tensor, vec_tensor])
    return results


def _op_svd(targets: list, params: dict, create_new: bool) -> list:
    """Singular Value Decomposition."""
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        try:
            U, S, Vt = np.linalg.svd(t.to_numpy(), full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        U_t = TensorData(
            id=_generate_id(),
            data=_numpy_to_tuples(U),
            shape=U.shape,
            dtype=t.dtype,
            label=f"U({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("svd",)
        )
        S_t = TensorData(
            id=_generate_id(),
            data=tuple(float(s) for s in S),
            shape=(len(S),),
            dtype=t.dtype,
            label=f"S({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("svd",)
        )
        V_t = TensorData(
            id=_generate_id(),
            data=_numpy_to_tuples(Vt),
            shape=Vt.shape,
            dtype=t.dtype,
            label=f"Vt({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("svd",)
        )
        results.extend([U_t, S_t, V_t])
    return results


def _op_qr(targets: list, params: dict, create_new: bool) -> list:
    """QR decomposition."""
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        try:
            Q, R = np.linalg.qr(t.to_numpy())
        except np.linalg.LinAlgError:
            continue
        Q_t = TensorData(
            id=_generate_id(),
            data=_numpy_to_tuples(Q),
            shape=Q.shape,
            dtype=t.dtype,
            label=f"Q({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("qr",)
        )
        R_t = TensorData(
            id=_generate_id(),
            data=_numpy_to_tuples(R),
            shape=R.shape,
            dtype=t.dtype,
            label=f"R({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("qr",)
        )
        results.extend([Q_t, R_t])
    return results


def _op_lu(targets: list, params: dict, create_new: bool) -> list:
    """LU decomposition (simple Doolittle via numpy)."""
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        A = t.to_numpy()
        n, m = A.shape
        if n != m:
            continue
        try:
            L = np.zeros_like(A, dtype=float)
            U = np.zeros_like(A, dtype=float)
            for i in range(n):
                for k in range(i, n):
                    U[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(i))
                for k in range(i, n):
                    if i == k:
                        L[i, i] = 1.0
                    else:
                        if abs(U[i, i]) < 1e-12:
                            raise np.linalg.LinAlgError
                        L[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(i))) / U[i, i]
        except np.linalg.LinAlgError:
            continue
        L_t = TensorData(
            id=_generate_id(),
            data=_numpy_to_tuples(L),
            shape=L.shape,
            dtype=t.dtype,
            label=f"L({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("lu",)
        )
        U_t = TensorData(
            id=_generate_id(),
            data=_numpy_to_tuples(U),
            shape=U.shape,
            dtype=t.dtype,
            label=f"U({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("lu",)
        )
        results.extend([L_t, U_t])
    return results


def _op_apply_kernel(targets: list, params: dict, create_new: bool) -> list:
    """Apply convolution kernel to images."""
    kernel_name = params.get("kernel", "identity")
    results = []
    for t in targets:
        if not _is_image_dtype(t):
            continue
        from domain.images.convolution.convolution import apply_kernel
        from domain.images.image_matrix import ImageMatrix

        img_mat = ImageMatrix(t.to_numpy(), name=t.label)
        result_img = apply_kernel(img_mat, kernel_name)
        result_pixels = result_img.data

        new_t = TensorData.create_image(
            pixels=result_pixels,
            name=f"{t.label}_{kernel_name}" if create_new else t.label,
            history=t.history + (f"kernel:{kernel_name}",),
            preserve_original=False  # Don't overwrite original
        )
        if not create_new:
            # Preserve the original_data from source tensor for reset
            new_t = replace(new_t, id=t.id, original_data=t.original_data)
        results.append(new_t)
    return results


def _op_rotate_image(targets: list, params: dict, create_new: bool) -> list:
    """Rotate image by nearest 90-degree step for simplicity."""
    angle = float(params.get("angle", 0.0))
    k = int(round(angle / 90.0)) % 4  # quarter turns
    results = []
    for t in targets:
        if not _is_image_dtype(t):
            continue
        arr = t.to_numpy()
        rotated = np.rot90(arr, k=k, axes=(0, 1))
        new_t = TensorData.create_image(
            rotated, f"{t.label}_rot{angle:.0f}",
            history=t.history + (f"rotate:{angle}",),
            preserve_original=False
        )
        if not create_new:
            new_t = replace(new_t, id=t.id, original_data=t.original_data)
        results.append(new_t)
    return results


def _op_scale_image(targets: list, params: dict, create_new: bool) -> list:
    """Nearest-neighbor scale for images."""
    factor = max(0.01, float(params.get("factor", 1.0)))
    results = []
    for t in targets:
        if not _is_image_dtype(t):
            continue
        arr = t.to_numpy()
        new_h = max(1, int(round(arr.shape[0] * factor)))
        new_w = max(1, int(round(arr.shape[1] * factor)))
        y_idx = (np.linspace(0, arr.shape[0] - 1, new_h)).astype(int)
        x_idx = (np.linspace(0, arr.shape[1] - 1, new_w)).astype(int)
        scaled = arr[np.ix_(y_idx, x_idx)]
        new_t = TensorData.create_image(
            scaled, f"{t.label}_s{factor:.2f}",
            history=t.history + (f"scale:{factor}",),
            preserve_original=False
        )
        if not create_new:
            new_t = replace(new_t, id=t.id, original_data=t.original_data)
        results.append(new_t)
    return results


def _op_flip_image(targets: list, params: dict, axis: int, create_new: bool) -> list:
    """Flip image horizontally or vertically."""
    results = []
    flip_type = "horizontal" if axis == 1 else "vertical"
    for t in targets:
        if not _is_image_dtype(t):
            continue
        arr = np.flip(t.to_numpy(), axis=axis)
        new_t = TensorData.create_image(
            arr, f"{t.label}_flip",
            history=t.history + (f"flip:{flip_type}",),
            preserve_original=False
        )
        if not create_new:
            new_t = replace(new_t, id=t.id, original_data=t.original_data)
        results.append(new_t)
    return results


def _op_normalize_image(targets: list, params: dict, create_new: bool) -> list:
    """Normalize image by mean/std."""
    mean = float(params.get("mean", 0.0))
    std = float(params.get("std", 1.0))
    if abs(std) < 1e-12:
        std = 1.0
    results = []
    for t in targets:
        if not _is_image_dtype(t):
            continue
        arr = t.to_numpy().astype(float)
        normed = (arr - mean) / std
        # clamp to 0..1 for display
        normed = np.clip(normed, 0.0, 1.0)
        new_t = TensorData.create_image(
            normed, f"{t.label}_norm",
            history=t.history + (f"normalize:{mean},{std}",),
            preserve_original=False
        )
        if not create_new:
            new_t = replace(new_t, id=t.id, original_data=t.original_data)
        results.append(new_t)
    return results


def _op_to_grayscale(targets: list, params: dict, create_new: bool) -> list:
    """Convert RGB image to grayscale."""
    results = []
    for t in targets:
        if not _is_image_dtype(t):
            continue
        arr = t.to_numpy().astype(float)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
        else:
            gray = arr
        gray = np.clip(gray, 0.0, 1.0)
        new_t = TensorData.create_image(
            gray, f"{t.label}_gray",
            history=t.history + ("to_grayscale",),
            preserve_original=False
        )
        if not create_new:
            new_t = replace(new_t, id=t.id, original_data=t.original_data)
        results.append(new_t)
    return results


def _op_invert_image(targets: list, params: dict, create_new: bool) -> list:
    """Invert image colors."""
    results = []
    for t in targets:
        if not _is_image_dtype(t):
            continue
        arr = t.to_numpy().astype(float)
        inv = 1.0 - arr
        inv = np.clip(inv, 0.0, 1.0)
        new_t = TensorData.create_image(
            inv, f"{t.label}_inv",
            history=t.history + ("invert",),
            preserve_original=False
        )
        if not create_new:
            new_t = replace(new_t, id=t.id, original_data=t.original_data)
        results.append(new_t)
    return results


def _op_image_to_matrix(targets: list, params: dict, create_new: bool) -> list:
    """Convert image to numeric matrix tensor (grayscale)."""
    results = []
    for t in targets:
        if not _is_image_dtype(t):
            continue
        arr = t.to_numpy()
        if arr.ndim == 3 and arr.shape[2] >= 3:
            gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
        else:
            gray = arr
        data = _numpy_to_tuples(gray)
        mat_tensor = TensorData(
            id=_generate_id() if create_new else t.id,
            data=data,
            shape=gray.shape,
            dtype=TensorDType.NUMERIC,
            label=f"{t.label}_matrix",
            color=t.color,
            visible=True,
            history=t.history + ("to_matrix",)
        )
        results.append(mat_tensor)
    return results


def _op_reset_image(targets: list, params: dict, create_new: bool) -> list:
    """Reset image to its original state."""
    results = []
    for t in targets:
        if not _is_image_dtype(t):
            continue
        # Check if we have original data stored
        if t.original_data is not None:
            # Restore to original
            new_t = replace(
                t,
                data=t.original_data,
                history=()  # Clear history since we're resetting
            )
        else:
            # No original data stored, just clear history
            new_t = replace(t, history=())
        results.append(new_t)
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
