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
    SetBinaryOperation, ClearBinaryOperation,
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

    if isinstance(action, SetBinaryOperation):
        return replace(state,
            awaiting_second_tensor=action.operation_name,
            first_tensor_id=action.first_tensor_id,
        )

    if isinstance(action, ClearBinaryOperation):
        return replace(state,
            awaiting_second_tensor=None,
            first_tensor_id=None,
        )

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
            selection_id = vector.id
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
                selection_id = result_id

            new_state = replace(
                state,
                tensors=new_tensors,
                operation_history=state.operation_history + (record,),
                selected_tensor_id=selection_id,
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

    selection_id = result_tensors[-1].id if result_tensors else state.selected_tensor_id

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
        selected_tensor_id=selection_id,
        show_image_on_grid=True if processed_image is not None else state.show_image_on_grid,
        # Clear binary operation state after operation completes
        awaiting_second_tensor=None,
        first_tensor_id=None,
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
    if operation_name == "reject":
        return _op_reject(targets, params, create_new)
    if operation_name == "angle_between":
        return _op_angle_between(targets, params, create_new)
    if operation_name == "to_origin":
        return _op_to_origin(targets, params, create_new)
    if operation_name == "hadamard":
        return _op_hadamard(targets, params, create_new)
    if operation_name == "outer_product":
        return _op_outer_product(targets, params, create_new)
    if operation_name == "kronecker":
        return _op_kronecker(targets, params, create_new)

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
    if operation_name == "cholesky":
        return _op_cholesky(targets, params, create_new)
    if operation_name == "schur":
        return _op_schur(targets, params, create_new)

    # Matrix properties
    if operation_name == "rank":
        return _op_rank(targets, params, create_new)
    if operation_name == "condition_number":
        return _op_condition_number(targets, params, create_new)
    if operation_name == "nullity":
        return _op_nullity(targets, params, create_new)

    # Transformations
    if operation_name == "pseudoinverse":
        return _op_pseudoinverse(targets, params, create_new)
    if operation_name == "adjoint":
        return _op_adjoint(targets, params, create_new)
    if operation_name == "cofactor":
        return _op_cofactor(targets, params, create_new)
    if operation_name == "adjugate":
        return _op_adjugate(targets, params, create_new)

    # Eigenvalues
    if operation_name == "eigenvalues":
        return _op_eigenvalues(targets, params, create_new)
    if operation_name == "eigenvectors":
        return _op_eigenvectors(targets, params, create_new)
    if operation_name == "spectral_radius":
        return _op_spectral_radius(targets, params, create_new)
    if operation_name == "power_iteration":
        return _op_power_iteration(targets, params, create_new)

    # Change of basis
    if operation_name == "change_basis":
        return _op_change_basis(targets, params, create_new)
    if operation_name == "similarity_transform":
        return _op_similarity_transform(targets, params, create_new)
    if operation_name == "orthogonalize":
        return _op_orthogonalize(targets, params, create_new)
    if operation_name == "project_subspace":
        return _op_project_subspace(targets, params, create_new)

    # Norms
    if operation_name == "frobenius_norm":
        return _op_frobenius_norm(targets, params, create_new)
    if operation_name == "l1_norm":
        return _op_l1_norm(targets, params, create_new)
    if operation_name == "l2_norm":
        return _op_l2_norm(targets, params, create_new)
    if operation_name == "inf_norm":
        return _op_inf_norm(targets, params, create_new)
    if operation_name == "nuclear_norm":
        return _op_nuclear_norm(targets, params, create_new)

    # Linear systems
    if operation_name == "gaussian_elimination":
        return _op_gaussian_elimination(targets, params, create_new)
    if operation_name == "rref":
        return _op_rref(targets, params, create_new)
    if operation_name == "back_substitution":
        return _op_back_substitution(targets, params, create_new)
    if operation_name == "solve_linear":
        return _op_solve_linear(targets, params, create_new)
    if operation_name == "least_squares":
        return _op_least_squares(targets, params, create_new)

    # Special matrices
    if operation_name == "symmetrize":
        return _op_symmetrize(targets, params, create_new)
    if operation_name == "skew_symmetrize":
        return _op_skew_symmetrize(targets, params, create_new)
    if operation_name == "diagonalize":
        return _op_diagonalize(targets, params, create_new)
    if operation_name == "triangular_upper":
        return _op_triangular_upper(targets, params, create_new)
    if operation_name == "triangular_lower":
        return _op_triangular_lower(targets, params, create_new)

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
    """Transpose matrices. Vectors are unchanged (transpose doesn't apply to 1D)."""
    import numpy as np
    results = []
    for t in targets:
        if _is_rank1(t):
            # Vectors don't transpose - return unchanged
            if create_new:
                new_t = TensorData(
                    id=_generate_id(),
                    data=t.data,
                    shape=t.shape,
                    dtype=t.dtype,
                    label=f"{t.label}",
                    color=t.color,
                    visible=True,
                    history=t.history
                )
            else:
                new_t = t
            results.append(new_t)
        elif _is_rank2(t):
            arr = t.to_numpy().T
            new_data = _numpy_to_tuples(arr)
            new_shape = (t.shape[1], t.shape[0])
            if create_new:
                new_t = TensorData(
                    id=_generate_id(),
                    data=new_data,
                    shape=new_shape,
                    dtype=t.dtype,
                    label=f"{t.label}ᵀ",
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


# =============================================================================
# ADDITIONAL VECTOR OPERATIONS
# =============================================================================

def _op_reject(targets: list, params: dict, create_new: bool) -> list:
    """Vector rejection: a - proj_b(a)."""
    if len(targets) < 2:
        return []
    a, b = targets[0], targets[1]
    if not (_is_rank1(a) and _is_rank1(b)):
        return []
    arr_a = a.to_numpy()
    arr_b = b.to_numpy()
    dot_ab = np.dot(arr_a, arr_b)
    dot_bb = np.dot(arr_b, arr_b)
    if abs(dot_bb) < 1e-10:
        return []
    proj = (dot_ab / dot_bb) * arr_b
    rej = arr_a - proj
    new_data = tuple(float(x) for x in rej)
    new_t = TensorData(
        id=_generate_id() if create_new else a.id,
        data=new_data,
        shape=(len(new_data),),
        dtype=a.dtype,
        label=f"rej_{b.label}({a.label})",
        color=a.color,
        visible=True,
        history=a.history + ("reject",)
    )
    return [new_t]


def _op_angle_between(targets: list, params: dict, create_new: bool) -> list:
    """Angle between two vectors in radians."""
    if len(targets) < 2:
        return []
    a, b = targets[0], targets[1]
    if not (_is_rank1(a) and _is_rank1(b)):
        return []
    arr_a = a.to_numpy()
    arr_b = b.to_numpy()
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return []
    cos_theta = np.clip(np.dot(arr_a, arr_b) / (norm_a * norm_b), -1.0, 1.0)
    angle = np.arccos(cos_theta)
    new_t = TensorData(
        id=_generate_id(),
        data=(float(angle),),
        shape=(1,),
        dtype=TensorDType.NUMERIC,
        label=f"angle({a.label},{b.label})",
        color=a.color,
        visible=True,
        history=a.history + ("angle_between",)
    )
    return [new_t]


def _op_hadamard(targets: list, params: dict, create_new: bool) -> list:
    """Element-wise multiplication (Hadamard product)."""
    if len(targets) < 2:
        return []
    a, b = targets[0], targets[1]
    if a.shape != b.shape:
        return []
    arr_a = a.to_numpy()
    arr_b = b.to_numpy()
    result = arr_a * arr_b
    new_data = _numpy_to_tuples(result)
    new_t = TensorData(
        id=_generate_id() if create_new else a.id,
        data=new_data,
        shape=a.shape,
        dtype=a.dtype,
        label=f"{a.label}⊙{b.label}",
        color=a.color,
        visible=True,
        history=a.history + ("hadamard",)
    )
    return [new_t]


def _op_outer_product(targets: list, params: dict, create_new: bool) -> list:
    """Outer product of two vectors."""
    if len(targets) < 2:
        return []
    a, b = targets[0], targets[1]
    if not (_is_rank1(a) and _is_rank1(b)):
        return []
    arr_a = a.to_numpy()
    arr_b = b.to_numpy()
    result = np.outer(arr_a, arr_b)
    new_data = _numpy_to_tuples(result)
    new_t = TensorData(
        id=_generate_id() if create_new else a.id,
        data=new_data,
        shape=result.shape,
        dtype=a.dtype,
        label=f"{a.label}⊗{b.label}",
        color=a.color,
        visible=True,
        history=a.history + ("outer_product",)
    )
    return [new_t]


def _op_kronecker(targets: list, params: dict, create_new: bool) -> list:
    """Kronecker product of two matrices."""
    if len(targets) < 2:
        return []
    a, b = targets[0], targets[1]
    if not (_is_rank2(a) and _is_rank2(b)):
        return []
    arr_a = a.to_numpy()
    arr_b = b.to_numpy()
    result = np.kron(arr_a, arr_b)
    new_data = _numpy_to_tuples(result)
    new_t = TensorData(
        id=_generate_id() if create_new else a.id,
        data=new_data,
        shape=result.shape,
        dtype=a.dtype,
        label=f"{a.label}⊗{b.label}",
        color=a.color,
        visible=True,
        history=a.history + ("kronecker",)
    )
    return [new_t]


# =============================================================================
# MATRIX PROPERTIES
# =============================================================================

def _op_rank(targets: list, params: dict, create_new: bool) -> list:
    """Compute matrix rank."""
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        r = np.linalg.matrix_rank(t.to_numpy())
        new_t = TensorData(
            id=_generate_id(),
            data=(float(r),),
            shape=(1,),
            dtype=TensorDType.NUMERIC,
            label=f"rank({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("rank",)
        )
        results.append(new_t)
    return results


def _op_condition_number(targets: list, params: dict, create_new: bool) -> list:
    """Compute condition number."""
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        try:
            cond = np.linalg.cond(t.to_numpy())
        except np.linalg.LinAlgError:
            continue
        new_t = TensorData(
            id=_generate_id(),
            data=(float(cond),),
            shape=(1,),
            dtype=TensorDType.NUMERIC,
            label=f"cond({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("condition_number",)
        )
        results.append(new_t)
    return results


def _op_nullity(targets: list, params: dict, create_new: bool) -> list:
    """Compute nullity (dimension of null space)."""
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        arr = t.to_numpy()
        r = np.linalg.matrix_rank(arr)
        nullity = arr.shape[1] - r  # nullity = n - rank
        new_t = TensorData(
            id=_generate_id(),
            data=(float(nullity),),
            shape=(1,),
            dtype=TensorDType.NUMERIC,
            label=f"nullity({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("nullity",)
        )
        results.append(new_t)
    return results


# =============================================================================
# TRANSFORMATIONS
# =============================================================================

def _op_pseudoinverse(targets: list, params: dict, create_new: bool) -> list:
    """Moore-Penrose pseudoinverse."""
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        try:
            pinv = np.linalg.pinv(t.to_numpy())
        except np.linalg.LinAlgError:
            continue
        new_data = _numpy_to_tuples(pinv)
        new_t = TensorData(
            id=_generate_id() if create_new else t.id,
            data=new_data,
            shape=pinv.shape,
            dtype=t.dtype,
            label=f"{t.label}⁺",
            color=t.color,
            visible=True,
            history=t.history + ("pseudoinverse",)
        )
        results.append(new_t)
    return results


def _op_adjoint(targets: list, params: dict, create_new: bool) -> list:
    """Conjugate transpose (for real matrices, same as transpose)."""
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        adj = np.conj(t.to_numpy().T)
        new_data = _numpy_to_tuples(adj)
        new_t = TensorData(
            id=_generate_id() if create_new else t.id,
            data=new_data,
            shape=adj.shape,
            dtype=t.dtype,
            label=f"{t.label}*",
            color=t.color,
            visible=True,
            history=t.history + ("adjoint",)
        )
        results.append(new_t)
    return results


def _op_cofactor(targets: list, params: dict, create_new: bool) -> list:
    """Cofactor matrix."""
    results = []
    for t in targets:
        if not _is_rank2(t) or t.rows != t.cols:
            continue
        arr = t.to_numpy()
        n = arr.shape[0]
        cofactors = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                minor = np.delete(np.delete(arr, i, axis=0), j, axis=1)
                cofactors[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
        new_data = _numpy_to_tuples(cofactors)
        new_t = TensorData(
            id=_generate_id() if create_new else t.id,
            data=new_data,
            shape=cofactors.shape,
            dtype=t.dtype,
            label=f"cof({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("cofactor",)
        )
        results.append(new_t)
    return results


def _op_adjugate(targets: list, params: dict, create_new: bool) -> list:
    """Adjugate matrix (transpose of cofactor matrix)."""
    results = []
    for t in targets:
        if not _is_rank2(t) or t.rows != t.cols:
            continue
        arr = t.to_numpy()
        n = arr.shape[0]
        cofactors = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                minor = np.delete(np.delete(arr, i, axis=0), j, axis=1)
                cofactors[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
        adj = cofactors.T
        new_data = _numpy_to_tuples(adj)
        new_t = TensorData(
            id=_generate_id() if create_new else t.id,
            data=new_data,
            shape=adj.shape,
            dtype=t.dtype,
            label=f"adj({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("adjugate",)
        )
        results.append(new_t)
    return results


# =============================================================================
# DECOMPOSITIONS
# =============================================================================

def _op_cholesky(targets: list, params: dict, create_new: bool) -> list:
    """Cholesky decomposition (for positive definite matrices)."""
    results = []
    for t in targets:
        if not _is_rank2(t) or t.rows != t.cols:
            continue
        try:
            L = np.linalg.cholesky(t.to_numpy())
        except np.linalg.LinAlgError:
            continue
        new_data = _numpy_to_tuples(L)
        new_t = TensorData(
            id=_generate_id() if create_new else t.id,
            data=new_data,
            shape=L.shape,
            dtype=t.dtype,
            label=f"chol({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("cholesky",)
        )
        results.append(new_t)
    return results


def _op_schur(targets: list, params: dict, create_new: bool) -> list:
    """Schur decomposition."""
    from scipy import linalg as sp_linalg
    results = []
    for t in targets:
        if not _is_rank2(t) or t.rows != t.cols:
            continue
        try:
            T, Z = sp_linalg.schur(t.to_numpy())
        except Exception:
            continue
        T_tensor = TensorData(
            id=_generate_id(),
            data=_numpy_to_tuples(T),
            shape=T.shape,
            dtype=t.dtype,
            label=f"T({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("schur",)
        )
        Z_tensor = TensorData(
            id=_generate_id(),
            data=_numpy_to_tuples(Z),
            shape=Z.shape,
            dtype=t.dtype,
            label=f"Z({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("schur",)
        )
        results.extend([T_tensor, Z_tensor])
    return results


# =============================================================================
# EIGENVALUE OPERATIONS
# =============================================================================

def _op_eigenvalues(targets: list, params: dict, create_new: bool) -> list:
    """Compute eigenvalues only."""
    results = []
    for t in targets:
        if not _is_rank2(t) or t.rows != t.cols:
            continue
        try:
            vals = np.linalg.eigvals(t.to_numpy())
        except np.linalg.LinAlgError:
            continue
        # Return real parts for real matrices
        eigvals = tuple(float(np.real(v)) for v in vals)
        new_t = TensorData(
            id=_generate_id(),
            data=eigvals,
            shape=(len(eigvals),),
            dtype=t.dtype,
            label=f"λ({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("eigenvalues",)
        )
        results.append(new_t)
    return results


def _op_eigenvectors(targets: list, params: dict, create_new: bool) -> list:
    """Compute eigenvectors only."""
    results = []
    for t in targets:
        if not _is_rank2(t) or t.rows != t.cols:
            continue
        try:
            _, vecs = np.linalg.eig(t.to_numpy())
        except np.linalg.LinAlgError:
            continue
        eigvecs = _numpy_to_tuples(np.real(vecs))
        new_t = TensorData(
            id=_generate_id(),
            data=eigvecs,
            shape=vecs.shape,
            dtype=t.dtype,
            label=f"v({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("eigenvectors",)
        )
        results.append(new_t)
    return results


def _op_spectral_radius(targets: list, params: dict, create_new: bool) -> list:
    """Spectral radius (largest eigenvalue magnitude)."""
    results = []
    for t in targets:
        if not _is_rank2(t) or t.rows != t.cols:
            continue
        try:
            vals = np.linalg.eigvals(t.to_numpy())
        except np.linalg.LinAlgError:
            continue
        rho = float(np.max(np.abs(vals)))
        new_t = TensorData(
            id=_generate_id(),
            data=(rho,),
            shape=(1,),
            dtype=TensorDType.NUMERIC,
            label=f"ρ({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("spectral_radius",)
        )
        results.append(new_t)
    return results


def _op_power_iteration(targets: list, params: dict, create_new: bool) -> list:
    """Power iteration for dominant eigenvector."""
    max_iter = int(params.get("max_iter", 100))
    results = []
    for t in targets:
        if not _is_rank2(t) or t.rows != t.cols:
            continue
        arr = t.to_numpy()
        n = arr.shape[0]
        v = np.ones(n) / np.sqrt(n)
        for _ in range(max_iter):
            v_new = arr @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-10:
                break
            v = v_new / norm
        new_data = tuple(float(x) for x in v)
        new_t = TensorData(
            id=_generate_id(),
            data=new_data,
            shape=(n,),
            dtype=t.dtype,
            label=f"dom_eigvec({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("power_iteration",)
        )
        results.append(new_t)
    return results


# =============================================================================
# CHANGE OF BASIS
# =============================================================================

def _op_change_basis(targets: list, params: dict, create_new: bool) -> list:
    """Change of basis: P⁻¹AP."""
    if len(targets) < 2:
        return []
    A, P = targets[0], targets[1]
    if not (_is_rank2(A) and _is_rank2(P)):
        return []
    if A.rows != A.cols or P.rows != P.cols or A.rows != P.rows:
        return []
    try:
        P_inv = np.linalg.inv(P.to_numpy())
        result = P_inv @ A.to_numpy() @ P.to_numpy()
    except np.linalg.LinAlgError:
        return []
    new_data = _numpy_to_tuples(result)
    new_t = TensorData(
        id=_generate_id() if create_new else A.id,
        data=new_data,
        shape=result.shape,
        dtype=A.dtype,
        label=f"{P.label}⁻¹{A.label}{P.label}",
        color=A.color,
        visible=True,
        history=A.history + ("change_basis",)
    )
    return [new_t]


def _op_similarity_transform(targets: list, params: dict, create_new: bool) -> list:
    """Similarity transformation (same as change_basis)."""
    return _op_change_basis(targets, params, create_new)


def _op_orthogonalize(targets: list, params: dict, create_new: bool) -> list:
    """Gram-Schmidt orthogonalization."""
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        arr = t.to_numpy()
        Q, _ = np.linalg.qr(arr)
        new_data = _numpy_to_tuples(Q)
        new_t = TensorData(
            id=_generate_id() if create_new else t.id,
            data=new_data,
            shape=Q.shape,
            dtype=t.dtype,
            label=f"orth({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("orthogonalize",)
        )
        results.append(new_t)
    return results


def _op_project_subspace(targets: list, params: dict, create_new: bool) -> list:
    """Project onto column space of second matrix."""
    if len(targets) < 2:
        return []
    v, A = targets[0], targets[1]
    if not (_is_rank1(v) and _is_rank2(A)):
        return []
    arr_v = v.to_numpy()
    arr_A = A.to_numpy()
    # Projection matrix: A(A^T A)^{-1} A^T
    try:
        proj_matrix = arr_A @ np.linalg.inv(arr_A.T @ arr_A) @ arr_A.T
        result = proj_matrix @ arr_v
    except np.linalg.LinAlgError:
        return []
    new_data = tuple(float(x) for x in result)
    new_t = TensorData(
        id=_generate_id() if create_new else v.id,
        data=new_data,
        shape=(len(new_data),),
        dtype=v.dtype,
        label=f"proj_{A.label}({v.label})",
        color=v.color,
        visible=True,
        history=v.history + ("project_subspace",)
    )
    return [new_t]


# =============================================================================
# NORMS
# =============================================================================

def _op_frobenius_norm(targets: list, params: dict, create_new: bool) -> list:
    """Frobenius norm."""
    results = []
    for t in targets:
        arr = t.to_numpy()
        norm = float(np.linalg.norm(arr, 'fro'))
        new_t = TensorData(
            id=_generate_id(),
            data=(norm,),
            shape=(1,),
            dtype=TensorDType.NUMERIC,
            label=f"‖{t.label}‖F",
            color=t.color,
            visible=True,
            history=t.history + ("frobenius_norm",)
        )
        results.append(new_t)
    return results


def _op_l1_norm(targets: list, params: dict, create_new: bool) -> list:
    """L1 norm (max column sum for matrices, sum of abs for vectors)."""
    results = []
    for t in targets:
        arr = t.to_numpy()
        if _is_rank1(t):
            norm = float(np.sum(np.abs(arr)))
        else:
            norm = float(np.linalg.norm(arr, 1))
        new_t = TensorData(
            id=_generate_id(),
            data=(norm,),
            shape=(1,),
            dtype=TensorDType.NUMERIC,
            label=f"‖{t.label}‖₁",
            color=t.color,
            visible=True,
            history=t.history + ("l1_norm",)
        )
        results.append(new_t)
    return results


def _op_l2_norm(targets: list, params: dict, create_new: bool) -> list:
    """L2 norm (spectral norm for matrices, Euclidean for vectors)."""
    results = []
    for t in targets:
        arr = t.to_numpy()
        norm = float(np.linalg.norm(arr, 2))
        new_t = TensorData(
            id=_generate_id(),
            data=(norm,),
            shape=(1,),
            dtype=TensorDType.NUMERIC,
            label=f"‖{t.label}‖₂",
            color=t.color,
            visible=True,
            history=t.history + ("l2_norm",)
        )
        results.append(new_t)
    return results


def _op_inf_norm(targets: list, params: dict, create_new: bool) -> list:
    """Infinity norm (max row sum for matrices, max abs for vectors)."""
    results = []
    for t in targets:
        arr = t.to_numpy()
        if _is_rank1(t):
            norm = float(np.max(np.abs(arr)))
        else:
            norm = float(np.linalg.norm(arr, np.inf))
        new_t = TensorData(
            id=_generate_id(),
            data=(norm,),
            shape=(1,),
            dtype=TensorDType.NUMERIC,
            label=f"‖{t.label}‖∞",
            color=t.color,
            visible=True,
            history=t.history + ("inf_norm",)
        )
        results.append(new_t)
    return results


def _op_nuclear_norm(targets: list, params: dict, create_new: bool) -> list:
    """Nuclear norm (sum of singular values)."""
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        arr = t.to_numpy()
        norm = float(np.linalg.norm(arr, 'nuc'))
        new_t = TensorData(
            id=_generate_id(),
            data=(norm,),
            shape=(1,),
            dtype=TensorDType.NUMERIC,
            label=f"‖{t.label}‖*",
            color=t.color,
            visible=True,
            history=t.history + ("nuclear_norm",)
        )
        results.append(new_t)
    return results


# =============================================================================
# LINEAR SYSTEMS
# =============================================================================

def _op_gaussian_elimination(targets: list, params: dict, create_new: bool) -> list:
    """Row echelon form via Gaussian elimination."""
    from scipy import linalg as sp_linalg
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        arr = t.to_numpy().copy()
        m, n = arr.shape
        row = 0
        for col in range(n):
            if row >= m:
                break
            # Find pivot
            max_row = row + np.argmax(np.abs(arr[row:, col]))
            if np.abs(arr[max_row, col]) < 1e-10:
                continue
            arr[[row, max_row]] = arr[[max_row, row]]
            # Eliminate below
            for i in range(row + 1, m):
                factor = arr[i, col] / arr[row, col]
                arr[i, col:] -= factor * arr[row, col:]
            row += 1
        new_data = _numpy_to_tuples(arr)
        new_t = TensorData(
            id=_generate_id() if create_new else t.id,
            data=new_data,
            shape=arr.shape,
            dtype=t.dtype,
            label=f"REF({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("gaussian_elimination",)
        )
        results.append(new_t)
    return results


def _op_rref(targets: list, params: dict, create_new: bool) -> list:
    """Reduced row echelon form."""
    from sympy import Matrix
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        arr = t.to_numpy()
        sympy_mat = Matrix(arr.tolist())
        rref_mat, _ = sympy_mat.rref()
        rref_arr = np.array(rref_mat.tolist(), dtype=float)
        new_data = _numpy_to_tuples(rref_arr)
        new_t = TensorData(
            id=_generate_id() if create_new else t.id,
            data=new_data,
            shape=rref_arr.shape,
            dtype=t.dtype,
            label=f"RREF({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("rref",)
        )
        results.append(new_t)
    return results


def _op_back_substitution(targets: list, params: dict, create_new: bool) -> list:
    """Back substitution for upper triangular system."""
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        arr = t.to_numpy()
        if arr.shape[0] != arr.shape[1] - 1:
            # Expect augmented matrix [A|b]
            continue
        n = arr.shape[0]
        A = arr[:, :n]
        b = arr[:, n]
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if np.abs(A[i, i]) < 1e-10:
                continue
            x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        new_data = tuple(float(v) for v in x)
        new_t = TensorData(
            id=_generate_id(),
            data=new_data,
            shape=(n,),
            dtype=t.dtype,
            label=f"x({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("back_substitution",)
        )
        results.append(new_t)
    return results


def _op_solve_linear(targets: list, params: dict, create_new: bool) -> list:
    """Solve linear system Ax = b."""
    if len(targets) < 2:
        return []
    A, b = targets[0], targets[1]
    if not (_is_rank2(A) and _is_rank1(b)):
        return []
    try:
        x = np.linalg.solve(A.to_numpy(), b.to_numpy())
    except np.linalg.LinAlgError:
        return []
    new_data = tuple(float(v) for v in x)
    new_t = TensorData(
        id=_generate_id(),
        data=new_data,
        shape=(len(new_data),),
        dtype=A.dtype,
        label=f"x",
        color=b.color,
        visible=True,
        history=b.history + ("solve_linear",)
    )
    return [new_t]


def _op_least_squares(targets: list, params: dict, create_new: bool) -> list:
    """Least squares solution."""
    if len(targets) < 2:
        return []
    A, b = targets[0], targets[1]
    if not (_is_rank2(A) and _is_rank1(b)):
        return []
    try:
        x, residuals, rank, s = np.linalg.lstsq(A.to_numpy(), b.to_numpy(), rcond=None)
    except np.linalg.LinAlgError:
        return []
    new_data = tuple(float(v) for v in x)
    new_t = TensorData(
        id=_generate_id(),
        data=new_data,
        shape=(len(new_data),),
        dtype=A.dtype,
        label=f"x̂",
        color=b.color,
        visible=True,
        history=b.history + ("least_squares",)
    )
    return [new_t]


# =============================================================================
# SPECIAL MATRICES
# =============================================================================

def _op_symmetrize(targets: list, params: dict, create_new: bool) -> list:
    """Make symmetric: (A + Aᵀ) / 2."""
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        arr = t.to_numpy()
        sym = (arr + arr.T) / 2
        new_data = _numpy_to_tuples(sym)
        new_t = TensorData(
            id=_generate_id() if create_new else t.id,
            data=new_data,
            shape=sym.shape,
            dtype=t.dtype,
            label=f"sym({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("symmetrize",)
        )
        results.append(new_t)
    return results


def _op_skew_symmetrize(targets: list, params: dict, create_new: bool) -> list:
    """Make skew-symmetric: (A - Aᵀ) / 2."""
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        arr = t.to_numpy()
        skew = (arr - arr.T) / 2
        new_data = _numpy_to_tuples(skew)
        new_t = TensorData(
            id=_generate_id() if create_new else t.id,
            data=new_data,
            shape=skew.shape,
            dtype=t.dtype,
            label=f"skew({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("skew_symmetrize",)
        )
        results.append(new_t)
    return results


def _op_diagonalize(targets: list, params: dict, create_new: bool) -> list:
    """Extract diagonal as vector or create diagonal matrix."""
    results = []
    for t in targets:
        if _is_rank2(t):
            # Extract diagonal
            diag = np.diag(t.to_numpy())
            new_data = tuple(float(x) for x in diag)
            new_t = TensorData(
                id=_generate_id() if create_new else t.id,
                data=new_data,
                shape=(len(new_data),),
                dtype=t.dtype,
                label=f"diag({t.label})",
                color=t.color,
                visible=True,
                history=t.history + ("diagonalize",)
            )
            results.append(new_t)
        elif _is_rank1(t):
            # Create diagonal matrix from vector
            diag_mat = np.diag(t.to_numpy())
            new_data = _numpy_to_tuples(diag_mat)
            new_t = TensorData(
                id=_generate_id() if create_new else t.id,
                data=new_data,
                shape=diag_mat.shape,
                dtype=t.dtype,
                label=f"diag({t.label})",
                color=t.color,
                visible=True,
                history=t.history + ("diagonalize",)
            )
            results.append(new_t)
    return results


def _op_triangular_upper(targets: list, params: dict, create_new: bool) -> list:
    """Extract upper triangular part."""
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        upper = np.triu(t.to_numpy())
        new_data = _numpy_to_tuples(upper)
        new_t = TensorData(
            id=_generate_id() if create_new else t.id,
            data=new_data,
            shape=upper.shape,
            dtype=t.dtype,
            label=f"triu({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("triangular_upper",)
        )
        results.append(new_t)
    return results


def _op_triangular_lower(targets: list, params: dict, create_new: bool) -> list:
    """Extract lower triangular part."""
    results = []
    for t in targets:
        if not _is_rank2(t):
            continue
        lower = np.tril(t.to_numpy())
        new_data = _numpy_to_tuples(lower)
        new_t = TensorData(
            id=_generate_id() if create_new else t.id,
            data=new_data,
            shape=lower.shape,
            dtype=t.dtype,
            label=f"tril({t.label})",
            color=t.color,
            visible=True,
            history=t.history + ("triangular_lower",)
        )
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
