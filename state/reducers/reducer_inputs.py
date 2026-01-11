"""
UI input action reducers.
"""

from dataclasses import replace

from state.actions import (
    SetInputVector, SetInputMatrixCell, SetInputMatrixSize, SetInputMatrixLabel,
    SetImagePath, SetSamplePattern, SetSampleSize,
    SetTransformRotation, SetTransformScale, SetSelectedKernel,
    SetImageNormalizeMean, SetImageNormalizeStd,
)


def reduce_inputs(state, action):
    if isinstance(action, SetInputVector):
        return replace(state,
            input_vector_coords=action.coords if action.coords is not None else state.input_vector_coords,
            input_vector_label=action.label if action.label is not None else state.input_vector_label,
            input_vector_color=action.color if action.color is not None else state.input_vector_color,
        )

    if isinstance(action, SetInputMatrixCell):
        new_matrix = list(list(row) for row in state.input_matrix)
        while len(new_matrix) <= action.row:
            new_matrix.append([0.0] * state.input_matrix_size)
        while len(new_matrix[action.row]) <= action.col:
            new_matrix[action.row].append(0.0)
        new_matrix[action.row][action.col] = action.value
        return replace(state, input_matrix=tuple(tuple(row) for row in new_matrix))

    if isinstance(action, SetInputMatrixSize):
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

    if isinstance(action, SetImageNormalizeMean):
        return replace(state, input_image_normalize_mean=action.mean)

    if isinstance(action, SetImageNormalizeStd):
        safe_std = max(0.001, action.std)
        return replace(state, input_image_normalize_std=safe_std)

    return None
