"""
Image action reducers.
"""

from state.reducer_image_load import reduce_image_load
from state.reducer_image_kernel import reduce_image_kernel
from state.reducer_image_transform import reduce_image_transform
from state.reducer_image_basic import reduce_image_basic


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

    result = reduce_image_basic(state, action)
    if result is not None:
        return result

    return None
