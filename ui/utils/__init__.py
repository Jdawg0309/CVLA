"""Utility helpers for the UI."""

import imgui


def set_next_window_position(x: float, y: float, cond=None):
    """Call set_next_window_position with optional cond support."""
    func = imgui.set_next_window_position
    if cond:
        try:
            func(x, y, cond=cond)
            return
        except TypeError:
            pass
    func(x, y)


def set_next_window_size(size_tuple, cond=None):
    """Call set_next_window_size with optional cond support."""
    width, height = size_tuple
    func = imgui.set_next_window_size
    if cond:
        try:
            func(width, height, cond=cond)
            return
        except TypeError:
            pass
    func(width, height)


__all__ = ["set_next_window_position", "set_next_window_size"]
