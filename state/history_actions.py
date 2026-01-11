"""
Undo/redo action definitions.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Undo:
    """Undo the last action."""
    pass


@dataclass(frozen=True)
class Redo:
    """Redo the last undone action."""
    pass
