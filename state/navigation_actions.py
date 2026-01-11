"""
Navigation and UI state action definitions.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SetActiveTab:
    """Switch to a different tab."""
    tab: str  # 'vectors', 'matrices', 'systems', 'images', 'visualize'


@dataclass(frozen=True)
class ToggleMatrixEditor:
    """Toggle the matrix editor visibility."""
    pass


@dataclass(frozen=True)
class ToggleMatrixValues:
    """Toggle showing matrix values in the Images tab."""
    pass


@dataclass(frozen=True)
class ToggleImageOnGrid:
    """Toggle image rendering on the 3D grid."""
    pass


@dataclass(frozen=True)
class TogglePreview:
    """Toggle matrix preview mode."""
    pass


@dataclass(frozen=True)
class ClearSelection:
    """Clear the current selection."""
    pass
