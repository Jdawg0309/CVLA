"""
Navigation and UI state action definitions.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SetActiveTab:
    """Switch to a different tab."""
    tab: str  # 'vectors', 'matrices', 'systems', 'images', 'visualize'


@dataclass(frozen=True)
class SetActiveMode:
    """Switch to a different mode."""
    mode: str  # 'vectors', 'matrices', 'systems', 'images', 'visualize'


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
class SetActiveImageTab:
    """Set which image sub-tab is active (raw/preprocess)."""
    tab: str


@dataclass(frozen=True)
class ClearSelection:
    """Clear the current selection."""
    pass


@dataclass(frozen=True)
class SetTheme:
    """Set the UI theme ('dark', 'light', 'high-contrast')."""
    theme: str


@dataclass(frozen=True)
class SetActiveTool:
    """Set the active tool (selection, move, add, etc.)."""
    tool: str


@dataclass(frozen=True)
class SetViewPreset:
    """Set view preset (cube, xy, xz, yz)."""
    preset: str


@dataclass(frozen=True)
class SetViewUpAxis:
    """Set up axis (x, y, z)."""
    axis: str


@dataclass(frozen=True)
class ToggleViewGrid:
    """Toggle grid visibility."""
    pass


@dataclass(frozen=True)
class ToggleViewAxes:
    """Toggle axes visibility."""
    pass


@dataclass(frozen=True)
class ToggleViewLabels:
    """Toggle label visibility."""
    pass


@dataclass(frozen=True)
class SetViewGridSize:
    """Set grid size."""
    size: int


@dataclass(frozen=True)
class SetViewMajorTick:
    """Set major tick spacing."""
    value: int


@dataclass(frozen=True)
class SetViewMinorTick:
    """Set minor tick spacing."""
    value: int


@dataclass(frozen=True)
class ToggleViewAutoRotate:
    """Toggle auto-rotation for cubic view."""
    pass


@dataclass(frozen=True)
class SetViewRotationSpeed:
    """Set cubic view rotation speed."""
    speed: float


@dataclass(frozen=True)
class ToggleViewCubeFaces:
    """Toggle cube face visibility."""
    pass


@dataclass(frozen=True)
class ToggleViewCubeCorners:
    """Toggle cube corner indicators."""
    pass


@dataclass(frozen=True)
class SetViewCubicGridDensity:
    """Set cubic grid density."""
    density: float


@dataclass(frozen=True)
class SetViewCubeFaceOpacity:
    """Set cube face opacity."""
    opacity: float


@dataclass(frozen=True)
class ToggleView2D:
    """Toggle 2D camera mode."""
    pass
