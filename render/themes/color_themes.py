"""
Color theme definitions for CVLA visual system.

Provides configurable color themes for grid, axes, background, and post-processing.
"""

from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass(frozen=True)
class ColorTheme:
    """
    Immutable color theme configuration.

    All colors are RGBA tuples (0.0-1.0).
    """
    name: str

    # Background colors
    background_color: Tuple[float, float, float, float]
    background_color_cube: Tuple[float, float, float, float]

    # Grid colors
    grid_color_major: Tuple[float, float, float, float]
    grid_color_minor: Tuple[float, float, float, float]
    grid_color_subminor: Tuple[float, float, float, float]

    # Axis colors
    axis_color_x: Tuple[float, float, float, float]
    axis_color_y: Tuple[float, float, float, float]
    axis_color_z: Tuple[float, float, float, float]

    # Cube face colors (6 faces)
    cube_face_colors: Tuple[
        Tuple[float, float, float, float],
        Tuple[float, float, float, float],
        Tuple[float, float, float, float],
        Tuple[float, float, float, float],
        Tuple[float, float, float, float],
        Tuple[float, float, float, float],
    ]

    # Corner and border colors
    corner_color: Tuple[float, float, float, float]
    border_color: Tuple[float, float, float, float]

    # Origin point color
    origin_color: Tuple[float, float, float, float]

    # Faint axis color (extended axes)
    faint_axis_color: Tuple[float, float, float, float]

    # Post-processing parameters
    bloom_intensity: float = 0.3
    bloom_threshold: float = 0.8
    exposure: float = 1.0
    gamma: float = 2.2


# Dark Modern (default) - Clean dark theme with subtle colors
DARK_MODERN = ColorTheme(
    name="dark_modern",
    background_color=(0.08, 0.08, 0.10, 1.0),
    background_color_cube=(0.05, 0.06, 0.08, 1.0),
    grid_color_major=(0.35, 0.37, 0.40, 0.45),
    grid_color_minor=(0.22, 0.24, 0.28, 0.22),
    grid_color_subminor=(0.20, 0.22, 0.26, 0.12),
    axis_color_x=(0.95, 0.45, 0.45, 1.0),
    axis_color_y=(0.45, 0.95, 0.45, 1.0),
    axis_color_z=(0.55, 0.60, 1.00, 1.0),
    cube_face_colors=(
        (0.30, 0.30, 0.80, 0.05),
        (0.80, 0.30, 0.30, 0.05),
        (0.30, 0.80, 0.30, 0.05),
        (0.80, 0.80, 0.30, 0.05),
        (0.80, 0.30, 0.80, 0.05),
        (0.30, 0.80, 0.80, 0.05),
    ),
    corner_color=(0.80, 0.80, 0.90, 0.60),
    border_color=(0.35, 0.35, 0.38, 0.25),
    origin_color=(0.90, 0.90, 0.95, 0.90),
    faint_axis_color=(0.60, 0.60, 0.60, 0.35),
    bloom_intensity=0.3,
    bloom_threshold=0.8,
    exposure=1.0,
    gamma=2.2,
)


# Midnight Blue - Deep blue theme
MIDNIGHT_BLUE = ColorTheme(
    name="midnight_blue",
    background_color=(0.04, 0.05, 0.12, 1.0),
    background_color_cube=(0.03, 0.04, 0.10, 1.0),
    grid_color_major=(0.25, 0.30, 0.50, 0.50),
    grid_color_minor=(0.18, 0.22, 0.38, 0.25),
    grid_color_subminor=(0.15, 0.18, 0.32, 0.15),
    axis_color_x=(1.00, 0.50, 0.50, 1.0),
    axis_color_y=(0.50, 1.00, 0.50, 1.0),
    axis_color_z=(0.50, 0.70, 1.00, 1.0),
    cube_face_colors=(
        (0.20, 0.30, 0.70, 0.06),
        (0.70, 0.25, 0.25, 0.06),
        (0.25, 0.70, 0.30, 0.06),
        (0.70, 0.70, 0.25, 0.06),
        (0.60, 0.25, 0.70, 0.06),
        (0.25, 0.65, 0.70, 0.06),
    ),
    corner_color=(0.70, 0.75, 0.95, 0.55),
    border_color=(0.30, 0.35, 0.50, 0.30),
    origin_color=(0.85, 0.90, 1.00, 0.90),
    faint_axis_color=(0.50, 0.55, 0.70, 0.30),
    bloom_intensity=0.35,
    bloom_threshold=0.75,
    exposure=1.05,
    gamma=2.2,
)


# Warm Sunset - Orange/amber tones
WARM_SUNSET = ColorTheme(
    name="warm_sunset",
    background_color=(0.12, 0.08, 0.06, 1.0),
    background_color_cube=(0.10, 0.06, 0.04, 1.0),
    grid_color_major=(0.50, 0.40, 0.30, 0.45),
    grid_color_minor=(0.35, 0.28, 0.22, 0.25),
    grid_color_subminor=(0.30, 0.24, 0.18, 0.15),
    axis_color_x=(1.00, 0.55, 0.30, 1.0),
    axis_color_y=(0.70, 0.95, 0.40, 1.0),
    axis_color_z=(0.50, 0.65, 1.00, 1.0),
    cube_face_colors=(
        (0.80, 0.50, 0.20, 0.05),
        (0.90, 0.35, 0.20, 0.05),
        (0.60, 0.80, 0.30, 0.05),
        (0.95, 0.80, 0.30, 0.05),
        (0.80, 0.40, 0.60, 0.05),
        (0.40, 0.70, 0.75, 0.05),
    ),
    corner_color=(0.95, 0.85, 0.70, 0.55),
    border_color=(0.50, 0.40, 0.30, 0.28),
    origin_color=(1.00, 0.95, 0.85, 0.90),
    faint_axis_color=(0.65, 0.55, 0.45, 0.35),
    bloom_intensity=0.4,
    bloom_threshold=0.7,
    exposure=1.1,
    gamma=2.1,
)


# Light Academic - Clean light theme for presentations
LIGHT_ACADEMIC = ColorTheme(
    name="light_academic",
    background_color=(0.95, 0.95, 0.96, 1.0),
    background_color_cube=(0.92, 0.93, 0.95, 1.0),
    grid_color_major=(0.55, 0.55, 0.60, 0.50),
    grid_color_minor=(0.70, 0.70, 0.75, 0.30),
    grid_color_subminor=(0.80, 0.80, 0.82, 0.20),
    axis_color_x=(0.85, 0.25, 0.25, 1.0),
    axis_color_y=(0.20, 0.65, 0.25, 1.0),
    axis_color_z=(0.25, 0.40, 0.85, 1.0),
    cube_face_colors=(
        (0.40, 0.45, 0.75, 0.08),
        (0.75, 0.40, 0.40, 0.08),
        (0.40, 0.70, 0.45, 0.08),
        (0.75, 0.70, 0.35, 0.08),
        (0.65, 0.40, 0.70, 0.08),
        (0.40, 0.65, 0.70, 0.08),
    ),
    corner_color=(0.40, 0.40, 0.50, 0.50),
    border_color=(0.60, 0.60, 0.65, 0.35),
    origin_color=(0.30, 0.30, 0.35, 0.90),
    faint_axis_color=(0.55, 0.55, 0.60, 0.25),
    bloom_intensity=0.15,
    bloom_threshold=0.95,
    exposure=0.9,
    gamma=2.4,
)


# Neon Cyberpunk - Vibrant neon colors
NEON_CYBERPUNK = ColorTheme(
    name="neon_cyberpunk",
    background_color=(0.02, 0.02, 0.05, 1.0),
    background_color_cube=(0.01, 0.01, 0.03, 1.0),
    grid_color_major=(0.00, 0.80, 0.90, 0.40),
    grid_color_minor=(0.60, 0.00, 0.80, 0.25),
    grid_color_subminor=(0.80, 0.00, 0.50, 0.15),
    axis_color_x=(1.00, 0.20, 0.50, 1.0),
    axis_color_y=(0.20, 1.00, 0.50, 1.0),
    axis_color_z=(0.20, 0.60, 1.00, 1.0),
    cube_face_colors=(
        (0.00, 0.50, 0.90, 0.08),
        (0.90, 0.10, 0.40, 0.08),
        (0.10, 0.90, 0.40, 0.08),
        (0.90, 0.80, 0.10, 0.08),
        (0.80, 0.10, 0.90, 0.08),
        (0.10, 0.80, 0.90, 0.08),
    ),
    corner_color=(0.00, 0.95, 1.00, 0.70),
    border_color=(0.50, 0.00, 0.80, 0.40),
    origin_color=(1.00, 1.00, 1.00, 1.00),
    faint_axis_color=(0.40, 0.00, 0.60, 0.30),
    bloom_intensity=0.6,
    bloom_threshold=0.5,
    exposure=1.2,
    gamma=2.0,
)


# High Contrast - Accessibility-focused
HIGH_CONTRAST = ColorTheme(
    name="high_contrast",
    background_color=(0.00, 0.00, 0.00, 1.0),
    background_color_cube=(0.00, 0.00, 0.00, 1.0),
    grid_color_major=(1.00, 1.00, 1.00, 0.60),
    grid_color_minor=(0.80, 0.80, 0.80, 0.35),
    grid_color_subminor=(0.60, 0.60, 0.60, 0.20),
    axis_color_x=(1.00, 0.30, 0.30, 1.0),
    axis_color_y=(0.30, 1.00, 0.30, 1.0),
    axis_color_z=(0.30, 0.50, 1.00, 1.0),
    cube_face_colors=(
        (0.30, 0.30, 1.00, 0.10),
        (1.00, 0.30, 0.30, 0.10),
        (0.30, 1.00, 0.30, 0.10),
        (1.00, 1.00, 0.30, 0.10),
        (1.00, 0.30, 1.00, 0.10),
        (0.30, 1.00, 1.00, 0.10),
    ),
    corner_color=(1.00, 1.00, 1.00, 0.80),
    border_color=(1.00, 1.00, 1.00, 0.50),
    origin_color=(1.00, 1.00, 1.00, 1.00),
    faint_axis_color=(0.70, 0.70, 0.70, 0.40),
    bloom_intensity=0.2,
    bloom_threshold=0.9,
    exposure=1.0,
    gamma=2.2,
)


# Theme registry
THEMES: Dict[str, ColorTheme] = {
    "dark_modern": DARK_MODERN,
    "midnight_blue": MIDNIGHT_BLUE,
    "warm_sunset": WARM_SUNSET,
    "light_academic": LIGHT_ACADEMIC,
    "neon_cyberpunk": NEON_CYBERPUNK,
    "high_contrast": HIGH_CONTRAST,
}

# Theme display names for UI
THEME_DISPLAY_NAMES: Dict[str, str] = {
    "dark_modern": "Dark Modern",
    "midnight_blue": "Midnight Blue",
    "warm_sunset": "Warm Sunset",
    "light_academic": "Light Academic",
    "neon_cyberpunk": "Neon Cyberpunk",
    "high_contrast": "High Contrast",
}

DEFAULT_THEME = "neon_cyberpunk"


def get_theme(name: str) -> ColorTheme:
    """Get a theme by name, falling back to default if not found."""
    return THEMES.get(name, DARK_MODERN)


def list_themes() -> list:
    """Return list of available theme names."""
    return list(THEMES.keys())
