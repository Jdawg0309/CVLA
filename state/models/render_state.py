"""
Render state for camera/grid/highlight configuration.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RenderState:
    camera: Tuple[Tuple[str, float], ...] = ()
    grid_config: Tuple[Tuple[str, float], ...] = ()
    highlight_config: Tuple[Tuple[str, float], ...] = ()
