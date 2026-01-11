"""
Scene Adapter - Compatibility Layer for Renderer

This adapter provides a Scene-like interface that reads from AppState.
The renderer expects certain attributes; this adapter provides them
without owning any state.

IMPORTANT:
- This is READ-ONLY
- It derives visual data from AppState
- It NEVER stores authoritative data
- It NEVER mutates the AppState
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from runtime.app_state import AppState
from state.models import VectorData, MatrixData


@dataclass
class RendererVector:
    """
    Vector representation for the renderer.

    Mirrors the old Vector3D interface but is derived from VectorData.
    """
    coords: np.ndarray
    color: tuple
    label: str
    visible: bool = True

    @staticmethod
    def from_vector_data(v: VectorData) -> 'RendererVector':
        return RendererVector(
            coords=np.array(v.coords, dtype=np.float32),
            color=v.color,
            label=v.label,
            visible=v.visible,
        )


@dataclass
class RendererMatrix:
    """
    Matrix representation for the renderer.
    """
    matrix: np.ndarray
    label: str
    visible: bool = True
    color: tuple = (0.8, 0.5, 0.2, 0.6)

    @staticmethod
    def from_matrix_data(m: MatrixData) -> 'RendererMatrix':
        return RendererMatrix(
            matrix=np.array(m.values, dtype=np.float32),
            label=m.label,
            visible=m.visible,
        )


class SceneAdapter:
    """
    Adapter that provides a Scene-like interface from AppState.

    Usage:
        state = store.get_state()
        scene = SceneAdapter(state)
        renderer.render(scene)

    The renderer can access:
        scene.vectors - List of RendererVector
        scene.matrices - List of RendererMatrix (as dicts for compatibility)
        scene.planes - List of plane dicts
        scene.selected_object - The selected object
        scene.selection_type - 'vector', 'matrix', etc.
        scene.preview_matrix - Preview transformation matrix
    """

    def __init__(self, state: AppState):
        """
        Create adapter from current state.

        Args:
            state: Current AppState (read-only)
        """
        self._state = state

        # Convert vectors to renderer format
        self._vectors = [
            RendererVector.from_vector_data(v)
            for v in state.vectors
        ]

        # Convert matrices to renderer format (dict-like for compatibility)
        self._matrices = [
            {
                'matrix': np.array(m.values, dtype=np.float32),
                'label': m.label,
                'visible': m.visible,
                'color': (0.8, 0.5, 0.2, 0.6),
                'transformations': [],
            }
            for m in state.matrices
        ]

        # Planes (if any)
        self._planes = [
            {
                'equation': np.array(p.equation, dtype=np.float32),
                'color': p.color,
                'label': p.label,
                'visible': p.visible,
            }
            for p in state.planes
        ]

        # Selection
        self._selected_object = None
        self._selection_type = state.selected_type

        if state.selected_id and state.selected_type == 'vector':
            for rv in self._vectors:
                # Match by label (since we don't have id on RendererVector)
                for v in state.vectors:
                    if v.id == state.selected_id and rv.label == v.label:
                        self._selected_object = rv
                        break

        elif state.selected_id and state.selected_type == 'matrix':
            for i, m in enumerate(state.matrices):
                if m.id == state.selected_id:
                    self._selected_object = self._matrices[i]
                    break

        # Preview matrix (from input_matrix if preview is enabled)
        self._preview_matrix = None
        if state.preview_enabled and state.input_matrix:
            self._preview_matrix = np.array(state.input_matrix, dtype=np.float32)

    @property
    def vectors(self) -> List[RendererVector]:
        """Get all vectors for rendering."""
        return self._vectors

    @property
    def matrices(self) -> List[dict]:
        """Get all matrices for rendering (dict format for compatibility)."""
        return self._matrices

    @property
    def planes(self) -> List[dict]:
        """Get all planes for rendering."""
        return self._planes

    @property
    def selected_object(self):
        """Get currently selected object."""
        return self._selected_object

    @property
    def selection_type(self) -> Optional[str]:
        """Get selection type."""
        return self._selection_type

    @property
    def preview_matrix(self) -> Optional[np.ndarray]:
        """Get preview transformation matrix."""
        return self._preview_matrix


def create_scene_from_state(state: AppState) -> SceneAdapter:
    """
    Factory function to create a SceneAdapter from AppState.

    This is the main entry point for the renderer.
    """
    return SceneAdapter(state)
