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
from typing import List, Optional, Protocol
from dataclasses import dataclass

from state.app_state import AppState
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
        coords = list(v.coords)
        if len(coords) < 3:
            coords = coords + [0.0] * (3 - len(coords))
        elif len(coords) > 3:
            coords = coords[:3]
        return RendererVector(
            coords=np.array(coords, dtype=np.float32),
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

        if state.active_mode == "vectors" and state.input_matrix_preview_vectors:
            preview_color = (0.4, 0.6, 0.9)
            for idx, coords in enumerate(state.input_matrix_preview_vectors):
                preview_vector = RendererVector(
                    coords=np.array(
                        (list(coords) + [0.0, 0.0, 0.0])[:3],
                        dtype=np.float32,
                    ),
                    color=preview_color,
                    label=f"preview_c{idx+1}",
                    visible=True,
                )
                self._vectors.append(preview_vector)

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

        self._show_matrix_plot = state.matrix_plot_enabled

    @property
    def vectors(self) -> List[RendererVector]:
        """Get all vectors for rendering."""
        return self._vectors

    @property
    def matrices(self) -> List[dict]:
        """Get all matrices for rendering (dict format for compatibility)."""
        return self._matrices

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

    @property
    def show_matrix_plot(self) -> bool:
        """Indicate whether the 3D matrix plot should render."""
        return self._show_matrix_plot


class RendererSceneProtocol(Protocol):
    """Protocol that renderers expect from scene adapters."""

    @property
    def vectors(self) -> List['RendererVector']: ...

    @property
    def matrices(self) -> List[dict]: ...

    @property
    def selected_object(self) -> Optional['RendererVector']: ...

    @property
    def selection_type(self) -> Optional[str]: ...

    @property
    def preview_matrix(self) -> Optional[np.ndarray]: ...

    @property
    def show_matrix_plot(self) -> bool: ...


def create_scene_from_state(state: AppState) -> SceneAdapter:
    """
    Factory function to create a SceneAdapter from AppState.

    This is the main entry point for the renderer.
    """
    return SceneAdapter(state)
