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
from typing import List, Optional, Protocol, Tuple
from dataclasses import dataclass

from state.app_state import AppState
from state.models import VectorData, MatrixData
from state.models.tensor_model import TensorData


@dataclass
class RendererVector:
    """
    Vector representation for the renderer.

    Mirrors the old Vector3D interface but is derived from VectorData.
    """
    id: str
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
            id=v.id,
            coords=np.array(coords, dtype=np.float32),
            color=v.color,
            label=v.label,
            visible=v.visible,
        )

    @staticmethod
    def from_tensor_data(t: TensorData) -> 'RendererVector':
        coords = list(t.coords)
        if len(coords) < 3:
            coords = coords + [0.0] * (3 - len(coords))
        elif len(coords) > 3:
            coords = coords[:3]
        return RendererVector(
            id=t.id,
            coords=np.array(coords, dtype=np.float32),
            color=t.color,
            label=t.label,
            visible=t.visible,
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

    @staticmethod
    def from_tensor_data(t: TensorData) -> 'RendererMatrix':
        return RendererMatrix(
            matrix=np.array(t.values, dtype=np.float32),
            label=t.label,
            visible=t.visible,
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

        # Convert legacy vectors to renderer format
        self._vectors = [RendererVector.from_vector_data(v) for v in state.vectors]

        # Convert tensor vectors to renderer format
        for t in state.tensors:
            if t.is_vector:
                self._vectors.append(RendererVector.from_tensor_data(t))

        if (
            state.active_mode == "vectors"
            and state.preview_enabled
            and state.input_matrix_preview_vectors
        ):
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
        self._matrices = []
        for m in state.matrices:
            matrix_np = np.array(m.values, dtype=np.float32)
            self._matrices.append({
                'id': m.id,
                'matrix': matrix_np,
                'label': m.label,
                'visible': m.visible,
                'color': (0.8, 0.5, 0.2, 0.6),
                'transformations': [],
            })
            # --- Workaround: render matrix columns as vectors for visibility ---
            self._vectors.extend(_matrix_columns_as_vectors(
                matrix_np,
                matrix_id=m.id,
                label=m.label,
                color=(0.8, 0.5, 0.2),
                visible=m.visible,
            ))

        for t in state.tensors:
            if t.is_matrix:
                try:
                    matrix = np.array(t.values, dtype=np.float32)
                except Exception:
                    continue
                self._matrices.append({
                    'id': t.id,
                    'matrix': matrix,
                    'label': t.label,
                    'visible': t.visible,
                    'color': (0.4, 0.9, 0.6, 0.6),
                    'transformations': [],
                })
                self._vectors.extend(_matrix_columns_as_vectors(
                    matrix,
                    matrix_id=t.id,
                    label=t.label,
                    color=t.color[:3] if len(t.color) >= 3 else (0.4, 0.9, 0.6),
                    visible=t.visible,
                ))

        # Selection (prefer tensor selection when present)
        self._selected_object = None
        self._selection_type = None

        vector_by_id = {v.id: v for v in self._vectors}
        matrix_by_id = {m.get('id'): m for m in self._matrices}

        if state.selected_tensor_id:
            selected_tensor = next(
                (t for t in state.tensors if t.id == state.selected_tensor_id),
                None
            )
            if selected_tensor is not None:
                if selected_tensor.is_vector:
                    self._selection_type = 'vector'
                    self._selected_object = vector_by_id.get(selected_tensor.id)
                elif selected_tensor.is_matrix:
                    self._selection_type = 'matrix'
                    self._selected_object = matrix_by_id.get(selected_tensor.id)

        if self._selected_object is None:
            self._selection_type = state.selected_type
            if state.selected_id and state.selected_type == 'vector':
                self._selected_object = vector_by_id.get(state.selected_id)
            elif state.selected_id and state.selected_type == 'matrix':
                self._selected_object = matrix_by_id.get(state.selected_id)

        # Preview matrix (from input_matrix if preview is enabled)
        self._preview_matrix = None
        if state.preview_enabled and state.input_matrix:
            self._preview_matrix = np.array(state.input_matrix, dtype=np.float32)

        self._show_matrix_plot = state.matrix_plot_enabled

        # Vector span visualization based on the most recent add/subtract operation
        self._vector_span = None
        if state.operation_history:
            last = state.operation_history[-1]
            if last.operation_name in ("add", "subtract") and len(last.target_ids) >= 2:
                v1 = vector_by_id.get(last.target_ids[0])
                v2 = vector_by_id.get(last.target_ids[1])
                if v1 is not None and v2 is not None:
                    self._vector_span = (v1, v2)

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

    @property
    def vector_span(self) -> Optional[Tuple[RendererVector, RendererVector]]:
        """Optional vector pair to visualize as a span/parallelogram."""
        return self._vector_span


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

    @property
    def vector_span(self) -> Optional[Tuple['RendererVector', 'RendererVector']]: ...


def create_scene_from_state(state: AppState) -> SceneAdapter:
    """
    Factory function to create a SceneAdapter from AppState.

    This is the main entry point for the renderer.
    """
    return SceneAdapter(state)


def _matrix_columns_as_vectors(matrix: np.ndarray, matrix_id: str, label: str,
                               color: tuple, visible: bool):
    """
    Convert matrix columns into RendererVector instances for rendering.

    This is a temporary workaround while dedicated matrix rendering is incomplete.
    Each column is treated as a vector rooted at the origin.
    """
    vectors = []
    if matrix.ndim != 2:
        return vectors

    rows, cols = matrix.shape
    for j in range(cols):
        col = matrix[:, j]
        coords = list(col.tolist())
        if len(coords) < 3:
            coords += [0.0] * (3 - len(coords))
        elif len(coords) > 3:
            coords = coords[:3]

        # Slightly brighten to distinguish from tensor vectors
        r, g, b = color
        bright = lambda c: min(1.0, c * 0.8 + 0.2)
        vec_color = (bright(r), bright(g), bright(b))

        vectors.append(RendererVector(
            id=f"{matrix_id}_col{j+1}",
            coords=np.array(coords, dtype=np.float32),
            color=vec_color,
            label=f"{label}·e{j+1}",
            visible=visible,
        ))
    return vectors
