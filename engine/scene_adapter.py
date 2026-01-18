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
from state.models.tensor_model import TensorData, TensorDType
from state.selectors import get_vectors, get_matrices


def _build_tensor_faces(tensors: Tuple[TensorData, ...]) -> List[dict]:
    """Build triangle batches for rank-2 tensors shaped (N, 3)."""
    try:
        from scipy.spatial import ConvexHull
        from scipy.spatial.qhull import QhullError
    except Exception:
        ConvexHull = None
        QhullError = Exception

    faces = []
    for t in tensors:
        if t.rank != 2 or len(t.shape) < 2:
            continue
        points = None
        if t.shape[1] == 3 and t.shape[0] >= 3:
            points = np.array(t.data, dtype=np.float32)
        elif t.shape[0] == 3 and t.shape[1] >= 3:
            points = np.array(t.data, dtype=np.float32).T
        if points is None:
            continue
        if points.ndim != 2 or points.shape[1] != 3:
            continue

        triangles = []
        if points.shape[0] == 3:
            triangles = [(0, 1, 2)]
        elif ConvexHull is not None:
            try:
                hull = ConvexHull(points)
                triangles = [tuple(map(int, tri)) for tri in hull.simplices]
            except QhullError:
                triangles = []

        if not triangles:
            triangles = [(0, i, i + 1) for i in range(1, points.shape[0] - 1)]

        vertices = []
        normals = []
        colors = []
        base_color = t.color[:3] if len(t.color) >= 3 else (0.6, 0.7, 0.9)
        alpha = 0.25
        for tri in triangles:
            v0, v1, v2 = points[tri[0]], points[tri[1]], points[tri[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = float(np.linalg.norm(normal))
            if norm > 1e-6:
                normal = normal / norm
            else:
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            for v in (v0, v1, v2):
                vertices.append(v)
                normals.append(normal)
                colors.append((base_color[0], base_color[1], base_color[2], alpha))

        if vertices:
            faces.append({
                "vertices": np.array(vertices, dtype=np.float32),
                "normals": np.array(normals, dtype=np.float32),
                "colors": np.array(colors, dtype=np.float32),
            })

    return faces


@dataclass
class RendererVector:
    """
    Vector representation for the renderer.

    Converts TensorData (rank-1) to a format suitable for rendering.
    """
    id: str
    coords: np.ndarray
    color: tuple
    label: str
    visible: bool = True

    @staticmethod
    def from_tensor_data(t: TensorData) -> 'RendererVector':
        """Create RendererVector from TensorData (rank-1 tensor)."""
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

    Converts TensorData (rank-2, non-image) to a format suitable for rendering.
    """
    matrix: np.ndarray
    label: str
    visible: bool = True
    color: tuple = (0.8, 0.5, 0.2, 0.6)

    @staticmethod
    def from_tensor_data(t: TensorData) -> 'RendererMatrix':
        """Create RendererMatrix from TensorData (rank-2 tensor)."""
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

        # Get all vectors via unified selector (includes both legacy and tensor store)
        all_vectors = get_vectors(state)
        self._vectors = [RendererVector.from_tensor_data(v) for v in all_vectors]

        # Operation step overlays (render hints)
        if getattr(state, "operations", None) and state.operations.steps:
            step_index = max(0, min(state.operations.step_index, len(state.operations.steps) - 1))
            step = state.operations.steps[step_index]
            for i, hint in enumerate(getattr(step, "render_vectors", ())):
                coords = list(getattr(hint, "coords", (0.0, 0.0, 0.0)))
                coords = (coords + [0.0, 0.0, 0.0])[:3]
                self._vectors.append(RendererVector(
                    id=f"op:{step_index}:{i}",
                    coords=np.array(coords, dtype=np.float32),
                    color=getattr(hint, "color", (0.9, 0.9, 0.2)),
                    label=getattr(hint, "label", f"op_step_{step_index+1}"),
                    visible=getattr(hint, "visible", True),
                ))

        if (
            state.active_mode == "vectors"
            and state.preview_enabled
            and state.input_matrix_preview_vectors
        ):
            preview_color = (0.4, 0.6, 0.9)
            for idx, coords in enumerate(state.input_matrix_preview_vectors):
                preview_vector = RendererVector(
                    id=f"preview:{idx}",
                    coords=np.array(
                        (list(coords) + [0.0, 0.0, 0.0])[:3],
                        dtype=np.float32,
                    ),
                    color=preview_color,
                    label=f"preview_c{idx+1}",
                    visible=True,
                )
                self._vectors.append(preview_vector)

        # Get all matrices via unified selector (includes both legacy and tensor store)
        all_matrices = get_matrices(state)
        self._matrices = []
        for m in all_matrices:
            try:
                matrix_np = np.array(m.values, dtype=np.float32)
            except Exception:
                continue
            self._matrices.append({
                'id': m.id,
                'matrix': matrix_np,
                'label': m.label,
                'visible': m.visible,
                'color': tuple(m.color) + (0.6,) if len(m.color) == 3 else (0.8, 0.5, 0.2, 0.6),
                'transformations': [],
            })
            # --- Workaround: render matrix columns as vectors for visibility ---
            self._vectors.extend(_matrix_columns_as_vectors(
                matrix_np,
                matrix_id=m.id,
                label=m.label,
                color=m.color[:3] if len(m.color) >= 3 else (0.8, 0.5, 0.2),
                visible=m.visible,
            ))

        # Selection - tensor-only
        self._selected_object = None
        self._selection_type = None

        vector_by_id = {v.id: v for v in self._vectors}
        matrix_by_id = {m.get('id'): m for m in self._matrices}

        # Determine selection ID
        selection_id = state.selected_tensor_id

        if selection_id:
            # Check if it's a vector
            if selection_id in vector_by_id:
                self._selection_type = 'vector'
                self._selected_object = vector_by_id[selection_id]
            # Check if it's a matrix
            elif selection_id in matrix_by_id:
                self._selection_type = 'matrix'
                self._selected_object = matrix_by_id[selection_id]

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

        self._tensor_faces = []
        if getattr(state, "view_show_tensor_faces", False):
            self._tensor_faces = _build_tensor_faces(state.tensors)

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

    @property
    def tensor_faces(self) -> List[dict]:
        """Triangle batches for tensor face rendering."""
        return self._tensor_faces


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

    @property
    def tensor_faces(self) -> List[dict]: ...


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
