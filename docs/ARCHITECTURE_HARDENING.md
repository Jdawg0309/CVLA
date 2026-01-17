# CVLA Architecture Hardening Document

**Purpose:** Finish and harden the existing architecture before adding CLI or refactoring UI.
**Scope:** Minimal patches only. No rewrites, no new features, no architecture pivots.

---

# 1. LOOSE ENDS

## 1.1 Split Ownership Issues

### L1: Camera Lives Outside Redux

**Location:** `App.camera` (instance of `Camera` class)

**Problem:** Camera state (radius, theta, phi, target, mode_2d, view_preset, etc.) is mutable and modified directly by:
- `app_handlers.py:on_mouse_move` → `camera.orbit()`
- `app_handlers.py:on_scroll` → `camera.zoom()`
- `app_handlers.py:on_mouse_button` → `camera.focus_on_vector()`
- `app_handlers.py:on_key` → `camera.reset()`
- `toolbar.py` (indirectly via view_config)

**Why it matters:** CLI cannot control camera position through the Store. Camera state is invisible to undo/redo.

**Severity:** Medium. Intentional for performance, but creates two sources of truth.

---

### L2: ViewConfig Lives Outside Redux

**Location:** `App.view_config` (instance of `ViewConfig` class)

**Problem:** ViewConfig is a mutable object with ~40 fields (grid_mode, show_grid, show_axes, show_labels, show_cube_faces, auto_rotate, etc.). Modified via:
- `view_config.update(show_grid=...)` in `toolbar.py:47-51`
- `app_handlers.py:on_key` direct attribute assignment

**Why it matters:** View settings cannot be controlled through Store. A CLI would need direct ViewConfig access.

**Severity:** Medium. Same pattern as Camera.

---

### L3: Sidebar Holds Local Mutable State

**Location:** `ui/panels/sidebar/sidebar_state.py`

**Problem:** Several fields that arguably belong in AppState:

| Field | Current Location | Should Be |
|-------|------------------|-----------|
| `equation_input` | Sidebar local | AppState (linear system definition) |
| `equation_count` | Sidebar local | AppState |
| `selected_matrix_idx` | Sidebar local | Redundant with `selected_id`/`selected_type` |
| `operation_result` | Sidebar local | Transient OK, but inconsistent |
| `_cell_buffers` | Sidebar local | ImGui artifact, OK locally |

**Why it matters:** Linear system equations are domain data, not transient UI state. A CLI cannot access them.

**Severity:** Low-Medium. `equation_input` is the main concern.

---

### L4: Inspector Holds Comparison State

**Location:** `ui/inspectors/inspector_computed.py:21-22`

**Problem:** `self._compare_vector_id` is stored on Inspector instance. This is selection state that should be in AppState.

**Why it matters:** Minor, but violates single-source-of-truth principle.

**Severity:** Low.

---

### L5: Renderer Holds Display Flags

**Location:** `render/renderers/renderer.py:29-32`

**Problem:** Renderer has mutable flags:
```python
self.show_vector_labels = True
self.show_plane_visuals = True
self.show_vector_components = True
self.show_vector_spans = False
```

These are toggled by keyboard shortcuts in `app_handlers.py` (e.g., V key toggles `show_vector_components`).

**Why it matters:** Display settings split between ViewConfig and Renderer. No single place to query "what is visible?"

**Severity:** Low. Consolidation would help but not urgent.

---

## 1.2 Half-Integrated Systems

### H1: Ribbon UI Not Wired

**Location:** `ui/ribbon/` (11 files)

**Status:** Complete implementation exists:
- `RibbonTab` base class
- `RibbonGroup`, `RibbonButton` components
- `HomeTab`, `VectorsTab`, `MatricesTab`, `ImagesTab`, `ViewTab`
- Buttons already reference correct Actions (Undo, Redo, SetTheme, etc.)

**Current state:** `WorkspaceLayout` imports and uses `Toolbar`, not Ribbon.

**Problem:** Two toolbar implementations coexist. Ribbon is dead code.

**Severity:** High for code hygiene. Must either wire Ribbon or delete it.

---

### H2: Planes Infrastructure Unused

**Location:**
- `state/models/plane_model.py` - PlaneData class defined
- `state/app_state.py:43` - `planes: Tuple[PlaneData, ...] = ()` field exists
- `engine/scene_adapter.py:110-118` - Planes converted for renderer

**Missing:**
- No plane actions (AddPlane, DeletePlane, etc.)
- No plane reducers
- No plane UI
- No plane rendering (gizmos exist but not called)

**Problem:** Dead infrastructure. SceneAdapter builds plane list but nothing populates it.

**Severity:** Medium. Should be removed or completed.

---

### H3: SceneSolversMixin Orphaned

**Location:** `domain/scenes/solvers.py`

**Status:** Mixin class with three methods:
- `gaussian_elimination(A, b)` - Returns solution + steps
- `compute_null_space(A)` - SVD-based
- `compute_column_space(A)` - SVD-based

**Problem:** This is a mixin, but no class inherits from it. The sidebar has its own `_compute_null_space` and `_compute_column_space` that duplicate this logic.

**Severity:** Low. Duplication, not breakage.

---

### H4: domain/vectors/Vector3D Unused in State

**Location:** `domain/vectors/vector3d.py`

**Status:** Full `Vector3D` class with methods (normalize, scale, dot, cross, angle, project_onto, transform, etc.).

**Problem:** State uses `VectorData` (frozen dataclass). `Vector3D` is only used in domain layer for operations. Two vector representations exist.

**Severity:** Low. Intentional separation, but confusing.

---

### H5: domain/pipelines/ Empty

**Location:** `domain/pipelines/__init__.py`

**Status:** Empty directory with empty `__init__.py`.

**Problem:** Placeholder that was never implemented.

**Severity:** Low. Dead code.

---

## 1.3 Implicit Assumptions

### A1: SceneAdapter Assumes State Shape

**Location:** `engine/scene_adapter.py:82-143`

**Assumption:** Constructor directly accesses:
```python
state.vectors  # Must be iterable of VectorData
state.matrices  # Must be iterable of MatrixData
state.planes  # Must be iterable of PlaneData
state.selected_id
state.selected_type
state.preview_enabled
state.input_matrix
state.matrix_plot_enabled
```

**Risk:** If AppState fields are renamed or removed, SceneAdapter breaks silently.

---

### A2: Renderer Assumes SceneAdapter Interface

**Location:** `app/app_run.py:62-69`, `render/renderers/renderer.py:78-103`

**Assumption:** Renderer expects:
```python
scene_adapter.vectors  # List[RendererVector]
scene_adapter.selected_object  # RendererVector or dict
scene_adapter.show_matrix_plot  # bool
```

**Risk:** No interface/protocol defined. Duck typing only.

---

### A3: TempImageMatrix Inside Reducer

**Location:** `state/reducers/reducer_image_kernel.py:19-41`

**Problem:** Reducer defines a class inline:
```python
class TempImageMatrix:
    def __init__(self, data, name):
        ...
    def as_matrix(self):
        ...
    @property
    def is_grayscale(self):
        ...
```

This mimics `ImageMatrix` interface so `apply_kernel()` works. Fragile coupling.

**Risk:** If domain `ImageMatrix` interface changes, TempImageMatrix breaks.

---

### A4: UI Assumes dispatch May Be None

**Location:** Every UI file checks `if dispatch:` before calling.

**Pattern:**
```python
if imgui.button("Do Thing"):
    if dispatch:
        dispatch(SomeAction())
```

**Problem:** Defensive, but inconsistent. Some places check, some don't.

---

### A5: Keyboard Handler Mutates Multiple Systems

**Location:** `app/app_handlers.py:18-46`

**Problem:** Single `on_key` function directly mutates:
- `self.view_config.auto_rotate`
- `self.view_config.grid_mode`
- `self.view_config.show_cube_faces`
- `self.renderer.show_vector_components`
- `self.camera` (via reset)

No Actions dispatched. Changes bypass Redux entirely.

---

### A6: Color Cycling Split Between State and Sidebar

**Location:**
- `state/selectors/__init__.py:49-65` - `COLOR_PALETTE`, `get_next_color()`
- `ui/panels/sidebar/sidebar_state.py:42-52` - `self.color_palette`, `self.next_color_idx`

**Problem:** Two color palettes exist. Sidebar has its own cycling logic.

---

## 1.4 Dead/Legacy Code

### D1: App.selected Unused

**Location:** `app/app.py:111`

```python
self.selected = None
```

Never read or written elsewhere. Legacy from pre-Redux.

---

### D2: Sidebar.scene Always None

**Location:** `ui/panels/sidebar/sidebar_render.py:29`

```python
self.scene = None
```

Set to None every frame. Never used.

---

### D3: ribbon_tab Field in AppState

**Location:** `state/app_state.py:112`

```python
ribbon_tab: str = "File"
```

Field exists but Ribbon UI is not wired. No actions modify it.

---

---

# 2. MINIMAL PATCHES

## 2.1 Camera & ViewConfig (Accept Current State)

**Decision:** Do NOT move Camera/ViewConfig to Redux.

**Rationale:**
- High-frequency updates (mouse drag, scroll) would spam the reducer
- Camera state does not need undo/redo
- Performance cost outweighs architectural purity

**Patch:** Document this as an intentional exception. Add comment in `app.py`:

```python
# NOTE: Camera and ViewConfig are intentionally outside Redux.
# Reason: High-frequency updates (orbit, zoom) would flood the reducer.
# These are "viewport state" not "document state" - they don't need undo/redo.
# CLI access: Use App.camera and App.view_config directly when headless.
```

**Effort:** 5 minutes.

---

## 2.2 Wire Ribbon OR Delete It

**Decision:** Delete Ribbon code (keep Toolbar).

**Rationale:**
- Ribbon is complete but never tested in production
- Toolbar is working and wired
- Two implementations create maintenance burden
- Can re-add Ribbon later if needed

**Patch:** Delete entire `ui/ribbon/` directory. Remove `ribbon_tab` field from AppState.

**Files to delete:**
```
ui/ribbon/__init__.py
ui/ribbon/ribbon_button.py
ui/ribbon/ribbon_group.py
ui/ribbon/ribbon_tab.py
ui/ribbon/tabs/__init__.py
ui/ribbon/tabs/home_tab.py
ui/ribbon/tabs/images_tab.py
ui/ribbon/tabs/matrices_tab.py
ui/ribbon/tabs/vectors_tab.py
ui/ribbon/tabs/view_tab.py
```

**AppState change:** Remove line 112 (`ribbon_tab: str = "File"`).

**Effort:** 10 minutes.

---

## 2.3 Remove Planes Infrastructure

**Decision:** Remove planes until actually needed.

**Rationale:**
- No actions, reducers, or UI exist
- SceneAdapter builds empty list every frame
- Premature abstraction

**Patch:**

1. Delete `state/models/plane_model.py`
2. Remove from `state/models/__init__.py`: `from ... import PlaneData` and `'PlaneData'` from `__all__`
3. Remove from `state/app_state.py`: `planes: Tuple[PlaneData, ...] = ()` (line 43)
4. Remove from `state/__init__.py`: all `PlaneData` references
5. Remove from `engine/scene_adapter.py`: lines 110-118 (plane conversion), `_planes` field, `planes` property
6. Remove from `engine/picking_system.py`: `ray_intersect_plane` function (unused)

**Effort:** 20 minutes.

---

## 2.4 Consolidate Solver Logic

**Decision:** Delete `domain/scenes/solvers.py`, keep sidebar implementations.

**Rationale:**
- SceneSolversMixin is an orphaned mixin
- Sidebar already has working `_compute_null_space` and `_compute_column_space`
- No point maintaining two implementations

**Patch:** Delete `domain/scenes/solvers.py`.

**Effort:** 2 minutes.

---

## 2.5 Move Linear System Equations to AppState

**Decision:** Add `input_equations` field to AppState.

**Rationale:**
- Linear system definition is domain data, not transient UI state
- CLI will need to define systems
- Currently inaccessible through Store

**Patch:**

1. Add to `state/app_state.py`:
```python
input_equations: Tuple[Tuple[float, ...], ...] = (
    (1.0, 1.0, 1.0, 0.0),
    (2.0, -1.0, 0.0, 0.0),
    (0.0, 1.0, -1.0, 0.0),
)
input_equation_count: int = 3
```

2. Add actions to `state/actions/input_actions.py`:
```python
@dataclass(frozen=True)
class SetEquationCell:
    row: int
    col: int
    value: float

@dataclass(frozen=True)
class SetEquationCount:
    count: int
```

3. Add reducer cases to `reducer_inputs.py`

4. Update `sidebar_linear_systems_section.py` to read from `state.input_equations` and dispatch actions

5. Remove from `sidebar_state.py`: `equation_count`, `equation_input`

**Effort:** 45 minutes.

---

## 2.6 Remove App.selected

**Patch:** Delete line 111 in `app/app.py`:
```python
self.selected = None  # DELETE THIS LINE
```

**Effort:** 1 minute.

---

## 2.7 Remove Sidebar.scene Assignment

**Patch:** Delete line 29 in `ui/panels/sidebar/sidebar_render.py`:
```python
self.scene = None  # DELETE THIS LINE
```

**Effort:** 1 minute.

---

## 2.8 Remove Empty domain/pipelines/

**Patch:** Delete directory `domain/pipelines/` and its empty `__init__.py`.

**Effort:** 1 minute.

---

## 2.9 Extract TempImageMatrix to Adapter

**Decision:** Create proper adapter instead of inline class.

**Rationale:**
- Inline class in reducer is fragile
- Interface coupling should be explicit

**Patch:** Create `engine/image_adapter.py`:

```python
"""Adapter to make ImageData compatible with domain ImageMatrix interface."""

class ImageDataAdapter:
    """Wraps ImageData to provide ImageMatrix-compatible interface for domain functions."""

    def __init__(self, image_data):
        self.data = image_data.pixels
        self.name = image_data.name

    def as_matrix(self):
        if len(self.data.shape) == 2:
            return self.data
        return (0.299 * self.data[:, :, 0] +
                0.587 * self.data[:, :, 1] +
                0.114 * self.data[:, :, 2])

    @property
    def is_grayscale(self):
        return len(self.data.shape) == 2

    @property
    def height(self):
        return self.data.shape[0]

    @property
    def width(self):
        return self.data.shape[1]

    @property
    def history(self):
        return []
```

Then in `reducer_image_kernel.py`, replace inline TempImageMatrix with:
```python
from engine.image_adapter import ImageDataAdapter
...
adapter = ImageDataAdapter(state.current_image)
result = apply_kernel(adapter, action.kernel_name, normalize_output=True)
```

**Effort:** 20 minutes.

---

## 2.10 Unify Color Palette

**Decision:** Remove Sidebar's color palette, use only state selectors.

**Patch:**

1. In `sidebar_state.py`, remove:
```python
self.color_palette = [...]
self.next_color_idx = 0
```

2. In `sidebar_utils.py`, change `_get_next_color` to use state:
```python
def _get_next_color(self):
    if self._state is None:
        return (0.8, 0.2, 0.2)
    from state.selectors import get_next_color
    color, _ = get_next_color(self._state)
    return color
```

Note: This reads from state but doesn't update `next_color_index`. For full consistency, AddVector reducer should handle color cycling (it already does in `reducer_vectors.py:17-18`).

**Effort:** 10 minutes.

---

## 2.11 Add Defensive Protocol for SceneAdapter

**Decision:** Add type hints and docstrings, not runtime checks.

**Rationale:** Python protocols are documentation. Runtime checks would slow render loop.

**Patch:** Add to `engine/scene_adapter.py`:

```python
from typing import Protocol, List, Optional
import numpy as np

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
```

**Effort:** 10 minutes.

---

---

# 3. LOCKED INVARIANTS

These are the hard rules that must be enforced by convention and/or code.

## 3.1 State Ownership Invariants

### INV-1: AppState is the Single Source of Truth for Domain Data

**Rule:** All vectors, matrices, and images MUST live in AppState.

**Specifically:**
- `state.vectors` is the authoritative list of vectors
- `state.matrices` is the authoritative list of matrices
- `state.current_image` and `state.processed_image` are the authoritative images

**Violation examples:**
- Storing a vector in a UI component
- Caching matrix data in the renderer
- Keeping "backup" copies outside state

**Enforcement:** Code review. Search for `VectorData(` or `MatrixData(` outside reducers.

---

### INV-2: State Changes Only Through dispatch()

**Rule:** To modify AppState, you MUST call `store.dispatch(action)`.

**No exceptions** for domain data (vectors, matrices, images, selection).

**Allowed exceptions:**
- Camera (high-frequency viewport control)
- ViewConfig (high-frequency viewport control)
- ImGui widget buffers (transient text input)

**Enforcement:** Code review. Search for `replace(state,` outside `state/reducers/`.

---

### INV-3: Reducers Are Pure Functions

**Rule:** Reducers MUST NOT:
- Perform I/O (file reads, network)
- Mutate their input state
- Access global mutable state
- Call `dispatch()` themselves

**Allowed:**
- Call pure domain functions (e.g., `apply_kernel()`)
- Create new immutable objects
- Use numpy for computation

**Enforcement:** Code review. Check reducer files for `import os`, `open(`, `print(`.

---

### INV-4: Actions Are Immutable Descriptions

**Rule:** Actions MUST be frozen dataclasses describing "what happened."

Actions MUST NOT:
- Contain callbacks or functions
- Contain mutable objects (lists, dicts with mutable contents)
- Perform computation in `__init__`

**Enforcement:** All action classes must have `@dataclass(frozen=True)`.

---

## 3.2 Rendering Invariants

### INV-5: Renderer Reads, Never Writes State

**Rule:** The Renderer and all render/* code MUST NOT:
- Hold references to AppState
- Call `dispatch()`
- Mutate anything except OpenGL buffers

**Renderer receives:** SceneAdapter (read-only view of state)

**Enforcement:** render/ directory should not import from state/actions/.

---

### INV-6: SceneAdapter is Ephemeral

**Rule:** A new SceneAdapter MUST be created each frame from current state.

**SceneAdapter MUST NOT:**
- Be cached across frames
- Be stored on App or any long-lived object
- Mutate after construction

**Enforcement:** `build_scene_adapter(state)` is called in render loop, result is not stored.

---

### INV-7: Camera and ViewConfig Are Viewport State

**Rule:** Camera and ViewConfig are deliberately outside Redux.

**They represent:** How the user is looking at the scene (viewport), not what the scene contains (document).

**Implication:** Camera position and zoom are NOT undoable. This is intentional.

---

## 3.3 UI Invariants

### INV-8: UI Reads State, Dispatches Actions

**Rule:** UI components MUST:
- Read from `state` parameter (not cached copies)
- Modify domain data only by calling `dispatch(action)`

**UI components MAY:**
- Hold transient input buffers for ImGui (text fields)
- Track local UI state (expanded/collapsed, scroll position)

**Enforcement:** UI files should not contain `dataclasses.replace(state,`.

---

### INV-9: dispatch() is Always Available in Render Context

**Rule:** In `WorkspaceLayout.render()` and all child components, `dispatch` is always a valid callable.

**Current state:** Many components check `if dispatch:`. This is overly defensive.

**Going forward:** Assume `dispatch` is never None during normal rendering.

---

## 3.4 Domain Invariants

### INV-10: Domain Functions Are Pure

**Rule:** All functions in `domain/` MUST be pure:
- No state access
- No I/O
- Same inputs → same outputs

**Domain code MUST NOT:**
- Import from `state/`
- Import from `ui/`
- Import from `app/`

**Enforcement:** Check imports at top of domain files.

---

### INV-11: ImageMatrix is for Computation, ImageData is for Storage

**Rule:**
- `ImageData` (state/models) = immutable storage format in AppState
- `ImageMatrix` (domain/images) = mutable computation format for domain functions

**Conversion:**
- State → Domain: Use `ImageDataAdapter` or create `ImageMatrix` from `image_data.pixels`
- Domain → State: Use `ImageData.create(result.data, name)`

---

---

# 4. CLI READINESS CHECKLIST

## 4.1 API Surface a CLI Would Need

A headless CLI must be able to:

### Vector Operations
| Operation | Required Action | Status |
|-----------|-----------------|--------|
| Add vector | `AddVector(coords, color, label)` | EXISTS |
| Delete vector | `DeleteVector(id)` | EXISTS |
| Update vector | `UpdateVector(id, coords?, color?, label?, visible?)` | EXISTS |
| List vectors | `state.vectors` | EXISTS (read from state) |
| Select vector | `SelectVector(id)` | EXISTS |
| Clear all vectors | `ClearAllVectors()` | EXISTS |

### Matrix Operations
| Operation | Required Action | Status |
|-----------|-----------------|--------|
| Add matrix | `AddMatrix(values, label)` | EXISTS |
| Delete matrix | `DeleteMatrix(id)` | EXISTS |
| Update matrix cell | `UpdateMatrixCell(id, row, col, value)` | EXISTS |
| Apply to vector | `ApplyMatrixToSelected(matrix_id)` | EXISTS |
| Apply to all | `ApplyMatrixToAll(matrix_id)` | EXISTS |
| List matrices | `state.matrices` | EXISTS |

### Image Operations
| Operation | Required Action | Status |
|-----------|-----------------|--------|
| Load image | `LoadImage(path, max_size?)` | EXISTS |
| Create sample | `CreateSampleImage(pattern, size)` | EXISTS |
| Apply kernel | `ApplyKernel(kernel_name)` | EXISTS |
| Apply transform | `ApplyTransform(rotation, scale)` | EXISTS |
| Flip horizontal | `FlipImageHorizontal()` | EXISTS |
| Use result as input | `UseResultAsInput()` | EXISTS |
| Clear images | `ClearImage()` | EXISTS |
| Normalize | `NormalizeImage(mean?, std?)` | EXISTS |
| Get current image | `state.current_image` | EXISTS |
| Get processed image | `state.processed_image` | EXISTS |

### Linear Systems (after patch 2.5)
| Operation | Required Action | Status |
|-----------|-----------------|--------|
| Set equation cell | `SetEquationCell(row, col, value)` | NEEDS PATCH |
| Set equation count | `SetEquationCount(count)` | NEEDS PATCH |
| Solve system | (Use domain function directly) | EXISTS in domain |

### History
| Operation | Required Action | Status |
|-----------|-----------------|--------|
| Undo | `Undo()` | EXISTS |
| Redo | `Redo()` | EXISTS |

### Navigation (mostly irrelevant for CLI)
| Operation | Required Action | Status |
|-----------|-----------------|--------|
| Set active tab | `SetActiveTab(tab)` | EXISTS (GUI only) |
| Set theme | `SetTheme(theme)` | EXISTS (GUI only) |

---

## 4.2 What CLI Must Do Through Store

**MANDATORY via dispatch():**

1. Any operation that creates, modifies, or deletes:
   - Vectors
   - Matrices
   - Images

2. Any operation that should be undoable

3. Selection changes

**Example CLI flow:**
```python
# Correct
store.dispatch(AddVector(coords=(1,2,3), color=(1,0,0), label="v"))
store.dispatch(AddMatrix(values=((1,0,0),(0,1,0),(0,0,1)), label="I"))
store.dispatch(ApplyMatrixToAll(matrix_id=matrix_id))

# Wrong - bypasses state
scene.add_vector(...)  # NO - scene doesn't exist
state.vectors.append(...)  # NO - state is immutable
```

---

## 4.3 What CLI Must NOT Do

**FORBIDDEN:**

1. **Direct state mutation**
   ```python
   # WRONG
   state = store.get_state()
   state.vectors = state.vectors + (new_vec,)  # Frozen dataclass, will fail
   ```

2. **Calling UI functions**
   ```python
   # WRONG
   sidebar._render_vector_creation()  # UI code, requires imgui context
   ```

3. **Assuming ImGui context exists**
   ```python
   # WRONG
   imgui.button("Click")  # No window, will crash
   ```

4. **Modifying Camera/ViewConfig without App**
   ```python
   # WRONG (unless you have App instance)
   camera.orbit(10, 10)  # Camera is on App, not accessible headless
   ```

---

## 4.4 Recommended CLI Architecture

```
cli.py (entrypoint)
    │
    ├── Creates Store with initial state
    │
    ├── Dispatches actions based on CLI args
    │
    ├── Reads final state
    │
    └── Outputs result (JSON, image file, etc.)
```

**Minimal CLI bootstrap:**
```python
from state import Store, create_initial_state
from state.actions import AddVector, ApplyKernel, LoadImage

def main():
    store = Store(create_initial_state())

    # CLI operations via dispatch
    store.dispatch(LoadImage(path="input.png"))
    store.dispatch(ApplyKernel(kernel_name="sobel_x"))

    # Read result
    state = store.get_state()
    if state.processed_image:
        save_image(state.processed_image.pixels, "output.png")
```

**What CLI does NOT need:**
- App class
- Window/GLFW
- ImGui
- Renderer
- Camera
- ViewConfig

**What CLI DOES need:**
- Store
- Actions
- Reducers (implicitly, via Store)
- Domain functions (for direct computation if needed)

---

## 4.5 Pre-Flight Checklist Before Building CLI

- [ ] Patch 2.2 complete (Ribbon removed)
- [ ] Patch 2.3 complete (Planes removed)
- [ ] Patch 2.5 complete (Linear equations in state)
- [ ] Patch 2.9 complete (ImageDataAdapter extracted)
- [ ] All invariants documented in CONTRIBUTING.md or similar
- [ ] Example CLI script tested with:
  - [ ] Vector creation
  - [ ] Matrix application
  - [ ] Image load + kernel
  - [ ] Undo/redo
- [ ] No UI imports in cli.py

---

# 5. SUMMARY

## Patches by Priority

| Priority | Patch | Effort | Risk |
|----------|-------|--------|------|
| P0 | 2.2 Wire or Delete Ribbon | 10 min | Low |
| P0 | 2.6 Remove App.selected | 1 min | None |
| P0 | 2.7 Remove Sidebar.scene | 1 min | None |
| P0 | 2.8 Remove empty pipelines/ | 1 min | None |
| P1 | 2.3 Remove Planes | 20 min | Low |
| P1 | 2.4 Remove SceneSolversMixin | 2 min | None |
| P1 | 2.9 Extract ImageDataAdapter | 20 min | Low |
| P2 | 2.5 Move equations to state | 45 min | Medium |
| P2 | 2.10 Unify color palette | 10 min | Low |
| P3 | 2.1 Document Camera/ViewConfig | 5 min | None |
| P3 | 2.11 Add SceneAdapter protocol | 10 min | None |

**Total effort:** ~2 hours

## Invariants to Enforce

| ID | Rule | Enforcement |
|----|------|-------------|
| INV-1 | AppState owns domain data | Code review |
| INV-2 | Changes via dispatch() only | Code review |
| INV-3 | Reducers are pure | Import checks |
| INV-4 | Actions are frozen | Decorator check |
| INV-5 | Renderer never writes state | Import checks |
| INV-6 | SceneAdapter is ephemeral | Code pattern |
| INV-7 | Camera/ViewConfig outside Redux | Documented |
| INV-8 | UI dispatches, doesn't mutate | Code review |
| INV-9 | dispatch() always available | Convention |
| INV-10 | Domain functions are pure | Import checks |
| INV-11 | ImageMatrix vs ImageData | Convention |

## CLI Readiness

- **Sufficient actions exist** for vectors, matrices, images
- **Missing:** Equation input actions (Patch 2.5)
- **Architecture ready:** Store can be used headlessly
- **No blockers** after patches complete

---

END OF DOCUMENT
