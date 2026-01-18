# CVLA SPECIFICATION v1.0

> **Complete Visual Linear Algebra** - A research-grade visualization engine for understanding
> how neural networks learn through the lens of linear algebra.

**Document Purpose:** This specification provides everything needed for an AI or developer to implement CVLA from scratch.

---

## TABLE OF CONTENTS

1. [Vision & Goals](#vision--goals)
2. [Architecture Overview](#architecture-overview)
3. [Quality Standards](#quality-standards)
4. [Architecture Invariants](#architecture-invariants)
5. [Data Models](#data-models)
6. [Operation Registry System](#operation-registry-system)
7. [State Management](#state-management)
8. [Feature Checklist (100+ Features)](#feature-checklist)
9. [UI Layout Specification](#ui-layout-specification)
10. [Rendering System](#rendering-system)
11. [File Structure](#file-structure)
12. [Testing Requirements](#testing-requirements)
13. [Implementation Order](#implementation-order)
14. [API Reference](#api-reference)

---

## VISION & GOALS

### Problem Statement
Linear algebra feels abstract. Matrices are "just numbers" instead of *spaces*. When we apply a matrix transformation, we're changing the entire coordinate system - but this is invisible in textbooks.

### Solution
CVLA makes the invisible visible. Every matrix operation produces a step-by-step visualization showing:
- **WHAT** happens (the math)
- **WHY** it works (the intuition)
- **HOW** it looks (the geometry)

### End Goal
Visualize how a neural network learns edges and shapes from images:
- Load image datasets
- See images decomposed into matrices
- Watch convolutions slide across pixels
- Observe learned weights emerge
- Eventually: implement and visualize YOLO algorithm learning

### Target Users
- Students learning linear algebra
- ML practitioners wanting intuition
- Researchers debugging neural networks
- Educators creating visual explanations

---

## ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│  ┌─────────┐  ┌──────────────────┐  ┌────────────────────────┐  │
│  │  Input  │  │   3D Viewport    │  │   Operations Panel     │  │
│  │  Panel  │  │   (OpenGL)       │  │   + Step Display       │  │
│  └────┬────┘  └────────┬─────────┘  └───────────┬────────────┘  │
│       │                │                        │               │
│       └────────────────┼────────────────────────┘               │
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    ACTION DISPATCH                       │    │
│  │         (User clicks → Action objects created)          │    │
│  └─────────────────────────┬───────────────────────────────┘    │
└────────────────────────────┼────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        STATE LAYER                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   AppState   │◄───│   Reducers   │◄───│   Actions        │   │
│  │  (Immutable) │    │   (Pure)     │    │   (Immutable)    │   │
│  └──────┬───────┘    └──────────────┘    └──────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     SELECTORS                             │   │
│  │         (Derived state, memoized queries)                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DOMAIN LAYER                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 OPERATION REGISTRY                        │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐            │   │
│  │  │ VectorOps  │ │ MatrixOps  │ │ EigenOps   │  ...       │   │
│  │  └────────────┘ └────────────┘ └────────────┘            │   │
│  │                                                          │   │
│  │  Every operation implements:                             │   │
│  │  - validate(*tensors) → Result                           │   │
│  │  - compute(*tensors) → TensorData                        │   │
│  │  - steps(*tensors) → Tuple[Step, ...]                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RENDER LAYER                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ SceneAdapter │───▶│   Renderer   │───▶│   OpenGL/GPU     │   │
│  │ (State→Draw) │    │   (ModernGL) │    │                  │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow
1. **User Action** → UI captures input (click, type, etc.)
2. **Dispatch** → Action object created and sent to store
3. **Reduce** → Reducer computes new state from (old_state, action)
4. **Select** → Selectors derive view-specific data
5. **Render** → SceneAdapter converts state to draw calls
6. **Display** → OpenGL renders to screen

---

## QUALITY STANDARDS

### 1. CORRECTNESS (Non-Negotiable)
- Matches textbook definitions (Strang, Trefethen, Golub & Van Loan)
- Deterministic: same inputs → same outputs
- Numerically stable where expected
- **No silent failure** - if operation fails, UI explains why
- **No hidden normalization** - all scaling is explicit

### 2. STEP-TRACEABILITY (The Soul of CVLA)
Every operation MUST produce:
```python
@dataclass(frozen=True)
class Step:
    step_index: int
    math_expression: str      # LaTeX: r"\vec{v} \cdot \vec{w}"
    description: str          # Plain text explanation
    numeric_values: Dict[str, Any]
    geometric_change: GeometricDelta
    reversible: bool = True
    substeps: Tuple[Step, ...] = ()
```
**Rule: If it can't be stepped → it doesn't ship.**

### 3. VISUAL FIDELITY (3Blue1Brown Clarity)
- Vectors originate from origin unless explicitly translated
- Basis vectors visibly deform under transformations
- Subspaces are visually distinct (planes, lines, volumes)
- Color logic for intersections (RGB overlap, alpha layering)
- **If visual contradicts math → visual is wrong**

### 4. PERFORMANCE (Hard Realtime)
- **Target: 60 FPS** on mid-range hardware
- ≤ 2ms CPU per frame
- No math in render loop
- All computation triggered by actions
- Cached results reused
- **If FPS drops → feature is rolled back**

### 5. ARCHITECTURE (Compiler-Like Separation)
```
UI → Action → Operation → Steps → State → Renderer
```
- UI never computes math
- Reducers never render
- Renderers never mutate state
- Operations are pure functions

---

## ARCHITECTURE INVARIANTS

These rules MUST hold. Violation = bug.

| # | Invariant | Enforcement |
|---|-----------|-------------|
| 1 | AppState is the SINGLE source of truth | All reads via selectors |
| 2 | AppState is IMMUTABLE | Frozen dataclass, replace() only |
| 3 | All mutations via dispatch(Action) | No direct state modification |
| 4 | Reducers are PURE functions | state + action → new_state |
| 5 | UI reads state, dispatches actions | Never computes math |
| 6 | Renderer reads state only | Never mutates, never computes |
| 7 | Operations produce Steps | No operation without step list |
| 8 | Camera/ViewConfig live outside Redux | Performance exception, documented |
| 9 | No orphan state | Every field has owner and purpose |
| 10 | All operations registered | No ad-hoc math functions |
| 11 | Step index always valid | 0 ≤ index < len(steps) |

---

## DATA MODELS

### TensorDType (Data classification)
```python
class TensorDType(Enum):
    NUMERIC = "numeric"          # vectors, matrices
    IMAGE_RGB = "image_rgb"      # HxWx3 or HxWx4
    IMAGE_GRAYSCALE = "image_grayscale"  # HxW or HxWx1
```

### TensorData (Unified, immutable)
```python
@dataclass(frozen=True)
class TensorData:
    id: str
    data: Tuple[Union[float, Tuple], ...]  # nested tuples for N-D
    shape: Tuple[int, ...]
    dtype: TensorDType
    label: str
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    visible: bool = True
    history: Tuple[str, ...] = ()

    # ---- classification ----
    @property
    def rank(self) -> int: ...
    @property
    def tensor_type(self) -> str:  # 'vector' | 'matrix' | 'image'
    @property
    def is_vector(self) -> bool: ...
    @property
    def is_matrix(self) -> bool: ...
    @property
    def is_image(self) -> bool: ...

    # ---- factories ----
    @staticmethod
    def create_vector(coords: Tuple[float, ...], label: str,
                      color: Tuple[float, float, float] = (0.8, 0.8, 0.8)) -> TensorData
    @staticmethod
    def create_matrix(values: Tuple[Tuple[float, ...], ...], label: str,
                      color: Tuple[float, float, float] = (0.8, 0.8, 0.8)) -> TensorData
    @staticmethod
    def create_image(pixels: np.ndarray, name: str,
                     history: Tuple[str, ...] = ()) -> TensorData

    # ---- helpers ----
    def to_numpy(self) -> np.ndarray
    def with_history(self, operation: str) -> TensorData
    def with_data(self, new_data: Tuple, new_shape: Optional[Tuple[int, ...]] = None) -> TensorData
    def with_label(self, label: str) -> TensorData
    def with_color(self, color: Tuple[float, float, float]) -> TensorData
    def with_visible(self, visible: bool) -> TensorData

    # ---- numeric conveniences ----
    @property
    def coords(self) -> Tuple[float, ...]                  # vectors only
    @property
    def values(self) -> Tuple[Tuple[float, ...], ...]      # matrices only
    @property
    def rows(self) -> int
    @property
    def cols(self) -> int
```

### EducationalStep (Pipeline narration)
```python
@dataclass(frozen=True)
class EducationalStep:
    id: str
    title: str
    explanation: str
    operation: str
    input_data: Optional[ImageData] = None
    output_data: Optional[ImageData] = None
    kernel_name: Optional[str] = None
    kernel_values: Optional[Tuple[Tuple[float, ...], ...]] = None
    transform_matrix: Optional[Tuple[Tuple[float, ...], ...]] = None
    kernel_position: Optional[Tuple[int, int]] = None
```

### OperationRecord (History log)
```python
@dataclass(frozen=True)
class OperationRecord:
    id: str
    operation_name: str
    parameters: Tuple[Tuple[str, str], ...]
    target_ids: Tuple[str, ...]
    result_ids: Tuple[str, ...]
    timestamp: float
    description: str
```

### Legacy Vector/Matrix Models
`VectorData` and `MatrixData` remain for backward compatibility with
pre-unified UI code. SceneAdapter merges them with `TensorData` so the renderer
can treat all math objects uniformly.

### Result Helper (optional)
When an operation needs explicit success/failure, use a light-weight result:
```python
@dataclass(frozen=True)
class Ok(Generic[T]): ...
@dataclass(frozen=True)
class Err(Generic[E]): ...
Result = Union[Ok[T], Err[E]]
```

---

## OPERATION REGISTRY SYSTEM

*Status:* Not yet implemented in code; current math lives in `domain/vectors`, `domain/transforms`, and `domain/images`. This interface is the target for unifying operations and step traces.

### OperationSpec (Interface)
```python
TensorKind = Literal["vector", "matrix", "image"]

class OperationSpec(ABC):
    # === METADATA ===
    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def category(self) -> str: ...  # 'vector' | 'matrix' | 'transform' | 'image'

    @property
    @abstractmethod
    def inputs(self) -> Tuple[TensorKind, ...]: ...

    @property
    @abstractmethod
    def outputs(self) -> Tuple[TensorKind, ...]: ...

    @property
    def assumptions(self) -> Tuple[str, ...]:
        """Assumptions: ('same_dimension', 'square_matrix')"""
        return ()

    # === EDUCATIONAL ===
    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def intuition(self) -> str: ...

    @property
    def failure_modes(self) -> Tuple[str, ...]:
        return ()

    @property
    @abstractmethod
    def geometric_meaning(self) -> str: ...

    # === CORE METHODS ===
    @abstractmethod
    def validate(self, *tensors: TensorData) -> Result[None, str]: ...

    @abstractmethod
    def compute(self, *tensors: TensorData) -> TensorData: ...

    @abstractmethod
    def steps(self, *tensors: TensorData) -> Tuple[EducationalStep, ...]: ...
```

### Operation Registry
```python
class OperationRegistry:
    def register(self, operation: OperationSpec) -> None
    def get(self, operation_id: str) -> Optional[OperationSpec]
    def list_all(self) -> List[OperationSpec]
    def list_by_category(self, category: str) -> List[OperationSpec]
    def categories(self) -> List[str]
    def execute(self, operation_id: str, *tensors: TensorData) -> Result[Tuple[TensorData, Tuple[Step, ...]], str]

# Global registry instance
registry = OperationRegistry()

# Decorator for registration
@register_operation
class MyOperation(OperationSpec):
    ...
```

### Wiring into UI + history (next up)
- Operations panel lists registry entries by `category` and renders `description` + `intuition`.
- Executing an operation follows: `validate` → `compute` → `steps` (all pure); failures return `Err` and set a user-facing message, with **no** state mutation.
- Success path:
  - Store result as `operation_preview_tensor` when preview is enabled; commit to tensors/images only on confirm.
  - Append `OperationRecord` (operation id/name, parameters, target_ids, result_ids) to `operation_history`.
  - Replace `pipeline_steps` with the returned `EducationalStep` list and reset `pipeline_step_index = 0`.
- If `steps()` is empty, insert a single step that explains why the operation is non-stepped (e.g., constant-time ops).

### Step trace → timeline mapping (next up)
- Timeline titles come from `EducationalStep.title`; the long description uses `EducationalStep.explanation`.
- `EducationalStep.math_expression` is rendered in the right panel (LaTeX) and mirrored in the timeline tooltip.
- `EducationalStep.input_data`/`output_data` (when provided) are used by `SceneAdapter` to render before/after previews without recomputing math in the renderer.
- `OperationRecord` anchors the timeline group; steps are treated as children of the last operation until a new record is appended.

### Image & convolution step generation (next up)
- Convolution operations emit a **step per kernel position** (row-major), plus a final "assembled output" step.
- Each step fills:
  - `kernel_position` (x, y)
  - `kernel_values` (current kernel)
  - `input_data` (the local image patch as `ImageData`)
  - `output_data` (either the current output pixel or the partial output image)
- For performance, steps may store **downsampled** previews while the final compute writes full-resolution output.

---

## FEATURE CHECKLIST

### Vector Operations (12 features)
| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| V01 | Create vector (text input) | P0 | Parse "1, 2, 3" or "[1, 2, 3]" |
| V02 | Create vector (grid input) | P1 | Visual grid with sliders |
| V03 | Create vector (click in viewport) | P2 | Click to set endpoint |
| V04 | Vector addition | P0 | v + w with parallelogram visual |
| V05 | Vector subtraction | P0 | v - w with visual |
| V06 | Scalar multiplication | P0 | cv with scaling visual |
| V07 | Dot product | P0 | v·w with projection visual |
| V08 | Cross product | P0 | v×w with perpendicular visual (3D) |
| V09 | Magnitude | P0 | \|v\| with unit circle reference |
| V10 | Normalize | P0 | v/\|v\| showing unit vector |
| V11 | Projection | P0 | proj_w(v) with shadow visual |
| V12 | Angle between vectors | P1 | arccos(v·w/\|v\|\|w\|) |

### Matrix Operations (15 features)
| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| M01 | Create matrix (text input) | P0 | Parse "[[1,0],[0,1]]" |
| M02 | Create matrix (grid input) | P0 | Editable grid |
| M03 | Matrix-vector multiplication | P0 | Ax with basis change visual |
| M04 | Matrix-matrix multiplication | P0 | AB with composition visual |
| M05 | Transpose | P0 | A^T |
| M06 | Determinant | P0 | det(A) with volume visual |
| M07 | Inverse | P0 | A^(-1) with undo transformation visual |
| M08 | Trace | P1 | tr(A) = Σa_ii |
| M09 | Rank | P0 | Via SVD or RREF |
| M10 | Matrix addition | P1 | A + B |
| M11 | Scalar multiplication | P1 | cA |
| M12 | Power | P2 | A^n |
| M13 | Frobenius norm | P2 | \|\|A\|\|_F |
| M14 | Matrix equation Ax=b | P0 | Solve with steps |
| M15 | Least squares | P1 | A^T A x = A^T b |

### Linear Transformations (10 features)
| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| T01 | Apply matrix to vector (animated) | P0 | Show vector moving |
| T02 | Apply matrix to grid | P0 | Show grid deformation |
| T03 | Show basis change | P0 | i,j,k → Ai, Aj, Ak |
| T04 | 2D rotation matrix | P0 | R(θ) with angle input |
| T05 | 3D rotation (axis-angle) | P1 | Rodrigues' formula |
| T06 | Scaling matrix | P0 | diag(s_x, s_y, s_z) |
| T07 | Shear matrix | P1 | Show parallelogram |
| T08 | Reflection matrix | P1 | Householder |
| T09 | Projection matrix | P1 | P onto subspace |
| T10 | Composition visualization | P1 | AB as "B then A" |

### Eigenvalue/Eigenvector (8 features)
| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| E01 | Compute eigenvalues | P0 | det(A-λI)=0 |
| E02 | Compute eigenvectors | P0 | (A-λI)v=0 |
| E03 | Visualize eigenvectors | P0 | Show invariant directions |
| E04 | Eigendecomposition A=VΛV^(-1) | P1 | Full decomposition |
| E05 | Power iteration | P1 | Educational algorithm |
| E06 | Spectral radius | P2 | max(\|λ\|) |
| E07 | Eigenvalue interpretation | P0 | Stretch/compress/rotate |
| E08 | Complex eigenvalues | P2 | Rotation interpretation |

### Linear Systems (10 features)
| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| S01 | Gaussian elimination (stepped) | P0 | Row operations one at a time |
| S02 | Back substitution (stepped) | P0 | Solve from bottom up |
| S03 | LU decomposition | P0 | A = LU |
| S04 | Forward substitution | P1 | Ly = b |
| S05 | Gauss-Jordan elimination | P1 | Full RREF |
| S06 | Pivoting visualization | P1 | Show row swaps |
| S07 | Solution classification | P0 | Unique/infinite/none |
| S08 | Parametric solutions | P2 | Free variables |
| S09 | Augmented matrix [A\|b] | P0 | Visual representation |
| S10 | Condition number | P2 | Numerical stability |

### Subspaces (10 features)
| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| SS01 | Null space N(A) | P0 | Vectors A maps to 0 |
| SS02 | Column space C(A) | P0 | Image of A |
| SS03 | Row space R(A) | P1 | = C(A^T) |
| SS04 | Left null space N(A^T) | P1 | = C(A)^⊥ |
| SS05 | Visualize as plane/line | P0 | Draw subspace |
| SS06 | Orthogonal complement | P1 | S^⊥ |
| SS07 | Gram-Schmidt | P0 | Orthonormalize |
| SS08 | QR decomposition | P1 | A = QR |
| SS09 | Fundamental theorem visual | P2 | Four subspaces diagram |
| SS10 | Span visualization | P0 | span{v1, v2, ...} |

### Decompositions (8 features)
| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| D01 | SVD: A = UΣV^T | P0 | Full SVD with visuals |
| D02 | Truncated SVD | P1 | Low-rank approximation |
| D03 | SVD image compression | P1 | Visual demo |
| D04 | LU decomposition | P0 | A = LU |
| D05 | LU with pivoting | P1 | PA = LU |
| D06 | Cholesky | P2 | A = LL^T for SPD |
| D07 | QR decomposition | P1 | A = QR |
| D08 | Schur decomposition | P2 | A = QTQ^T |

### Image Processing (12 features)
| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| I01 | Load image as matrix | P0 | Grayscale or RGB channels |
| I02 | Display image as height field | P1 | 3D surface from pixels |
| I03 | 2D convolution (stepped) | P0 | Show kernel sliding |
| I04 | Sobel edge detection | P0 | Gradient magnitude |
| I05 | Gaussian blur | P1 | Smoothing kernel |
| I06 | Sharpen | P1 | Laplacian kernel |
| I07 | Custom kernel input | P1 | User-defined filter |
| I08 | Show kernel weights | P0 | Visual kernel display |
| I09 | Convolution as matrix mult | P2 | Toeplitz matrix view |
| I10 | Pooling operations | P2 | Max/avg pool |
| I11 | Image normalization | P1 | Mean/std normalization |
| I12 | Channel separation | P1 | R, G, B individual |

### UI Components (15 features)
| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| U01 | Left panel: Input creation | P0 | Text/grid input |
| U02 | Right panel: Operations | P0 | Operation buttons |
| U03 | Center: 3D viewport | P0 | OpenGL rendering |
| U04 | Bottom: Timeline | P0 | Step navigation |
| U05 | Step forward button | P0 | Next step |
| U06 | Step backward button | P0 | Previous step |
| U07 | Play/pause steps | P1 | Auto-advance |
| U08 | Speed control | P1 | Animation speed |
| U09 | Undo/redo | P0 | History navigation |
| U10 | Tensor list | P0 | Show all vectors/matrices |
| U11 | Selection highlight | P0 | Visual selection |
| U12 | Property inspector | P1 | Edit selected object |
| U13 | Dark/light theme | P2 | Theme toggle |
| U14 | Keyboard shortcuts | P1 | Ctrl+Z, etc. |
| U15 | Export steps to LaTeX | P2 | Generate document |

### Rendering Features (10 features)
| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| R01 | 3D vector rendering | P0 | Arrow with head |
| R02 | Vector labels | P0 | Text at tip |
| R03 | Grid rendering | P0 | Coordinate grid |
| R04 | Axis rendering | P0 | X, Y, Z axes |
| R05 | Grid deformation | P0 | Animated transform |
| R06 | Matrix heatmap | P1 | Color-coded values |
| R07 | Subspace plane | P1 | Transparent plane |
| R08 | Camera controls | P0 | Orbit, pan, zoom |
| R09 | Animation system | P0 | Smooth transitions |
| R10 | Anti-aliasing | P1 | MSAA |

---

## UI LAYOUT SPECIFICATION

```
+------------------------------------------------------------------+
|                        TOOLBAR (40px)                            |
|  [CVLA] [Undo] [Redo] [Export]      [Theme] [Settings]            |
+------------------------------------------------------------------+
| MODE |         LEFT: INPUT               |   RIGHT: OPERATIONS   |
| RAIL |         (Creation)                |   (Math/Reasoning)    |
| (60) |                                   |                       |
|      | [Text] [File] [Grid]              | Selected: v1          |
| [V]  | +-----------------------+         | Type: Vector          |
| [M]  | | 1, 2, 3               |         | Shape: (3,)           |
| [I]  | +-----------------------+         +-------------------+   |
|      | [Add Vector]                      | OPERATIONS        |   |
|      |                                   | [Normalize]       |   |
|      | TENSORS                           | [Scale] [____]    |   |
|      | > v1 (1,2,3)                      | [Dot] [Cross]     |   |
|      | > v2 (4,5,6)                      +-------------------+   |
|      | > M1 (3x3)                        | PREVIEW           |   |
|      |                                   | Before | After    |   |
|      |     3D VIEWPORT (CENTER)          |        |          |   |
|      |                                   +-------------------+   |
|      |                                   | STEPS             |   |
|      |                                   | 1. a·b = Σaᵢbᵢ    |   |
|      |                                   | 2. = 4+10+18      |   |
|      |                                   | 3. = 32           |   |
+------+-----------------------------------+-------------------+---+
|                    TIMELINE (100px)                              |
|  [|<] [<] Step 2/3 [>] [>|]    [====|=============]              |
|  "Computing dot product: 1*4 + 2*5 + 3*6"                        |
+------------------------------------------------------------------+
```

### Panel Specifications

**Toolbar (40px tall)**
- [Undo], [Redo], and [Export] (LaTeX) live here for global access
- Theme + settings toggles remain right-aligned

**Mode Rail (60px wide)**
- Vertical icons for mode switching
- [V] Vector mode
- [M] Matrix mode
- [I] Image mode
- Active mode highlighted

**Left Panel: Input (250px min)**
- Tabbed input methods: Text, File, Grid
- Text input with validation
- Tensor list with selection
- Add/delete buttons

**Center: Viewport (flexible)**
- OpenGL 3D rendering
- Mouse controls: orbit, pan, zoom
- Displays all visible tensors
- Grid and axes overlay

**Viewport Controls (overlay or inspector section)**
- Grid fade: toggle + start/end sliders (`view_grid_fade`, `view_grid_fade_start`, `view_grid_fade_end`)
- Depth ordering: grid `behind` scene vs `overlay` on top (`view_grid_depth_order`)
- Axes remain visible regardless of grid depth order

**Right Panel: Operations (300px)**
- Shows selected tensor info
- Available operations for selection
- Step-by-step display with LaTeX
- Preview before/after

**Bottom: Timeline (100px)**
- Step navigation: first, prev, play/pause, next, last
- Scrubber bar for direct seeking
- Current step description
- Total steps display

---

## RENDERING SYSTEM

**Pipeline**
- Store state → `SceneAdapter` (`engine/scene_adapter.py`) → renderer-friendly scene objects.
- `Renderer` (`render/renderers/renderer.py`) caches view-projection, clears with gradient, chooses environment (planar vs cubic grid), draws images, math visuals, vectors, and selection highlights.
- `Gizmos` provide reusable GL primitives (grids, axes, basis transforms, spans, point clouds) and are the only place that touches ModernGL buffers.
- `LabelRenderer` overlays text labels for vectors/axes.
- `ViewConfig` controls render presets and sits outside Redux alongside the camera to avoid high-frequency reducer traffic.

**Key behaviors**
- Grid modes: `plane` (xy/xz/yz) and `cube`, both obey right/left-handed axis mapping and perspective/orthographic view modes.
- Grid depth ordering: when `view_grid_depth_order == 'overlay'`, draw grid after scene with depth test off; otherwise draw before scene with depth test on.
- Grid fade: when `view_grid_fade` is enabled, fade grid alpha based on camera distance between `view_grid_fade_start` and `view_grid_fade_end`.
- Caching: view-projection matrix cached until camera/view config changes; image plane batches cached by `(resolution, render_mode, color_mode)`.
- Image rendering: plane or height-field; optional grid overlay; uses `engine.image_adapter` downsampling before upload.
- Linear algebra overlays: basis transform previews, vector span parallelograms, parallelepiped for 3 vectors, optional matrix 3D point plot.
- Safety defaults: depth test + alpha blend on; back-face culling off so grids/images stay visible when orbiting.
- Selection: ring/highlight when `scene.selection_type == 'vector'`.
- Tunables: `vector_scale`, plane visuals toggle, cube face colors, cubic grid density, auto-rotate, label density.

**Performance rules**
- Renderer never mutates state; all math happens before render.
- Keep per-frame CPU under 2 ms; reuse gizmo buffers and cached projections.
- Heavy previews (matrix plots, image grids) must be explicitly toggled to avoid frame drops.

---

## FILE STRUCTURE

```
CVLA/
├── main.py
├── README.md
├── requirements.txt
├── imgui.ini
├── docs/
│   └── CVLA_1_spec.md
├── exports/                 # Generated LaTeX exports (optional)
├── app/                      # Window + ImGui bootstrap and handlers
│   ├── app.py
│   ├── app_run.py
│   ├── app_handlers.py
│   ├── app_style.py
│   └── app_logging.py
├── domain/                   # Pure math helpers (no UI side effects)
│   ├── vectors/              # vector_ops.py, vector3d.py
│   ├── transforms/           # affine_matrices.py, affine_transform.py, affine_helpers.py, transforms.py
│   └── images/               # image.py, image_matrix.py, image_samples.py
├── engine/                   # State → render bridge + runtime helpers
│   ├── scene_adapter.py
│   ├── picking_system.py
│   ├── history_manager.py
│   ├── execution_loop.py
│   └── image_adapter.py
├── render/                   # ModernGL layer
│   ├── cameras/              # camera.py + core/projection/controls modules
│   ├── viewconfigs/          # viewconfig.py and helpers (core, axis, cubic, grid_basis)
│   ├── gizmos/               # grid, vector visuals, draw_points, volume helpers
│   ├── renderers/            # renderer.py, renderer_vectors.py, renderer_linear_algebra.py, renderer_image.py, labels/
│   └── shaders/              # gizmo_programs.py
├── state/                    # Redux-style store
│   ├── app_state.py
│   ├── store.py
│   ├── actions/              # vectors, matrices, tensors, images, navigation, pipeline, history, input
│   ├── reducers/             # reducer_vectors.py, reducer_matrices.py, reducer_images.py, reducer_pipeline.py, reducer_navigation.py, reducer_input_panel.py, ...
│   ├── selectors/            # tensor_selectors.py, ...
│   └── models/               # tensor_model.py, vector_model.py, matrix_model.py, image_model.py, educational_step.py, operation_record.py, pipeline_models.py, tensor_compat.py
├── ui/                       # ImGui interface
│   ├── layout/workspace.py
│   ├── inspectors/
│   ├── panels/
│   │   ├── input_panel/
│   │   ├── operations_panel/
│   │   ├── timeline/
│   │   ├── images/
│   │   ├── sidebar/
│   │   ├── mode_selector/
│   │   └── tool_palette/
│   ├── toolbars/
│   ├── themes/
│   └── utils/
├── samples/                  # Demo assets and fixtures
└── state/__init__.py, ui/__init__.py, domain/__init__.py, render/__init__.py
```

---

## STATE MANAGEMENT

### AppState (Single Source of Truth)
`state/app_state.py` is canonical. Camera and `ViewConfig` live outside Redux for performance. High-level layout:

```python
@dataclass(frozen=True)
class AppState:
    # Scene entities (legacy + unified)
    vectors: Tuple[VectorData, ...] = ()
    matrices: Tuple[MatrixData, ...] = ()
    tensors: Tuple[TensorData, ...] = ()
    selected_tensor_id: Optional[str] = None

    # Input panel (text/file/grid + controlled inputs)
    active_input_method: str = "text"
    input_text_content: str = ""
    input_text_parsed_type: str = ""
    input_file_path: str = ""
    input_grid_rows: int = 3
    input_grid_cols: int = 3
    input_grid_cells: Tuple[Tuple[float, ...], ...] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    input_vector_coords: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    input_vector_label: str = ""
    input_vector_color: Tuple[float, float, float] = (0.8, 0.2, 0.2)
    input_matrix: Tuple[Tuple[float, ...], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    input_matrix_label: str = "A"
    input_matrix_rows: int = 3
    input_matrix_cols: int = 3
    input_equations: Tuple[Tuple[float, ...], ...] = (
        (1.0, 1.0, 1.0, 0.0),
        (2.0, -1.0, 0.0, 0.0),
        (0.0, 1.0, -1.0, 0.0),
    )
    input_equation_count: int = 3
    input_image_path: str = ""
    input_sample_pattern: str = "checkerboard"
    input_sample_size: int = 32
    input_transform_rotation: float = 0.0
    input_transform_scale: float = 1.0
    input_image_normalize_mean: float = 0.0
    input_image_normalize_std: float = 1.0
    active_image_tab: str = "raw"
    input_expression: str = ""
    input_expression_type: str = ""
    input_expression_error: str = ""
    input_matrix_preview_vectors: Tuple[Tuple[float, ...], ...] = ()

    # Operations + preview + history
    pending_operation: Optional[str] = None
    pending_operation_params: Tuple[Tuple[str, str], ...] = ()
    operation_preview_tensor: Optional[TensorData] = None
    show_operation_preview: bool = True
    operation_history: Tuple[OperationRecord, ...] = ()

    # Selection (legacy UI)
    selected_id: Optional[str] = None
    selected_type: Optional[str] = None  # 'vector', 'matrix', 'plane', 'image'

    # Image / vision state
    current_image: Optional[ImageData] = None
    processed_image: Optional[ImageData] = None
    selected_kernel: str = "sobel_x"
    image_status: str = ""
    image_status_level: str = "info"
    selected_pixel: Optional[Tuple[int, int]] = None
    image_render_mode: str = "plane"   # 'plane' | 'height-field'
    image_render_scale: float = 1.0
    image_color_mode: str = "rgb"
    image_auto_fit: bool = True
    show_image_grid_overlay: bool = False
    image_downsample_enabled: bool = False
    image_preview_resolution: int = 128
    image_max_resolution: int = 512
    current_image_stats: Optional[Tuple[float, float, float, float]] = None
    processed_image_stats: Optional[Tuple[float, float, float, float]] = None
    current_image_preview: Optional[Tuple[Tuple[float, ...], ...]] = None
    processed_image_preview: Optional[Tuple[Tuple[float, ...], ...]] = None
    selected_kernel_matrix: Optional[Tuple[Tuple[float, ...], ...]] = None

    # Educational pipeline
    pipeline_steps: Tuple[EducationalStep, ...] = ()
    pipeline_step_index: int = 0

    # UI view state
    active_mode: str = "vectors"
    active_tab: str = "vectors"
    ui_theme: str = "dark"
    active_tool: str = "select"
    show_matrix_editor: bool = False
    show_matrix_values: bool = False
    show_image_on_grid: bool = True
    preview_enabled: bool = False
    matrix_plot_enabled: bool = False
    view_preset: str = "cube"
    view_up_axis: str = "z"
    view_grid_mode: str = "cube"
    view_grid_plane: str = "xy"
    view_grid_depth_order: str = "behind"  # 'behind' | 'overlay'
    view_grid_fade: bool = False
    view_grid_fade_start: float = 0.0
    view_grid_fade_end: float = 1.0
    view_show_grid: bool = True
    view_show_axes: bool = True
    view_show_labels: bool = True
    view_grid_size: int = 15
    view_base_major_tick: int = 5
    view_base_minor_tick: int = 1
    view_major_tick: int = 5
    view_minor_tick: int = 1
    view_auto_rotate: bool = False
    view_rotation_speed: float = 0.5
    view_show_cube_faces: bool = True
    view_show_cube_corners: bool = True
    view_cubic_grid_density: float = 1.0
    view_mode_2d: bool = False

    # History and counters
    history: Tuple['AppState', ...] = ()
    future: Tuple['AppState', ...] = ()
    next_vector_id: int = 1
    next_matrix_id: int = 1
    next_color_index: int = 0
```

`MAX_HISTORY = 20`; `create_initial_state()` seeds the Sobel kernel if available.

### Actions (organized by folder)
- `vector_actions.py`: AddVector, UpdateVector, DeleteVector, DuplicateVector, SelectVector, DeselectVector, ClearAllVectors
- `matrix_actions.py`: AddMatrix, UpdateMatrix/Cell, DeleteMatrix, SelectMatrix, ApplyMatrixToSelected/All, ToggleMatrixPlot
- `tensor_actions.py`: AddTensor, UpdateTensor, DeleteTensor, SelectTensor, DeselectTensor
- `image_actions.py`: Load/Process/Normalize image, SelectPixel, SetRenderMode, ToggleGridOverlay
- `input_actions.py` & `input_panel_actions.py`: maintain controlled inputs, parse text/file/grid
- `navigation_actions.py`: SetActiveMode/Tab/Tool, TogglePreview, SetViewPreset
- `pipeline_actions.py`: StepForward/Backward, JumpToStep, SetPipeline
- `history_actions.py`: Undo, Redo (bounded by MAX_HISTORY)
- `pipeline_actions.py` + `operation_history` combine to feed the timeline

### Selectors (state/selectors/)
- `get_selected_tensor`, `get_selected_vector`, `get_selected_matrix`
- `get_vectors_by_mode`, `get_visible_tensors`
- `get_pipeline_step(index)`, `get_operation_history`
- `can_undo`, `can_redo`
- `get_view_settings` (derived view-related toggles)

### Timeline + undo/redo wiring (next up)
- Timeline reads from `pipeline_steps`; `JumpToStep` updates `pipeline_step_index` and re-renders the preview only.
- `operation_history` entries are appended **once per commit**, not per step.
- Undo/redo restores `pipeline_steps`, `pipeline_step_index`, and `operation_history` so the timeline remains consistent.
- View-only actions (camera movement, theme, view toggles) should **not** record history.

### LaTeX export (next up)
- Export uses the current `operation_history` + `pipeline_steps` to generate a standalone `.tex`.
- Each step includes: `math_expression` (display math), `description` (paragraph), and `numeric_values` (aligned key/value or matrix block).
- Default output path: `exports/cvla_steps_YYYYMMDD_HHMMSS.tex` (configurable in settings).

---

## TESTING REQUIREMENTS

### Unit Test Coverage Targets
| Category | Target | Priority |
|----------|--------|----------|
| Domain math helpers (`domain/vectors`, `domain/transforms`, `domain/images`) | 100% | P0 |
| Reducers (`state/reducers/`) | 100% | P0 |
| Selectors (`state/selectors/`) | 90% | P1 |
| Input parsers (`ui/panels/input_panel/input_parsers.py`) | 90% | P1 |
| Scene adapter + history manager | 90% | P1 |
| Renderer smoke tests (headless, no GL asserts) | 70% | P2 |

### Headless renderer smoke tests (next up)
- Create an offscreen context and render a minimal scene (one vector, one grid, one image plane).
- Assert **no exceptions**, a valid framebuffer size, and a non-empty pixel buffer.
- Include a `SceneAdapter` regression case: known state → expected counts (vectors, planes, labels).

### Test Patterns
```python
from domain.vectors import vector_ops
import numpy as np

def test_dot_product():
    v1 = np.array([1, 2, 3], dtype=np.float32)
    v2 = np.array([4, 5, 6], dtype=np.float32)

    assert vector_ops.dot(v1, v2) == 32.0
```

```python
from state import Store, create_initial_state
from state.actions.vector_actions import AddVector, SelectVector

def test_add_and_select_vector():
    store = Store(create_initial_state())
    store.dispatch(AddVector((1, 0, 0), (1.0, 0.0, 0.0), "v1"))

    state = store.get_state()
    store.dispatch(SelectVector(state.vectors[0].id))

    new_state = store.get_state()
    assert len(new_state.vectors) == 1
    assert new_state.selected_id == new_state.vectors[0].id
```

```python
from engine.scene_adapter import create_scene_from_state
from state.actions.vector_actions import AddVector

def test_scene_adapter_produces_renderer_vectors():
    store = Store(create_initial_state())
    store.dispatch(AddVector((0, 1, 0), (0.2, 0.6, 1.0), "v"))

    scene = create_scene_from_state(store.get_state())
    assert len(scene.vectors) == 1
    assert scene.vectors[0].coords.shape == (3,)
```

---

## IMPLEMENTATION ORDER

*Updated Jan 18, 2026*

### Current Status
- ModernGL renderer with cube/plane grids, labels, and image planes is operational.
- Redux-style AppState with unified `TensorData`; input/operations/timeline/images/sidebar panels exist.
- Domain math helpers for vectors/matrices/transforms are present; no centralized operation registry yet.
- Operation history and educational pipeline scaffolding exist; automated tests are minimal.

### Next Milestones
1. **Operation registry + step traces**
   - Introduce `OperationSpec` + registry wrapping `vector_ops`, matrix helpers, and linear-system solvers.
   - Emit `EducationalStep` and `OperationRecord` entries; hook previews into `SceneAdapter`.
2. **Image & convolution pipeline**
   - Normalize ingestion via `engine.image_adapter`; expose kernel presets and consistent downsampling.
   - Generate stepped convolution/pooling outputs and link them to the timeline.
3. **Timeline & playback**
   - Connect `pipeline_actions` to UI timeline controls + keyboard shortcuts; keep undo/redo in sync with tensors/images.
   - Enforce bounded history (MAX_HISTORY) with sensible coalescing for slider changes.
4. **Performance + UX polish**
   - Cache heavy overlays (matrix plots, image grids); expose toggles in inspector/sidebar.
   - Profile frame time; guard against slow paths inside reducers and renderers.
5. **Testing**
   - Hit coverage targets above; add headless renderer smoke tests and regression tests for operation history.

---

## API REFERENCE

### Quick Start (state + actions)
```python
from state import Store, create_initial_state
from state.actions.vector_actions import AddVector, SelectVector
from engine.scene_adapter import create_scene_from_state

store = Store(create_initial_state())
store.dispatch(AddVector((1, 2, 3), (0.8, 0.2, 0.2), "v1"))
store.dispatch(AddVector((4, 5, 6), (0.2, 0.6, 1.0), "v2"))
store.dispatch(SelectVector(store.get_state().vectors[0].id))

scene = create_scene_from_state(store.get_state())
# renderer.render(scene)  # see rendering hook below
```

### Tensor factories
```python
from state.models.tensor_model import TensorData
import numpy as np

v = TensorData.create_vector((1, 0, 0), "v")
m = TensorData.create_matrix(((1, 0), (0, 1)), "A")
img = TensorData.create_image(np.zeros((64, 64, 3)), "blank")
```

### Math helpers
```python
from domain.vectors import vector_ops
from domain.transforms import affine_matrices

dot = vector_ops.dot([1, 2, 3], [4, 5, 6])
R = affine_matrices.rotation_matrix(axis="z", degrees=45)
```

### Rendering hook (headless)
```python
import moderngl
from render.renderers.renderer import Renderer
from render.cameras.camera import Camera
from render.viewconfigs.viewconfig import ViewConfig
from engine.scene_adapter import create_scene_from_state

ctx = moderngl.create_standalone_context()
renderer = Renderer(ctx, Camera(), ViewConfig())
renderer.render(create_scene_from_state(store.get_state()))
```

---

## DEFINITION OF DONE (v1.0)

CVLA v1.0 is complete when:

### Functional Requirements
- [ ] Build a vector space with 3+ vectors
- [ ] Add matrices and apply to vectors
- [ ] Perform Gaussian elimination step-by-step
- [ ] Visualize null space and column space
- [ ] Watch convolution slide across an image
- [ ] Replay entire computation like a film

### Quality Requirements
- [ ] 60 FPS maintained throughout
- [ ] All operations produce step lists
- [ ] Zero silent failures
- [ ] All 11 invariants hold

### Developer Experience
- [ ] Add new operation in < 10 minutes
- [ ] Never touch renderer for new math
- [ ] All operations in registry

---

## APPENDIX: KEYBOARD SHORTCUTS

| Key | Action |
|-----|--------|
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+Shift+Z` | Redo (alternate) |
| `Space` | Play/pause steps |
| `←` | Previous step |
| `→` | Next step |
| `Home` | First step |
| `End` | Last step |
| `Delete` | Delete selected tensor |
| `Escape` | Deselect / cancel |
| `V` | Vector mode |
| `M` | Matrix mode |
| `I` | Image mode |

---

## APPENDIX: COLOR SCHEME

| Element | Color (RGBA) |
|---------|--------------|
| X-axis | (1.0, 0.2, 0.2, 1.0) |
| Y-axis | (0.2, 1.0, 0.2, 1.0) |
| Z-axis | (0.2, 0.2, 1.0, 1.0) |
| Selected | (1.0, 0.8, 0.0, 1.0) |
| Input vector | (0.2, 0.6, 1.0, 1.0) |
| Output vector | (1.0, 0.4, 0.2, 1.0) |
| Intermediate | (0.8, 0.8, 0.2, 0.7) |
| Grid | (0.3, 0.3, 0.3, 0.5) |
| Subspace plane | (0.5, 0.5, 0.8, 0.3) |

---

## PROGRESS CHECKLIST (2026-01-18 snapshot)

**Definition of Done alignment**
- [x] Build a vector space with 3+ vectors (interactive)  
- [x] Add matrices and apply to vectors (with visuals)  
- [ ] Gaussian elimination step-by-step  
- [ ] Visualize null space and column space  
- [ ] Watch convolution slide across an image  
- [ ] Replay entire computation like a film  
- [ ] 60 FPS maintained throughout (target perf)  
- [ ] All operations produce step lists  
- [ ] Zero silent failures & invariants enforced  
- [ ] Add new operation in < 10 minutes  
- [ ] Never touch renderer for new math (registry abstraction)  
- [ ] All operations registered

**Implemented / in-code**
- [x] Unified `TensorData` / `TensorDType`
- [x] SceneAdapter includes legacy + unified tensors
- [x] Renderer grid/axes tuned for zoom visibility
- [x] Matrix columns rendered as vectors (temp)
- [x] Image plane render path (depth off) for visibility
- [x] Matrix decompositions (eig/qr/svd/lu) wired to operations panel
- [x] Image ops wired: rotate, scale, flip, normalize, grayscale, invert, to-matrix

**In progress / next up**
- [ ] Operation registry wired to domain math helpers + UI
- [ ] Step traces attached to timeline & operation history
- [ ] Image/convolution pipeline with stepped outputs
- [ ] Headless renderer smoke tests; state→scene adapter tests
- [ ] UI toggles for grid fade/depth ordering
- [ ] Export steps to LaTeX
- [ ] Undo/redo fully wired for tensors and images

*Spec note:* The items above are now fully specified in the Operation Registry, State Management, UI Layout, Rendering System, and Testing sections; implementation remains pending.

---

*Document Version: 1.0*
*Last Updated: 2026-01-18*
*Target Implementation: Python 3.10+, ModernGL, ImGui*
