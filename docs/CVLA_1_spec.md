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

### TensorData (Unified Mathematical Object)
```python
@dataclass(frozen=True)
class TensorData:
    data: Tuple[float, ...]     # Flattened data
    shape: Tuple[int, ...]      # Shape information
    dtype: str = "float64"
    label: str = ""

    # Factory methods
    @staticmethod
    def create_scalar(value: float, label: str = "") -> TensorData
    @staticmethod
    def create_vector(values: Tuple[float, ...], label: str = "") -> TensorData
    @staticmethod
    def create_matrix(values: Tuple[Tuple[float, ...], ...], label: str = "") -> TensorData
    @staticmethod
    def from_numpy(arr: np.ndarray, label: str = "") -> TensorData

    def to_numpy(self) -> np.ndarray

    @property
    def tensor_type(self) -> TensorType  # SCALAR, VECTOR, MATRIX, TENSOR3
```

### TensorType (Classification)
```python
class TensorType(Enum):
    SCALAR = auto()      # 0-dimensional: single number
    VECTOR = auto()      # 1-dimensional: (n,)
    MATRIX = auto()      # 2-dimensional: (m, n)
    TENSOR3 = auto()     # 3-dimensional: (d, m, n) - for images
    IMAGE = auto()       # Special case with channel semantics
```

### Step (Educational Trace)
```python
@dataclass(frozen=True)
class Step:
    step_index: int
    math_expression: str          # LaTeX
    description: str              # Plain English
    numeric_values: Dict[str, Any]
    geometric_change: GeometricDelta
    reversible: bool = True
    substeps: Tuple[Step, ...] = ()
```

### GeometricDelta (Visualization Hint)
```python
@dataclass(frozen=True)
class GeometricDelta:
    change_type: str  # 'scale', 'rotate', 'shear', 'translate', 'project', 'reflect', 'none'
    scale_factor: Optional[Tuple[float, ...]] = None
    rotation_axis: Optional[Tuple[float, float, float]] = None
    rotation_angle: Optional[float] = None  # radians
    translation: Optional[Tuple[float, ...]] = None
    old_basis: Optional[Tuple[Tuple[float, ...], ...]] = None
    new_basis: Optional[Tuple[Tuple[float, ...], ...]] = None
    highlight_vectors: Tuple[str, ...] = ()
```

### RenderHints (How to Visualize)
```python
@dataclass(frozen=True)
class RenderHints:
    show_input_vectors: bool = True
    show_output_vectors: bool = True
    show_intermediate: bool = True
    show_basis_change: bool = False
    show_grid_deformation: bool = False
    animate_transformation: bool = True
    animation_duration: float = 0.5  # seconds
    input_color: Tuple[float, float, float, float] = (0.2, 0.6, 1.0, 1.0)
    output_color: Tuple[float, float, float, float] = (1.0, 0.4, 0.2, 1.0)
    show_projection_line: bool = False
    show_orthogonal_complement: bool = False
    show_span: bool = False
    show_kernel: bool = False
    show_image_space: bool = False
```

### Result Type (Error Handling)
```python
@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T
    def is_ok(self) -> bool: return True
    def unwrap(self) -> T: return self.value

@dataclass(frozen=True)
class Err(Generic[E]):
    error: E
    def is_ok(self) -> bool: return False
    def unwrap(self): raise ValueError(self.error)

Result = Union[Ok[T], Err[E]]
```

---

## OPERATION REGISTRY SYSTEM

### OperationSpec (Interface)
Every operation MUST implement this interface:

```python
class OperationSpec(ABC):
    # === METADATA ===
    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier: 'dot_product', 'matrix_multiply'"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name: 'Dot Product', 'Matrix Multiplication'"""

    @property
    @abstractmethod
    def category(self) -> str:
        """Category: 'vector', 'matrix', 'transform', 'decomposition', 'image'"""

    @property
    @abstractmethod
    def inputs(self) -> Tuple[TensorType, ...]:
        """Required input types"""

    @property
    @abstractmethod
    def outputs(self) -> Tuple[TensorType, ...]:
        """Output types"""

    @property
    def assumptions(self) -> Tuple[str, ...]:
        """Assumptions: ('same_dimension', 'square_matrix')"""
        return ()

    # === EDUCATIONAL ===
    @property
    @abstractmethod
    def description(self) -> str:
        """What it does"""

    @property
    @abstractmethod
    def intuition(self) -> str:
        """Why it works"""

    @property
    def failure_modes(self) -> Tuple[str, ...]:
        """When it fails"""
        return ()

    @property
    @abstractmethod
    def geometric_meaning(self) -> str:
        """What it represents geometrically"""

    # === CORE METHODS ===
    @abstractmethod
    def validate(self, *tensors: TensorData) -> Result[None, str]:
        """Validate inputs. Return Ok(None) or Err(message)."""

    @abstractmethod
    def compute(self, *tensors: TensorData) -> TensorData:
        """Perform computation. Precondition: validate() returned Ok."""

    @abstractmethod
    def steps(self, *tensors: TensorData) -> Tuple[Step, ...]:
        """Generate step-by-step breakdown."""

    def render_hints(self) -> RenderHints:
        """Provide visualization hints."""
        return RenderHints()
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
|  [CVLA] [Undo] [Redo]              [Theme] [Settings]            |
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

## FILE STRUCTURE

```
CVLA/
├── main.py                          # Entry point
├── docs/
│   └── CVLA_1_spec.md              # This specification
│
├── state/                           # Redux-like state management
│   ├── __init__.py
│   ├── app_state.py                 # AppState dataclass
│   ├── store.py                     # Store with dispatch
│   ├── actions/
│   │   ├── __init__.py
│   │   ├── tensor_actions.py        # AddTensor, DeleteTensor, etc.
│   │   ├── operation_actions.py     # ExecuteOperation, etc.
│   │   ├── step_actions.py          # StepForward, StepBackward, etc.
│   │   └── ... (existing actions)
│   ├── reducers/
│   │   ├── __init__.py
│   │   ├── reducer_tensors.py
│   │   ├── reducer_operations.py
│   │   ├── reducer_steps.py
│   │   └── ... (existing reducers)
│   └── selectors/
│       ├── __init__.py
│       └── tensor_selectors.py
│
├── domain/                          # Pure math, no UI
│   ├── __init__.py
│   ├── operations/
│   │   ├── __init__.py
│   │   ├── registry.py              # OperationSpec, registry
│   │   ├── vector_ops.py            # VectorAddition, DotProduct, etc.
│   │   ├── matrix_ops.py            # MatrixMultiply, Determinant, etc.
│   │   ├── transform_ops.py         # RotationMatrix, Shear, etc.
│   │   ├── eigen_ops.py             # Eigenvalues, Eigenvectors
│   │   ├── system_ops.py            # GaussianElimination, LU
│   │   ├── subspace_ops.py          # NullSpace, ColumnSpace
│   │   └── conv_ops.py              # Convolution2D, SobelEdge
│   └── models/
│       ├── __init__.py
│       ├── tensor.py                # TensorData, VisualTensor
│       └── step.py                  # Step, StepSequence, StepState
│
├── engine/                          # Rendering bridge
│   ├── __init__.py
│   ├── scene_adapter.py             # State → renderable objects
│   └── step_player.py               # Step animation controller
│
├── render/                          # OpenGL rendering
│   ├── __init__.py
│   ├── cameras/
│   │   └── camera.py
│   ├── renderers/
│   │   ├── renderer.py              # Main renderer
│   │   ├── renderer_vectors.py
│   │   ├── renderer_axes.py
│   │   └── ...
│   ├── shaders/
│   │   └── gizmo_programs.py
│   └── viewconfigs/
│       └── viewconfig.py
│
├── ui/                              # ImGui interface
│   ├── __init__.py
│   ├── layout/
│   │   └── workspace.py             # Main layout
│   ├── panels/
│   │   ├── input_panel/
│   │   │   ├── __init__.py
│   │   │   ├── text_input.py
│   │   │   └── grid_input.py
│   │   ├── operations_panel/
│   │   │   └── __init__.py
│   │   ├── timeline/
│   │   │   └── timeline_panel.py
│   │   └── sidebar/
│   │       └── ... (existing)
│   └── toolbars/
│       └── toolbar.py
│
├── app/                             # Application core
│   ├── app.py
│   ├── app_run.py
│   └── ...
│
└── tests/                           # Test coverage
    ├── __init__.py
    ├── test_operations/
    │   ├── test_vector_ops.py
    │   ├── test_matrix_ops.py
    │   └── ...
    ├── test_reducers/
    │   └── ...
    └── test_integration/
        └── ...
```

---

## STATE MANAGEMENT

### AppState (Single Source of Truth)
```python
@dataclass(frozen=True)
class AppState:
    # Scene data
    tensors: Tuple[VisualTensor, ...] = ()

    # Selection
    selected_tensor_id: Optional[str] = None

    # Current operation
    active_operation_id: Optional[str] = None
    operation_result: Optional[TensorData] = None
    operation_steps: Tuple[Step, ...] = ()
    current_step_index: int = 0

    # Playback
    is_playing: bool = False
    playback_speed: float = 1.0

    # UI state
    active_mode: str = "vector"  # vector, matrix, image
    active_panel: str = "input"

    # History
    history: Tuple[AppState, ...] = ()
    future: Tuple[AppState, ...] = ()

    # Counters
    next_tensor_id: int = 1
```

### Actions
```python
# Tensor actions
AddTensor(data, shape, label, color)
DeleteTensor(tensor_id)
UpdateTensor(tensor_id, **updates)
SelectTensor(tensor_id)
DeselectTensor()

# Operation actions
ExecuteOperation(operation_id, input_tensor_ids)
ClearOperationResult()
SetActiveOperation(operation_id)

# Step actions
StepForward()
StepBackward()
JumpToStep(step_index)
ToggleStepPlayback()
SetPlaybackSpeed(speed)

# History actions
Undo()
Redo()
```

### Selectors
```python
def get_selected_tensor(state: AppState) -> Optional[VisualTensor]
def get_tensors_by_type(state: AppState, tensor_type: TensorType) -> Tuple[VisualTensor, ...]
def get_current_step(state: AppState) -> Optional[Step]
def get_available_operations(state: AppState) -> List[OperationSpec]
def can_undo(state: AppState) -> bool
def can_redo(state: AppState) -> bool
```

---

## TESTING REQUIREMENTS

### Unit Test Coverage Targets
| Category | Target | Priority |
|----------|--------|----------|
| Operations (domain/) | 100% | P0 |
| Reducers (state/reducers/) | 100% | P0 |
| Selectors (state/selectors/) | 90% | P1 |
| Input parsers | 90% | P1 |
| Step generation | 100% | P0 |

### Test Pattern for Operations
```python
def test_dot_product():
    # Given
    v1 = TensorData.create_vector((1, 2, 3), "v1")
    v2 = TensorData.create_vector((4, 5, 6), "v2")

    # When
    op = registry.get("dot_product")
    result = op.compute(v1, v2)
    steps = op.steps(v1, v2)

    # Then
    assert result.data == (32.0,)  # 1*4 + 2*5 + 3*6
    assert len(steps) >= 3  # At least 3 steps
    assert steps[-1].numeric_values["result"] == 32.0

def test_dot_product_validation():
    # Given: vectors of different dimensions
    v1 = TensorData.create_vector((1, 2, 3), "v1")
    v2 = TensorData.create_vector((4, 5), "v2")

    # When
    op = registry.get("dot_product")
    result = op.validate(v1, v2)

    # Then
    assert result.is_err()
    assert "dimension" in result.error.lower()
```

### Integration Test Pattern
```python
def test_operation_flow():
    # Given: initial state with two vectors
    state = create_initial_state()
    store = Store(state)

    # When: execute dot product
    store.dispatch(ExecuteOperation("dot_product", ("v1", "v2")))

    # Then: result is stored, steps are generated
    new_state = store.get_state()
    assert new_state.operation_result is not None
    assert len(new_state.operation_steps) > 0
    assert new_state.current_step_index == 0
```

---

## IMPLEMENTATION ORDER

### Phase 1: Core Infrastructure (Week 1)
1. ✅ `domain/operations/registry.py` - OperationSpec, registry
2. ✅ `domain/models/tensor.py` - TensorData
3. ✅ `domain/models/step.py` - Step, StepBuilder
4. ✅ Vector operations with steps
5. ✅ Matrix operations with steps
6. State actions for operations
7. Reducers for operations
8. Basic selectors

### Phase 2: More Operations (Week 2)
1. ✅ Transform operations
2. ✅ Eigenvalue operations
3. ✅ Linear system operations
4. ✅ Subspace operations
5. ✅ Convolution operations
6. Unit tests for all operations

### Phase 3: UI Integration (Week 3)
1. Input panel (text/grid)
2. Operations panel
3. Timeline panel
4. Step navigation
5. Wire up to existing viewport

### Phase 4: Polish (Week 4)
1. Animation system
2. Step playback
3. Undo/redo
4. Keyboard shortcuts
5. Performance optimization

---

## API REFERENCE

### Quick Start
```python
from domain.operations import registry, TensorData

# Create vectors
v1 = TensorData.create_vector((1, 2, 3), "v1")
v2 = TensorData.create_vector((4, 5, 6), "v2")

# Execute operation
result = registry.execute("dot_product", v1, v2)
if result.is_ok():
    output, steps = result.unwrap()
    print(f"Result: {output.data}")  # (32.0,)
    for step in steps:
        print(f"{step.step_index}: {step.description}")
```

### List All Operations
```python
from domain.operations import registry

for op in registry.list_all():
    print(f"{op.id}: {op.name} ({op.category})")
    print(f"  {op.description}")
    print(f"  Inputs: {op.inputs}")
    print(f"  Outputs: {op.outputs}")
```

### Add Custom Operation
```python
from domain.operations.registry import OperationSpec, register_operation

@register_operation
class MyOperation(OperationSpec):
    @property
    def id(self): return "my_operation"
    @property
    def name(self): return "My Operation"
    @property
    def category(self): return "custom"
    # ... implement all abstract methods
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

*Document Version: 1.0*
*Last Updated: 2025*
*Target Implementation: Python 3.10+, ModernGL, ImGui*
