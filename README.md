# CVLA - Complete Visual Linear Algebra

A comprehensive 3D linear algebra visualization tool with an intuitive interface for exploring vectors, matrices, and linear transformations.

## Features

### Core Visualization
- **3D Vector Visualization**: Visualize vectors in 3D space with customizable colors and labels
- **Interactive Camera**: Orbit, pan, and zoom with intuitive mouse controls
- **Multiple View Modes**: 2D/3D toggle, orthographic/perspective projections
- **Grid Systems**: Customizable grid on XY, XZ, YZ planes or full 3D cube

### Linear Algebra Operations
- **Vector Operations**: Addition, subtraction, dot product, cross product, normalization
- **Matrix Operations**: Create and apply transformation matrices
- **Linear Systems**: Solve systems of equations with Gaussian elimination
- **Projections**: Visualize vector projections onto axes and planes

### Enhanced UI
- **Photoshop-Style Workspace**: Top ribbon, left tool palette, right inspector, bottom timeline
- **Theme Support**: Dark, light, and high-contrast modes
- **Real-time Inspector**: Detailed property inspection for selected vectors
- **Action-Driven UI**: Panels dispatch actions; state is the only source of truth

### Advanced Features
- **Gaussian Elimination**: Step-by-step visualization of solving linear systems
- **Null Space/Column Space**: Compute and visualize fundamental subspaces
- **Educational Timeline**: Navigate pipeline steps and intermediate states
- **Export Capabilities**: Export vectors to JSON, CSV, or Python code

## Architecture

CVLA follows a strict one-directional data flow:

1. UI dispatches actions
2. Reducers create new immutable state
3. SceneAdapter projects state into renderable data
4. Renderers visualize the projection

The codebase is organized into clear layers:

- `domain/` for pure math and vision logic
- `state/` for Redux-style store, actions, reducers, and selectors
- `engine/` for runtime orchestration (scene adapter, picking)
- `render/` for cameras, view configs, gizmos, and renderers
- `ui/` for panels, inspectors, layouts, and themes
- `app/` for bootstrap and wiring

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Jdawg0309/CVLA.git
cd CVLA
```

For a full, per-function mapping and usage notes see `docs/FUNCTIONS.md`.


2. Create and activate a virtual environment, then install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run the app:
```bash
python main.py
```

---

If you contributed an initial README on GitHub, I merged the remote README content and preserved the detailed project description here.
