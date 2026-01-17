# CVLA Codebase Inventory
## Technical Autopsy and Forensic Documentation

**Generated:** 2026-01-17
**Codebase Location:** `/home/junaet/Desktop/ML/CVLA`
**Total Python Files:** 168
**Architecture Style:** Redux-inspired GUI application with OpenGL rendering

---

# 1. GLOBAL OVERVIEW

## 1.1 Application Entry Point

**File:** `main.py` (lines 1-156)

The application starts via the `main()` function which:
1. Checks for command-line arguments (`--demo-vision`, `--help`)
2. If no arguments: instantiates `App()` from `app/app.py`
3. Calls `app.run()` to enter the main loop

```
python main.py           → Launches GUI application
python main.py --demo-vision → Runs CLI image processing demo
python main.py --help    → Shows usage information
```

The `demo_image_processing()` function (lines 21-126) provides a standalone CLI demonstration of image/convolution operations without launching the GUI.

## 1.2 Objects Instantiated at Startup

When `App()` is constructed (`app/app.py` lines 26-128), the following objects are created **in order**:

| Object | Type | Purpose |
|--------|------|---------|
| `self.window` | GLFW Window | Native window handle (1440x900) |
| `self.ctx` | ModernGL Context | OpenGL rendering context |
| `imgui context` | ImGui Context | Immediate-mode GUI context |
| `self.imgui` | GlfwRenderer | ImGui GLFW integration |
| `self.camera` | Camera | 3D orbital camera with view matrices |
| `self.view_config` | ViewConfig | Grid/axis/label display settings |
| `self.renderer` | Renderer | Main 3D rendering system |
| `self.labels` | LabelRenderer | Text label overlay system |
| `self.workspace` | WorkspaceLayout | UI panel layout manager |
| `self.store` | Store | Redux-style state container |

**Additional instance variables on App:**
- `self.selected` = None (legacy, unused)
- `self.rotating` = False (right-mouse-drag state)
- `self.last_mouse` = None (previous mouse position)
- `self.fps` = 0.0 (frames per second counter)

## 1.3 Main Loop / Control Flow

**File:** `app/app_run.py` function `run(self)`

The main loop follows this exact sequence each frame:

```
while not glfw.window_should_close(self.window):
    1. delta_time = timer.tick()           → Frame timing
    2. glfw.poll_events()                   → OS events
    3. self.imgui.process_inputs()          → ImGui input
    4. Get framebuffer size, skip if zero
    5. Set viewport and camera dimensions
    6. Handle auto-rotate if enabled
    7. Update renderer/labels with view_config
    8. state = self.store.get_state()       → READ STATE
    9. scene_adapter = build_scene_adapter(state)
    10. imgui.new_frame()
    11. self.workspace.render(...)          → ALL UI RENDERING
    12. self.renderer.render(scene_adapter, ...)  → 3D SCENE
    13. Calculate FPS
    14. self.labels.draw_*(...)             → LABEL OVERLAYS
    15. imgui.render()
    16. self.imgui.render(draw_data)
    17. glfw.swap_buffers(self.window)
```

**Critical observation:** State is read once per frame from the Store. The SceneAdapter is created fresh each frame from that state. UI and renderer both receive the same state snapshot.

## 1.4 User Interaction Entry/Exit Points

**Input Entry Points:**

| Callback | File | Triggered By |
|----------|------|--------------|
| `on_key` | `app_handlers.py:18` | Keyboard press |
| `on_mouse_button` | `app_handlers.py:53` | Mouse click |
| `on_mouse_move` | `app_handlers.py:97` | Mouse movement |
| `on_scroll` | `app_handlers.py:114` | Mouse wheel |
| `on_resize` | `app_handlers.py:48` | Window resize |
| ImGui widgets | (various UI files) | Button clicks, text input, sliders |

**State Mutation Exit Points:**

All state changes flow through exactly one path:
```
User Action → dispatch(Action) → reduce(state, action) → New State
```

The `dispatch` function is passed down through the UI hierarchy and called when users interact with widgets.

---

# 2. FULL FUNCTION INVENTORY

## 2.1 Application Layer (`app/`)

### app/app.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class App` | 25 | Main application class | - | - | - |
| `App.__init__` | 26-128 | Initialize GLFW, OpenGL, ImGui, all subsystems | - | Creates all instance vars | glfw.*, moderngl.*, imgui.*, Camera(), Renderer(), etc. |
| `_setup_imgui_style` | 129 | Alias to `app_style.setup_imgui_style` | self | imgui style colors | imgui.get_style() |
| `on_key` | 130 | Alias to `app_handlers.on_key` | - | view_config, camera | - |
| `on_resize` | 131 | Alias to `app_handlers.on_resize` | - | camera viewport | - |
| `on_mouse_button` | 132 | Alias to `app_handlers.on_mouse_button` | - | selection, camera | - |
| `on_mouse_move` | 133 | Alias to `app_handlers.on_mouse_move` | - | camera angles | - |
| `on_scroll` | 134 | Alias to `app_handlers.on_scroll` | - | camera zoom | - |
| `run` | 135 | Alias to `app_run.run` | - | All via dispatch | - |

### app/app_handlers.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `on_key` | 18-46 | Handle keyboard shortcuts | self, win, key, scancode, action, mods | view_config.auto_rotate, grid_mode, show_cube_faces; renderer.show_vector_components; camera | glfw.PRESS checks, imgui.keyboard_callback |
| `on_resize` | 48-50 | Handle window resize | self, win, width, height | camera viewport | camera.set_viewport() |
| `on_mouse_button` | 53-94 | Handle mouse clicks | self, win, btn, action, mods | self.rotating, camera target, store (via dispatch) | pick_vector(), build_scene_adapter(), store.dispatch(SelectVector) |
| `on_mouse_move` | 97-111 | Handle mouse drag for camera orbit | self, win, x, y | camera.theta, camera.phi, self.last_mouse | camera.orbit() |
| `on_scroll` | 114-117 | Handle scroll for zoom | self, win, xoff, yoff | camera.radius | camera.zoom() |

### app/app_run.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `run` | 16-98 | Main rendering loop | self | All via workspace.render and renderer.render | FrameTimer.tick(), glfw.*, imgui.*, workspace.render(), renderer.render(), labels.draw_*() |

### app/app_state_bridge.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `build_scene_adapter` | 6-8 | Create SceneAdapter from AppState | state: AppState | None (read-only) | SceneAdapter(state) |

### app/app_style.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `setup_imgui_style` | 8-29 | Configure ImGui visual style | self | imgui style object | imgui.get_style() |

### app/app_logging.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `DEBUG` | 8 | Global constant from env var | - | - | os.getenv() |
| `dlog` | 11-13 | Debug logging function | msg: str | None | print() |

---

## 2.2 State Layer (`state/`)

### state/store.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class Store` | 12-34 | Redux-style state container | initial_state: AppState | self._state | - |
| `Store.__init__` | 17-19 | Initialize store | initial_state | self._state, self._listeners | - |
| `Store.get_state` | 21-23 | Return current state (read-only) | - | None | - |
| `Store.dispatch` | 25-29 | Process action through reducer | action: Action | self._state | reduce(state, action), all listeners |
| `Store.subscribe` | 31-34 | Add state change listener | listener: Callable | self._listeners | - |

### state/app_state.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `MAX_HISTORY` | 27 | Constant = 20 | - | - | - |
| `class AppState` | 30-137 | Frozen dataclass containing ALL application state | - | - | - |
| `create_initial_state` | 139-158 | Factory for initial state with default vectors | - | - | VectorData.create() |

**AppState Fields (Complete List):**

**Scene State:**
- `vectors: Tuple[VectorData, ...]` - All vectors in scene
- `matrices: Tuple[MatrixData, ...]` - All matrices in scene
- `planes: Tuple[PlaneData, ...]` - All planes in scene

**Selection State:**
- `selected_id: Optional[str]` - UUID of selected object
- `selected_type: Optional[str]` - 'vector', 'matrix', 'plane', 'image'

**Image/Vision State:**
- `current_image: Optional[ImageData]` - Loaded source image
- `processed_image: Optional[ImageData]` - Result after operations
- `selected_kernel: str` - Current kernel name ('sobel_x')
- `image_status: str` - Status message
- `image_status_level: str` - "info" or "error"
- `image_pipeline: Tuple[PipelineOp, ...]` - Operation sequence
- `active_pipeline_index: int` - Current step
- `micro_step_index/total: int` - Sub-step tracking
- `micro_op: Optional[MicroOp]` - Current micro-operation
- `selected_pixel: Optional[Tuple[int, int]]` - Highlighted pixel
- `image_render_mode: str` - 'plane' or 'height-field'
- `image_render_scale: float` - Pixel spacing (1.0)
- `image_color_mode: str` - 'rgb', 'grayscale', 'heatmap'
- `image_auto_fit: bool` - Auto-scale to grid
- `show_image_grid_overlay: bool` - Pixel grid lines
- `image_downsample_enabled: bool` - Reduce resolution on load
- `image_preview_resolution: int` - Downsample target (128)
- `image_max_resolution: int` - Hard limit (512)

**Educational Pipeline State:**
- `pipeline_steps: Tuple[EducationalStep, ...]` - Step definitions
- `pipeline_step_index: int` - Current step

**UI Input State (Controlled Inputs):**
- `input_vector_coords: Tuple[float, float, float]` - Vector form
- `input_vector_label: str` - Vector form
- `input_vector_color: Tuple[float, float, float]` - Vector form
- `input_matrix: Tuple[Tuple[float, ...], ...]` - Matrix form
- `input_matrix_label: str` - Matrix form
- `input_matrix_size: int` - Matrix dimensions
- `input_image_path: str` - File path input
- `input_sample_pattern: str` - 'checkerboard', 'gradient', etc.
- `input_sample_size: int` - Sample dimensions
- `input_transform_rotation: float` - Rotation degrees
- `input_transform_scale: float` - Scale factor
- `input_image_normalize_mean/std: float` - Normalization params
- `active_image_tab: str` - 'raw' or 'preprocess'

**UI View State:**
- `active_tab: str` - 'vectors', 'matrices', 'systems', 'images', 'visualize'
- `ribbon_tab: str` - 'File'
- `ui_theme: str` - 'dark', 'light', 'high-contrast'
- `active_tool: str` - 'select', 'move', 'add_vector', etc.
- `show_matrix_editor: bool` - Editor visibility
- `show_matrix_values: bool` - Value overlay
- `show_heatmap: bool` - Heatmap display
- `show_channels: bool` - Channel separation
- `show_image_on_grid: bool` - Image plane visibility
- `preview_enabled: bool` - Matrix preview mode
- `matrix_plot_enabled: bool` - 3D matrix visualization

**History (Undo/Redo):**
- `history: Tuple[AppState, ...]` - Past states (max 20)
- `future: Tuple[AppState, ...]` - Redo stack

**Counters:**
- `next_vector_id: int` - Auto-naming counter
- `next_matrix_id: int` - Auto-naming counter
- `next_color_index: int` - Color palette index

### state/models/vector_model.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class VectorData` | 12-38 | Frozen dataclass for vector | - | - | - |
| `VectorData.create` | 23-33 | Factory with UUID generation | coords, color, label | - | uuid4() |
| `VectorData.to_numpy` | 35-37 | Convert to numpy array | - | - | np.array() |

### state/models/matrix_model.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class MatrixData` | 12-62 | Frozen dataclass for matrix | - | - | - |
| `MatrixData.create` | 22-30 | Factory with UUID | values, label | - | uuid4() |
| `MatrixData.identity` | 32-39 | Create identity matrix | size, label | - | uuid4() |
| `MatrixData.to_numpy` | 41-43 | Convert to numpy | - | - | np.array() |
| `MatrixData.shape` | 45-50 | Return (rows, cols) | - | - | - |
| `MatrixData.with_cell` | 52-61 | Return copy with one cell changed | row, col, value | - | - |

### state/models/image_model.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class ImageData` | 12-72 | Frozen dataclass for image | - | - | - |
| `ImageData.__post_init__` | 22-25 | Ensure pixels are copied | - | self.pixels (via object.__setattr__) | - |
| `ImageData.create` | 27-37 | Factory with UUID | pixels, name, history | - | uuid4() |
| `ImageData.height/width/channels` | 39-49 | Property accessors | - | - | - |
| `ImageData.is_grayscale` | 51-53 | Check if single channel | - | - | - |
| `ImageData.as_matrix` | 55-61 | Convert to grayscale matrix | - | - | np operations |
| `ImageData.with_history` | 63-71 | Return copy with operation added | operation: str | - | - |

### state/models/educational_step.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class EducationalStep` | 14-54 | Frozen dataclass for pipeline step | - | - | - |
| `EducationalStep.create` | 29-54 | Factory method | title, explanation, operation, ... | - | uuid4() |

### state/models/pipeline_models.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class PipelineOp` | 10-25 | Pipeline operation definition | - | - | - |
| `PipelineOp.create` | 19-25 | Factory | name, op_type, kernel_name | - | uuid4() |
| `class MicroOp` | 28-43 | Micro-operation snapshot | - | - | - |

### state/actions/*.py (Complete Action Catalog)

**Vector Actions (vector_actions.py):**
| Action | Fields | Purpose |
|--------|--------|---------|
| `AddVector` | coords, color, label | Add new vector |
| `DeleteVector` | id | Remove vector by ID |
| `UpdateVector` | id, coords?, color?, label?, visible? | Modify vector |
| `SelectVector` | id | Set selection |
| `ClearAllVectors` | - | Remove all vectors |
| `DuplicateVector` | id | Clone vector |
| `DeselectVector` | - | Clear selection |

**Matrix Actions (matrix_actions.py):**
| Action | Fields | Purpose |
|--------|--------|---------|
| `AddMatrix` | values, label | Add new matrix |
| `DeleteMatrix` | id | Remove matrix |
| `UpdateMatrixCell` | id, row, col, value | Modify single cell |
| `UpdateMatrix` | id, values?, label? | Modify matrix |
| `SelectMatrix` | id | Set selection |
| `ApplyMatrixToSelected` | matrix_id | Transform selected vector |
| `ApplyMatrixToAll` | matrix_id | Transform all vectors |
| `ToggleMatrixPlot` | - | Toggle 3D visualization |

**Image Actions (image_actions.py):**
| Action | Fields | Purpose |
|--------|--------|---------|
| `LoadImage` | path, max_size? | Load from file |
| `CreateSampleImage` | pattern, size | Generate test image |
| `ApplyKernel` | kernel_name | Apply convolution |
| `ApplyTransform` | rotation, scale | Affine transform |
| `FlipImageHorizontal` | - | Mirror image |
| `UseResultAsInput` | - | Swap processed→current |
| `ClearImage` | - | Clear all images |
| `NormalizeImage` | mean?, std? | Normalize pixel values |

**Input Actions (input_actions.py):**
| Action | Fields | Purpose |
|--------|--------|---------|
| `SetInputVector` | coords?, label?, color? | Update vector form |
| `SetInputMatrixCell` | row, col, value | Update matrix form cell |
| `SetInputMatrixSize` | size | Resize matrix form |
| `SetInputMatrixLabel` | label | Update matrix label |
| `SetImagePath` | path | Update path input |
| `SetSamplePattern` | pattern | Update pattern selector |
| `SetSampleSize` | size | Update size input |
| `SetTransformRotation` | rotation | Update rotation slider |
| `SetTransformScale` | scale | Update scale slider |
| `SetSelectedKernel` | kernel_name | Update kernel selector |
| `SetImageRenderScale` | scale | Update render scale |
| `SetImageRenderMode` | mode | 'plane' or 'height-field' |
| `SetImageColorMode` | mode | 'grayscale', 'heatmap', 'rgb' |
| `SetImageNormalizeMean` | mean | Normalization param |
| `SetImageNormalizeStd` | std | Normalization param |
| `ToggleImageGridOverlay` | - | Toggle pixel grid |
| `ToggleImageDownsample` | - | Toggle downsampling |
| `SetImagePreviewResolution` | size | Set preview resolution |

**Navigation Actions (navigation_actions.py):**
| Action | Fields | Purpose |
|--------|--------|---------|
| `SetActiveTab` | tab | Switch main tab |
| `ToggleMatrixEditor` | - | Show/hide editor |
| `ToggleMatrixValues` | - | Show/hide values |
| `ToggleImageOnGrid` | - | Show/hide image plane |
| `TogglePreview` | - | Toggle preview mode |
| `SetActiveImageTab` | tab | 'raw' or 'preprocess' |
| `ClearSelection` | - | Deselect all |
| `SetTheme` | theme | Change UI theme |
| `SetActiveTool` | tool | Change active tool |

**Pipeline Actions (pipeline_actions.py):**
| Action | Fields | Purpose |
|--------|--------|---------|
| `StepForward` | - | Advance pipeline |
| `StepBackward` | - | Go back |
| `JumpToStep` | index | Jump to specific step |
| `ResetPipeline` | - | Clear pipeline |

**History Actions (history_actions.py):**
| Action | Fields | Purpose |
|--------|--------|---------|
| `Undo` | - | Restore previous state |
| `Redo` | - | Restore next state |

### state/reducers/__init__.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `reduce` | 20-59 | Main reducer function | state, action | Returns new AppState | reduce_history, reduce_vectors, reduce_matrices, reduce_images, reduce_pipeline, reduce_inputs, reduce_navigation |

### state/reducers/reducer_vectors.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `reduce_vectors` | 15-95 | Handle vector actions | state, action, with_history | Returns new state or None | VectorData.create, dataclasses.replace |

**Handled actions:** AddVector, DeleteVector, UpdateVector, SelectVector, DeselectVector, ClearAllVectors, DuplicateVector

### state/reducers/reducer_matrices.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `reduce_matrices` | 14-115 | Handle matrix actions | state, action, with_history | Returns new state or None | MatrixData, np.dot, dataclasses.replace |

**Handled actions:** AddMatrix, DeleteMatrix, UpdateMatrixCell, UpdateMatrix, SelectMatrix, ApplyMatrixToSelected, ApplyMatrixToAll, ToggleMatrixPlot

### state/reducers/reducer_images.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `reduce_images` | 12-33 | Delegate to sub-reducers | state, action, with_history | Returns new state or None | reduce_image_load, reduce_image_kernel, reduce_image_transform, reduce_image_preprocess, reduce_image_basic |

### state/reducers/reducer_image_load.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `_auto_fit_scale` | 11-17 | Compute scale to fit grid | image_data, grid_size, margin | - | - |
| `reduce_image_load` | 20-90 | Handle LoadImage, CreateSampleImage | state, action, with_history | Returns new state | load_image, create_sample_image, ImageData.create, EducationalStep.create |

### state/reducers/reducer_image_kernel.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `reduce_image_kernel` | 11-71 | Handle ApplyKernel | state, action, with_history | Returns new state | apply_kernel, get_kernel_by_name, ImageData.create, EducationalStep.create |

### state/reducers/reducer_inputs.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `reduce_inputs` | 15-74 | Handle all SetInput* actions | state, action | Returns new state or None | dataclasses.replace |

### state/reducers/reducer_navigation.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `reduce_navigation` | 16-67 | Handle navigation/view actions | state, action | Returns new state or None | dataclasses.replace |

### state/reducers/reducer_history.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `reduce_history` | 21-39 | Handle Undo/Redo | state, action | Returns restored state or None | dataclasses.replace |
| `should_record_history` | 42-52 | Determine if action should be recorded | action | - | isinstance() |

### state/selectors/__init__.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `get_vector_by_id` | 12-17 | Find vector by ID | state, id | - | - |
| `get_matrix_by_id` | 20-25 | Find matrix by ID | state, id | - | - |
| `get_selected_vector` | 28-32 | Get currently selected vector | state | - | get_vector_by_id |
| `get_selected_matrix` | 35-39 | Get currently selected matrix | state | - | get_matrix_by_id |
| `get_current_step` | 42-46 | Get current pipeline step | state | - | - |
| `COLOR_PALETTE` | 49-58 | Tuple of 8 RGB colors | - | - | - |
| `get_next_color` | 61-65 | Get next color and new index | state | - | - |

---

## 2.3 Domain Layer (`domain/`)

### domain/images/image_matrix.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class ImageMatrix` | 9-116 | Image as mathematical matrix | - | - | - |
| `ImageMatrix.__init__` | 14-22 | Initialize from numpy array | data, name | self.data, self.name, self._original, self.history | np.array |
| `ImageMatrix.shape/height/width/channels` | 24-40 | Property accessors | - | - | - |
| `ImageMatrix.is_grayscale/is_rgb` | 42-48 | Type checks | - | - | - |
| `ImageMatrix.as_matrix` | 50-55 | Convert to grayscale matrix | - | - | - |
| `ImageMatrix.get_channel` | 57-65 | Extract single channel | channel | - | - |
| `ImageMatrix.get_rgb_planes` | 67-76 | Split into R, G, B | - | - | - |
| `ImageMatrix.to_grayscale` | 78-83 | Return grayscale copy | - | - | ImageMatrix() |
| `ImageMatrix.get_pixel_region` | 85-89 | Extract neighborhood | row, col, size | - | np.pad |
| `ImageMatrix.reset` | 91-93 | Restore original | - | self.data, self.history | - |
| `ImageMatrix.apply_transform` | 95-96 | Record transformation | matrix, name | self.history | - |
| `ImageMatrix.get_statistics` | 98-107 | Compute mean, std, min, max | - | - | np.mean, np.std, etc. |

### domain/images/image_samples.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `create_sample_image` | 10-58 | Generate test patterns | size, pattern | - | ImageMatrix(), np operations |

**Supported patterns:** 'gradient', 'checkerboard', 'circle', 'edges', 'noise', 'rgb_gradient'

### domain/images/io/image_loader.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `PIL_AVAILABLE` | 14-16 | Global flag for Pillow | - | - | - |
| `load_image` | 19-49 | Load image from disk | path, max_size?, grayscale? | - | PIL.Image.open, ImageMatrix() |

### domain/images/kernels/kernels.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `KERNEL_REGISTRY` | 17-31 | Dict mapping names to kernels | - | - | - |
| `KERNEL_DESCRIPTIONS` | 33-47 | Dict mapping names to descriptions | - | - | - |
| `get_kernel_by_name` | 50-52 | Retrieve kernel array | name | - | - |
| `list_kernels` | 55-58 | List all kernels with descriptions | - | - | - |
| `create_gaussian_kernel` | 61-74 | Generate Gaussian kernel | size, sigma | - | np operations |
| `kernel_to_string` | 77-83 | Format kernel for display | kernel, precision | - | - |
| `visualize_kernel_weights` | 86-110 | ASCII visualization | kernel | - | - |

**Available kernels:** sobel_x, sobel_y, laplacian, edge_detect, box_blur, gaussian_blur, gaussian_blur_5x5, sharpen, emboss, identity, prewitt_x, prewitt_y, ridge_detect

### domain/images/convolution/convolution_core.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `convolve2d` | 12-31 | Apply 2D convolution | image, kernel, mode, boundary | - | scipy.signal.correlate2d |
| `apply_kernel` | 34-56 | Apply named kernel to ImageMatrix | image_matrix, kernel_name, normalize_output | - | get_kernel_by_name, convolve2d, ImageMatrix() |

### domain/images/transforms/image_transforms.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `apply_affine_transform` | 13-59 | Apply affine transform to image | image_matrix, transform, output_shape?, order? | - | scipy.ndimage.affine_transform, ImageMatrix() |
| `normalize_image` | 62-82 | Normalize pixel values | image_matrix, mean?, std? | - | ImageMatrix() |

### domain/transforms/affine_transform.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class AffineTransform` | 17-84 | Composable affine transformation | - | - | - |
| `AffineTransform.__init__` | 22-26 | Initialize with identity or matrix | matrix? | self.matrix | np.eye |
| `AffineTransform.rotate` | 28-31 | Add rotation | angle, center? | self.matrix | create_rotation_matrix |
| `AffineTransform.scale` | 33-37 | Add scaling | sx, sy?, center? | self.matrix | create_scale_matrix |
| `AffineTransform.translate` | 39-42 | Add translation | tx, ty | self.matrix | create_translation_matrix |
| `AffineTransform.shear` | 44-47 | Add shear | shx, shy | self.matrix | create_shear_matrix |
| `AffineTransform.flip_horizontal` | 49-52 | Add horizontal flip | width | self.matrix | create_flip_matrix |
| `AffineTransform.flip_vertical` | 54-57 | Add vertical flip | height | self.matrix | create_flip_matrix |
| `AffineTransform.compose` | 59-62 | Combine with another transform | other | - | - |
| `AffineTransform.inverse` | 64-67 | Compute inverse | - | - | np.linalg.inv |
| `AffineTransform.transform_point` | 69-72 | Transform single point | x, y | - | - |
| `AffineTransform.transform_points` | 74-78 | Transform point array | points | - | - |
| `AffineTransform.get_matrix_2x3` | 80-81 | Get 2x3 matrix | - | - | - |

### domain/transforms/affine_matrices.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `create_rotation_matrix` | 9-25 | Create 3x3 rotation matrix | angle, center? | - | np operations |
| `create_scale_matrix` | 28-45 | Create 3x3 scale matrix | sx, sy?, center? | - | np operations |
| `create_translation_matrix` | 48-53 | Create 3x3 translation matrix | tx, ty | - | np.array |
| `create_shear_matrix` | 56-61 | Create 3x3 shear matrix | shx, shy | - | np.array |
| `create_flip_matrix` | 64-75 | Create 3x3 flip matrix | horizontal?, vertical?, width?, height? | - | np.array |

### domain/vectors/vector3d.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class Vector3D` | 5-97 | Mutable 3D vector class | - | - | - |
| `Vector3D.__init__` | 6-17 | Initialize vector | coords, color?, label?, visible?, metadata? | All instance vars | np.array |
| `Vector3D.normalize` | 19-25 | Normalize in-place | - | self.coords, self.history | np.linalg.norm |
| `Vector3D.scale` | 27-31 | Scale in-place | factor | self.coords, self.history | - |
| `Vector3D.magnitude` | 33-35 | Return length | - | - | np.linalg.norm |
| `Vector3D.dot` | 37-39 | Dot product | other | - | np.dot |
| `Vector3D.cross` | 41-44 | Cross product | other | - | np.cross, Vector3D() |
| `Vector3D.angle` | 46-59 | Angle between vectors | other, degrees? | - | np.arccos, np.degrees |
| `Vector3D.project_onto` | 61-68 | Project onto another | other | - | Vector3D() |
| `Vector3D.transform` | 70-73 | Apply matrix transformation | matrix | - | np.dot, Vector3D() |
| `Vector3D.reset` | 75-78 | Restore original | - | self.coords, self.history | - |
| `Vector3D.copy` | 80-88 | Create copy | - | - | Vector3D() |
| `Vector3D.to_list` | 90-92 | Convert to Python list | - | - | - |

### domain/vectors/vector_ops.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `add` | 9-11 | Vector addition | a, b | - | np operations |
| `scale` | 14-16 | Vector scaling | v, s | - | np operations |
| `dot` | 19-21 | Dot product | a, b | - | np.dot |
| `cross` | 24-26 | Cross product | a, b | - | np.cross |
| `normalize` | 29-34 | Normalize vector | v | - | np.linalg.norm |
| `angle_between` | 37-52 | Angle between vectors | a, b, degrees? | - | np.arccos |
| `project` | 55-62 | Vector projection | v, onto | - | dot() |
| `reflect` | 65-67 | Reflect across plane | v, normal | - | dot() |
| `lerp` | 70-72 | Linear interpolation | a, b, t | - | - |
| `distance` | 75-77 | Distance between points | a, b | - | np.linalg.norm |
| `gram_schmidt` | 80-95 | Gram-Schmidt orthogonalization | vectors | - | project, np.linalg.norm |
| `matrix_multiply` | 98-100 | Matrix multiplication | A, B | - | np.dot |
| `matrix_inverse` | 103-108 | Matrix inverse | A | - | np.linalg.inv |
| `matrix_determinant` | 111-113 | Matrix determinant | A | - | np.linalg.det |
| `eigen_decomposition` | 116-122 | Eigenvalue decomposition | A | - | np.linalg.eig |
| `solve_linear_system` | 125-131 | Solve Ax = b | A, b | - | np.linalg.solve |
| `qr_decomposition` | 134-137 | QR decomposition | A | - | np.linalg.qr |
| `svd_decomposition` | 140-143 | SVD decomposition | A | - | np.linalg.svd |

---

## 2.4 Engine Layer (`engine/`)

### engine/scene_adapter.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class RendererVector` | 24-42 | Vector representation for renderer | - | - | - |
| `RendererVector.from_vector_data` | 36-42 | Convert from VectorData | v: VectorData | - | np.array |
| `class RendererMatrix` | 45-61 | Matrix representation for renderer | - | - | - |
| `RendererMatrix.from_matrix_data` | 56-61 | Convert from MatrixData | m: MatrixData | - | np.array |
| `class SceneAdapter` | 64-178 | Read-only adapter from AppState | - | - | - |
| `SceneAdapter.__init__` | 82-143 | Build adapter from state | state: AppState | All _* instance vars | RendererVector.from_vector_data, etc. |
| `SceneAdapter.vectors` | 145-148 | Property: list of RendererVector | - | - | - |
| `SceneAdapter.matrices` | 150-153 | Property: list of dicts | - | - | - |
| `SceneAdapter.planes` | 155-158 | Property: list of dicts | - | - | - |
| `SceneAdapter.selected_object` | 160-163 | Property: selected item | - | - | - |
| `SceneAdapter.selection_type` | 165-168 | Property: selection type | - | - | - |
| `SceneAdapter.preview_matrix` | 170-173 | Property: preview matrix | - | - | - |
| `SceneAdapter.show_matrix_plot` | 175-178 | Property: matrix plot flag | - | - | - |
| `create_scene_from_state` | 181-187 | Factory function | state: AppState | - | SceneAdapter() |

### engine/execution_loop.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class FrameTimer` | 6-15 | Track frame timing | - | - | - |
| `FrameTimer.__init__` | 7-8 | Initialize timer | - | self._last_time | time.time() |
| `FrameTimer.tick` | 10-15 | Return delta time | - | self._last_time | time.time() |

### engine/picking_system.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `pick_vector` | 8-52 | Pick vector by screen coords | screen_x, screen_y, width, height, camera, vectors, radius_px | - | camera.world_to_screen |
| `pick_object` | 55-69 | Pick any object | screen_x, screen_y, width, height, camera, scene | - | pick_vector |
| `ray_intersect_plane` | 72-97 | Ray-plane intersection | ray_origin, ray_dir, plane_eq | - | np.dot |
| `get_nearest_point_on_line` | 100-124 | Nearest point on line segment | point, line_start, line_end | - | np.linalg.norm, np.dot, np.clip |

### engine/history_manager.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `can_undo` | 6-7 | Check if undo available | state: AppState | - | - |
| `can_redo` | 10-11 | Check if redo available | state: AppState | - | - |

---

## 2.5 Render Layer (`render/`)

### render/renderers/renderer.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class Renderer` | 20-131 | Main 3D renderer | - | - | - |
| `Renderer.__init__` | 21-52 | Initialize renderer | ctx, camera, view? | All instance vars | Gizmos(), moderngl.enable |
| `Renderer._get_view_projection` | 54-59 | Get cached VP matrix | - | self._vp_cache | camera.vp() |
| `Renderer.update_view` | 61-76 | Update view config | view_config | self.view, self._vp_cache_dirty, self.vector_scale, etc. | - |
| `Renderer.render` | 78-103 | Main render method | scene, image_data?, ... | - | _clear_with_gradient, _render_*_environment, draw_image_plane, _render_*_visuals |
| `Renderer._clear_with_gradient` | 105-116 | Clear with background color | - | - | ctx.clear |

### render/cameras/camera.py

| Class | Description |
|-------|-------------|
| `Camera` | Composed from multiple modules: camera_core, camera_controls, camera_projection |

### render/cameras/camera_core.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `__init__` | 14-45 | Initialize camera | self | All camera instance vars | - |
| `set_viewport` | 48-51 | Set viewport dimensions | self, width, height | self.width, self.height, self.aspect | - |
| `position` | 54-74 | Compute camera position | self | self.phi (clamped) | np operations |
| `vp` | 77-99 | Compute view-projection matrix | self | - | Matrix44.look_at, Matrix44.perspective_projection |
| `_get_2d_up_vector` | 102-111 | Get up vector for 2D mode | self | - | - |

### render/cameras/camera_controls.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `orbit` | 10-30 | Orbit camera around target | self, dx, dy | self.theta, self.phi, self.last_cubic_rotation | - |
| `pan` | 33-54 | Pan camera target | self, dx, dy | self.target, self.target_smooth | Vector3 operations |
| `zoom` | 57-70 | Zoom in/out | self, scroll_y | self.radius or self.ortho_scale | np.clip |
| `set_view_preset` | 73-103 | Set view preset | self, preset | self.view_preset, self.theta, self.phi, self.up_vector, self.cubic_mode, self.mode_2d | - |
| `cubic_view_rotation` | 106-115 | Auto-rotate in cubic view | self, auto_rotate, speed | self.theta | time.time() |
| `reset` | 118-129 | Reset camera to defaults | self | All camera vars | - |
| `focus_on_vector` | 132-143 | Focus camera on vector | self, vector_coords | self.target, self.target_smooth, self.radius, self.phi, self.theta | np.linalg.norm |

### render/gizmos/gizmos.py

| Class | Description |
|-------|-------------|
| `Gizmos` | Immediate-mode drawing, composed from multiple gizmo modules |

**Key methods:** draw_lines, draw_triangles, draw_points, draw_volume, draw_cubic_grid, draw_cube, draw_vector_with_details, draw_vector_span, draw_parallelepiped, draw_basis_transform, draw_grid, draw_axes

### render/viewconfigs/viewconfig.py

| Class | Description |
|-------|-------------|
| `ViewConfig` | View configuration, composed from multiple viewconfig modules |

**Key properties:** up_axis, grid_mode, grid_plane, grid_size, major_tick, minor_tick, show_grid, show_axes, show_labels, show_cube_faces, show_cube_corners, cubic_grid_density, vector_scale

---

## 2.6 UI Layer (`ui/`)

### ui/layout/workspace.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class WorkspaceLayout` | 29-84 | Main UI layout manager | - | - | - |
| `WorkspaceLayout.__init__` | 30-36 | Initialize panels | - | All panel instances | Toolbar(), ToolPalette(), Sidebar(), Inspector(), TimelinePanel() |
| `WorkspaceLayout.render` | 38-84 | Render full UI | state, dispatch, camera, view_config, app | self._last_theme | apply_theme, toolbar.render, tool_palette.render, operations_panel.render, inspector.render, timeline.render |

### ui/panels/sidebar/sidebar.py

| Class | Description |
|-------|-------------|
| `Sidebar` | Operations panel, composed from 20+ sidebar modules |

**Key methods:** _render_vector_creation, _render_vector_operations, _render_vector_list, _render_matrix_operations, _render_linear_systems, _render_visualization_options, _render_image_operations, _render_export_dialog

### ui/panels/sidebar/sidebar_state.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `sidebar_init` | 14-68 | Initialize sidebar state | self | All sidebar instance vars | - |

**Sidebar instance variables:**
- `active_tab` - Fallback tab
- `equation_count`, `equation_input` - Linear system editor
- `show_equation_editor`, `show_export_dialog` - Dialog visibility
- `vector_list_filter` - Filter string
- `_cell_buffers`, `_cell_active` - Matrix editing buffers
- `color_palette`, `next_color_idx` - Color cycling
- `selected_matrix_idx` - Local selection
- `_state`, `_dispatch` - Runtime references
- `scale_factor` - UI scaling
- `current_operation`, `operation_result` - Operation display
- `convolution_position` - Convolution visualization

### ui/panels/sidebar/sidebar_render.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `render` | 18-104 | Main sidebar render | self, rect, camera, view_config, state, dispatch | self._state, self._dispatch, self.scene | All _render_* methods based on active_tab |

### ui/panels/sidebar/sidebar_vector_creation.py

| Function | Line | Description | Inputs | State Mutated | Calls |
|----------|------|-------------|--------|---------------|-------|
| `_render_vector_creation` | 12-71 | Render vector creation form | self | - | dispatch(SetInputVector), dispatch(AddVector) |

### ui/toolbars/toolbar.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class Toolbar` | 8-136 | Top toolbar | - | - | - |
| `Toolbar.__init__` | 9-23 | Initialize toolbar | - | self._theme_items, self._theme_map, self._tabs | - |
| `Toolbar._render_menu_bar` | 25-65 | Render File/Edit/View/Window/Help menus | state, dispatch, view_config, app | view_config via .update() | dispatch(Undo), dispatch(Redo), imgui.* |
| `Toolbar._render_options_bar` | 67-131 | Render tab bar and controls | state, dispatch, view_config | - | dispatch(SetActiveTab), dispatch(Undo), dispatch(Redo), dispatch(SetTheme) |
| `Toolbar.render` | 133-136 | Main render | state, dispatch, camera, view_config, app | - | _render_menu_bar, _render_options_bar |

### ui/inspectors/inspector.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class Inspector` | 22-75 | Right-side inspector panel | - | - | - |
| `Inspector.__init__` | 23-25 | Initialize inspector | - | self.show_transform_history, self.show_computed_properties | - |
| `Inspector.render` | 27-69 | Render inspector | state, dispatch, rect | - | get_selected_vector, _render_coordinate_editor, _render_properties, _render_transform_history, _render_computed_properties |

### ui/panels/timeline/timeline_panel.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class TimelinePanel` | 14-61 | Bottom timeline panel | - | - | - |
| `TimelinePanel.__init__` | 15-16 | Initialize panel | - | self._step_filter | - |
| `TimelinePanel.render` | 18-61 | Render timeline | rect, state, dispatch | self._step_filter | dispatch(StepForward), dispatch(StepBackward), dispatch(JumpToStep) |

### ui/panels/tool_palette/tool_palette.py

| Function/Class | Line | Description | Inputs | State Mutated | Calls |
|----------------|------|-------------|--------|---------------|-------|
| `class ToolPalette` | 15-58 | Left-side tool palette | - | - | - |
| `ToolPalette.__init__` | 16-25 | Initialize tools | - | self._tools | - |
| `ToolPalette.render` | 27-58 | Render tool buttons | rect, state, dispatch | - | dispatch(SetActiveTool) |

**Available tools:** Select, Move, Rotate, Add Vector, Add Matrix, Image, Pipeline

---

# 3. FEATURE INVENTORY

## 3.1 User-Visible Features (Verified from Code)

### Vectors Tab (`active_tab == "vectors"`)

| Feature | Implementation | State Fields |
|---------|----------------|--------------|
| Create vector with coords (x, y, z) | `sidebar_vector_creation.py` | `input_vector_coords` |
| Set vector label | `sidebar_vector_creation.py` | `input_vector_label` |
| Pick vector color | `sidebar_vector_creation.py` | `input_vector_color` |
| List all vectors | `sidebar_vector_list.py` | `state.vectors` |
| Filter vectors by name | `sidebar_vector_list.py` | `self.vector_list_filter` |
| Select vector (click in scene) | `app_handlers.py` | `selected_id`, `selected_type` |
| Delete selected vector | `sidebar_vector_operations.py` | via `DeleteVector` action |
| Duplicate vector | Reducer supports it | via `DuplicateVector` action |
| Edit vector coordinates | `inspector_coordinates.py` | via `UpdateVector` action |
| View computed properties (magnitude, angle) | `inspector_computed.py` | - |

### Matrices Tab (`active_tab == "matrices"`)

| Feature | Implementation | State Fields |
|---------|----------------|--------------|
| Create matrix (2x2, 3x3, 4x4) | `sidebar_matrix_section.py` | `input_matrix`, `input_matrix_size` |
| Set matrix label | `sidebar_matrix_section.py` | `input_matrix_label` |
| Edit matrix cells | `sidebar_matrix_ops.py` | via `SetInputMatrixCell` |
| Apply matrix to selected vector | `sidebar_matrix_ops.py` | via `ApplyMatrixToSelected` |
| Apply matrix to all vectors | `sidebar_matrix_ops.py` | via `ApplyMatrixToAll` |
| Compute null space | `sidebar_matrix_ops.py` | Local computation |
| Compute column space | `sidebar_matrix_ops.py` | Local computation |
| Toggle matrix 3D plot | Reducer supports | `matrix_plot_enabled` |

### Systems Tab (`active_tab == "systems"`)

| Feature | Implementation | State Fields |
|---------|----------------|--------------|
| Enter system of equations | `sidebar_linear_systems_section.py` | `self.equation_input` |
| Resize equation system | `sidebar_linear_system_ops.py` | `self.equation_count` |
| Solve linear system | `sidebar_linear_system_ops.py` | Uses `np.linalg.solve` |
| Add solution vectors to scene | `sidebar_linear_system_ops.py` | via `AddVector` |

### Images Tab (`active_tab == "images"`)

| Feature | Implementation | State Fields |
|---------|----------------|--------------|
| Load image from file | `images_source_section.py` | `input_image_path`, `current_image` |
| Generate sample pattern | `images_source_section.py` | `input_sample_pattern`, `input_sample_size` |
| View image info (size, channels) | `images_info_section.py` | `current_image.shape` |
| Apply convolution kernel | `images_convolution_section.py` | `selected_kernel`, `processed_image` |
| Select from 13 kernels | `images_convolution_section.py` | `selected_kernel` |
| Apply affine transform (rotation, scale) | `images_transform_section.py` | `input_transform_rotation`, `input_transform_scale` |
| Flip image horizontally | Reducer supports | via `FlipImageHorizontal` |
| Use result as input | `images_result_section.py` | via `UseResultAsInput` |
| Clear images | `images_result_section.py` | via `ClearImage` |
| Normalize image | Reducer supports | `input_image_normalize_mean/std` |
| Toggle image on grid | `sidebar_render.py` | `show_image_on_grid` |
| Set render scale | Navigation reducer | `image_render_scale` |
| Set render mode (plane/height-field) | Navigation reducer | `image_render_mode` |
| Set color mode (rgb/grayscale/heatmap) | Navigation reducer | `image_color_mode` |
| Toggle pixel grid overlay | Navigation reducer | `show_image_grid_overlay` |
| Enable downsampling | Navigation reducer | `image_downsample_enabled` |
| View educational steps | `images_educational_section.py` | `pipeline_steps` |

### View/Visualize Tab (`active_tab == "visualize"`)

| Feature | Implementation | State Fields |
|---------|----------------|--------------|
| Toggle grid display | `toolbar.py`, `sidebar_visualization_section.py` | `view_config.show_grid` |
| Toggle axes display | `toolbar.py` | `view_config.show_axes` |
| Toggle labels display | `toolbar.py` | `view_config.show_labels` |
| Switch grid mode (cube/plane) | `app_handlers.py` (C key) | `view_config.grid_mode` |
| Toggle cube faces | `app_handlers.py` (F key) | `view_config.show_cube_faces` |
| Toggle vector components | `app_handlers.py` (V key) | `renderer.show_vector_components` |
| Auto-rotate camera | `app_handlers.py` (R key) | `view_config.auto_rotate` |

### Global Features

| Feature | Implementation | State Fields |
|---------|----------------|--------------|
| Undo/Redo | `toolbar.py`, `reducer_history.py` | `history`, `future` |
| Theme switching (Dark/Light/High Contrast) | `toolbar.py`, `theme_manager.py` | `ui_theme` |
| Camera orbit (right-drag) | `app_handlers.py` | `camera.theta`, `camera.phi` |
| Camera zoom (scroll) | `app_handlers.py` | `camera.radius` |
| Camera reset (Space) | `app_handlers.py` | Camera defaults |
| Export (JSON/CSV/Python) | `sidebar_export.py` | - |
| FPS display | `toolbar.py` | `app.fps` |

### Tool Palette

| Tool | ID | Purpose |
|------|-----|---------|
| Select | `select` | Default selection mode |
| Move | `move` | Move objects |
| Rotate | `rotate` | Rotate objects |
| Add Vector | `add_vector` | Quick vector creation |
| Add Matrix | `add_matrix` | Quick matrix creation |
| Image | `image` | Image operations |
| Pipeline | `pipeline` | Pipeline visualization |

---

# 4. STATE OWNERSHIP & DATA FLOW

## 4.1 Where Matrices Are Created

| Location | Method | Storage |
|----------|--------|---------|
| `sidebar_matrix_section.py` | User fills `input_matrix`, clicks "Create Matrix" | `state.matrices` via `AddMatrix` action |
| `sidebar_matrix_ops.py` | Preset transformations (rotation, scale) | `state.matrices` via `AddMatrix` action |
| `reduce_matrices.py:17-30` | `AddMatrix` handler creates `MatrixData` | `state.matrices` (immutable tuple) |

## 4.2 Where Vectors Live

| Location | Purpose |
|----------|---------|
| `state.vectors: Tuple[VectorData, ...]` | **Authoritative storage** (in AppState) |
| `SceneAdapter._vectors: List[RendererVector]` | **Read-only rendering copy** (recreated each frame) |
| `domain/vectors/Vector3D` | **Domain class** (not used in state, used for operations) |

## 4.3 Where Images Live

| Location | Purpose |
|----------|---------|
| `state.current_image: Optional[ImageData]` | **Source image** |
| `state.processed_image: Optional[ImageData]` | **Result after operations** |
| `domain/images/ImageMatrix` | **Domain class** (used in reducers, not stored directly) |

## 4.4 Where "Current State" Is Stored

```
App.store: Store
    └── Store._state: AppState  ← SINGLE SOURCE OF TRUTH
```

All application state lives in `Store._state`. This is the only mutable reference to an `AppState` object. When an action is dispatched, `_state` is replaced with a new `AppState` instance.

## 4.5 Critical State Ownership Questions

### Does state live in the GUI layer?

**Partial YES.** Some transient UI state lives in the GUI layer:

| Location | Type | Example |
|----------|------|---------|
| `Sidebar.equation_input` | Local mutable | Linear system editor |
| `Sidebar._cell_buffers` | Local mutable | ImGui text input buffers |
| `Sidebar.vector_list_filter` | Local mutable | Filter string |
| `Inspector.show_transform_history` | Local mutable | Panel visibility |
| `TimelinePanel._step_filter` | Local mutable | Filter string |

**However:** All domain data (vectors, matrices, images) lives in `AppState`.

### Are there global variables or singletons?

**YES:**

| Global | Location | Purpose |
|--------|----------|---------|
| `DEBUG` | `app/app_logging.py:8` | Debug mode flag from env |
| `PIL_AVAILABLE` | `domain/images/io/image_loader.py:14` | Pillow availability |
| `KERNEL_REGISTRY` | `domain/images/kernels/kernels.py:17` | Kernel dictionary |
| `KERNEL_DESCRIPTIONS` | `domain/images/kernels/kernels.py:33` | Kernel descriptions |
| `COLOR_PALETTE` | `state/selectors/__init__.py:49` | Color palette tuple |

**None of these store mutable application state.**

### Do UI callbacks perform math directly?

**NO for state-changing operations.** Math operations flow through the reducer:

```
UI Widget → dispatch(Action) → Reducer → Pure math in reducer → New State
```

Example flow for `ApplyMatrixToSelected`:
1. UI button calls `dispatch(ApplyMatrixToSelected(matrix_id))`
2. `reduce_matrices.py:66-87` handles action
3. Reducer performs `mat_np @ vec_np` using numpy
4. Returns new state with transformed coordinates

**YES for read-only computations.** Some UI components compute derived values:
- `inspector_computed.py` computes magnitude, angle for display
- `sidebar_matrix_ops.py` computes null space, column space for display
- These don't mutate state.

---

# 5. DEPENDENCY GRAPH

## 5.1 Module Import Structure

```
main.py
└── app/app.py
    ├── glfw, moderngl, imgui (external)
    ├── render/renderers/renderer.py
    │   ├── render/viewconfigs/viewconfig.py
    │   ├── render/gizmos/gizmos.py
    │   └── render/renderers/renderer_*.py (6 files)
    ├── render/cameras/camera.py
    │   └── render/cameras/camera_*.py (3 files)
    ├── render/renderers/labels/labels.py
    ├── ui/layout/workspace.py
    │   ├── ui/toolbars/toolbar.py
    │   ├── ui/panels/sidebar/sidebar.py
    │   │   └── ui/panels/sidebar/sidebar_*.py (20+ files)
    │   ├── ui/panels/tool_palette/tool_palette.py
    │   ├── ui/inspectors/inspector.py
    │   │   └── ui/inspectors/inspector_*.py (5 files)
    │   └── ui/panels/timeline/timeline_panel.py
    ├── state/__init__.py
    │   ├── state/store.py
    │   ├── state/app_state.py
    │   ├── state/models/*.py (6 files)
    │   ├── state/actions/*.py (6 files)
    │   ├── state/reducers/*.py (12 files)
    │   └── state/selectors/__init__.py
    ├── app/app_handlers.py
    │   └── engine/picking_system.py
    ├── app/app_run.py
    │   ├── app/app_state_bridge.py
    │   │   └── engine/scene_adapter.py
    │   └── engine/execution_loop.py
    └── app/app_style.py

domain/ (imported by reducers, not by app directly)
├── domain/images/__init__.py
│   ├── domain/images/image.py → image_matrix.py
│   ├── domain/images/image_samples.py
│   ├── domain/images/io/image_loader.py
│   ├── domain/images/kernels/kernels.py
│   │   └── domain/images/kernels/kernels_*.py (4 files)
│   ├── domain/images/convolution/convolution.py
│   │   └── domain/images/convolution/convolution_*.py (4 files)
│   └── domain/images/transforms/image_transforms.py
├── domain/transforms/affine_transform.py
│   └── domain/transforms/affine_matrices.py
└── domain/vectors/vector_ops.py
```

## 5.2 Key Coupling Points

### High Coupling (changes here affect many files)

| Module | Coupled To | Risk |
|--------|-----------|------|
| `state/app_state.py` | All reducers, all UI, scene_adapter | Very High - defines all state |
| `state/actions/__init__.py` | All reducers, all UI components | High - defines action vocabulary |
| `engine/scene_adapter.py` | Renderer, labels, picking | Medium - rendering interface |
| `state/store.py` | app.py, all dispatch callers | Medium - dispatch mechanism |

### Low Coupling (isolated)

| Module | Notes |
|--------|-------|
| `domain/images/*` | Pure functions, no state dependencies |
| `domain/transforms/*` | Pure functions, only numpy |
| `domain/vectors/*` | Pure functions, only numpy |
| `render/gizmos/*` | Only depends on OpenGL/moderngl |
| `render/shaders/*` | Only shader source strings |

## 5.3 Circular Dependencies

**None detected.** The import graph is acyclic:
- `state/` never imports from `ui/`
- `domain/` never imports from `state/`
- `engine/` imports from `state/` but not `ui/`
- `render/` never imports from `state/` directly

---

# 6. ARCHITECTURAL CHARACTERIZATION

## 6.1 Architecture Type: GUI-Driven with Redux State

The architecture is **primarily GUI-driven** with a Redux-style state management overlay:

```
┌─────────────────────────────────────────────────────────────────┐
│                         GUI LAYER (ImGui)                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Toolbar │ │ Sidebar │ │Inspector│ │Timeline │ │ToolPal │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       │           │           │           │           │         │
│       └───────────┴───────────┴───────────┴───────────┘         │
│                               │                                 │
│                        dispatch(Action)                         │
│                               │                                 │
│                               ▼                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    STATE LAYER (Redux)                     │ │
│  │  ┌──────┐    ┌─────────┐    ┌──────────┐                   │ │
│  │  │Store │───▶│ Reducer │───▶│ AppState │                   │ │
│  │  └──────┘    └─────────┘    └──────────┘                   │ │
│  │                    │                                        │ │
│  │              Pure functions                                 │ │
│  │              (domain/* for math)                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                               │                                 │
│                         get_state()                             │
│                               │                                 │
│                               ▼                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    RENDER LAYER                            │ │
│  │  ┌──────────────┐    ┌──────────┐    ┌────────┐            │ │
│  │  │SceneAdapter  │───▶│ Renderer │───▶│ OpenGL │            │ │
│  │  └──────────────┘    └──────────┘    └────────┘            │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 6.2 Where is "Truth" Currently Stored?

| Data Type | Truth Location | Access Pattern |
|-----------|----------------|----------------|
| Vectors | `AppState.vectors` | Read via `state.vectors`, modify via `dispatch(VectorAction)` |
| Matrices | `AppState.matrices` | Read via `state.matrices`, modify via `dispatch(MatrixAction)` |
| Images | `AppState.current_image`, `processed_image` | Read via `state.*`, modify via `dispatch(ImageAction)` |
| Selection | `AppState.selected_id`, `selected_type` | Read via selectors, modify via `dispatch(Select*)` |
| UI Input Forms | `AppState.input_*` | Read via `state.input_*`, modify via `dispatch(SetInput*)` |
| Camera | `App.camera` (Camera object) | Direct mutation via callbacks |
| View Config | `App.view_config` (ViewConfig object) | Direct mutation via `view_config.update()` |
| Sidebar Transients | `Sidebar.*` (local vars) | Direct mutation |

**Inconsistency:** Camera and ViewConfig are mutable objects outside Redux, mutated directly by handlers and toolbar. This is intentional for performance (avoiding action overhead for high-frequency events like orbit/zoom).

## 6.3 Where Would Changes Be Hardest to Make?

### Very Hard

| Change | Reason |
|--------|--------|
| Add a new domain object type (e.g., "Curve") | Requires: new model, new actions (5+), new reducers, new SceneAdapter handling, new UI tabs/sections, new renderer logic |
| Change AppState structure | Breaks all reducers, all UI reads, scene adapter |
| Replace ImGui with different UI framework | ~80 UI files using ImGui directly |

### Medium Hard

| Change | Reason |
|--------|--------|
| Add new convolution kernel | Add to `KERNEL_REGISTRY`, `KERNEL_DESCRIPTIONS` |
| Add new action for existing type | Add action class, add reducer case, add UI dispatch call |
| Change rendering approach | Contained to `render/` layer |

### Easy

| Change | Reason |
|--------|--------|
| Add new sample image pattern | Add case in `create_sample_image()` |
| Add new vector operation | Add function in `domain/vectors/vector_ops.py` |
| Change UI colors/style | Modify `app_style.py` or `theme_manager.py` |
| Add keyboard shortcut | Add case in `on_key()` |

---

# 7. APPENDIX: FILE LISTING BY LAYER

## Application Layer (7 files)
```
main.py
app/app.py
app/app_handlers.py
app/app_logging.py
app/app_run.py
app/app_state_bridge.py
app/app_style.py
```

## State Layer (34 files)
```
state/__init__.py
state/store.py
state/app_state.py
state/models/__init__.py
state/models/vector_model.py
state/models/matrix_model.py
state/models/plane_model.py
state/models/image_model.py
state/models/educational_step.py
state/models/pipeline_models.py
state/actions/__init__.py
state/actions/vector_actions.py
state/actions/matrix_actions.py
state/actions/image_actions.py
state/actions/input_actions.py
state/actions/navigation_actions.py
state/actions/pipeline_actions.py
state/actions/history_actions.py
state/reducers/__init__.py
state/reducers/reducer_vectors.py
state/reducers/reducer_matrices.py
state/reducers/reducer_images.py
state/reducers/reducer_image_load.py
state/reducers/reducer_image_kernel.py
state/reducers/reducer_image_transform.py
state/reducers/reducer_image_preprocess.py
state/reducers/reducer_image_basic.py
state/reducers/reducer_inputs.py
state/reducers/reducer_navigation.py
state/reducers/reducer_pipeline.py
state/reducers/reducer_history.py
state/selectors/__init__.py
```

## Domain Layer (24 files)
```
domain/__init__.py
domain/images/__init__.py
domain/images/image.py
domain/images/image_matrix.py
domain/images/image_samples.py
domain/images/io/__init__.py
domain/images/io/image_loader.py
domain/images/kernels/__init__.py
domain/images/kernels/kernels.py
domain/images/kernels/kernels_blur.py
domain/images/kernels/kernels_edges.py
domain/images/kernels/kernels_sharpen.py
domain/images/kernels/kernels_utility.py
domain/images/convolution/__init__.py
domain/images/convolution/convolution.py
domain/images/convolution/convolution_core.py
domain/images/convolution/convolution_gradients.py
domain/images/convolution/convolution_multiscale.py
domain/images/convolution/convolution_visuals.py
domain/images/transforms/__init__.py
domain/images/transforms/image_transforms.py
domain/transforms/__init__.py
domain/transforms/affine_helpers.py
domain/transforms/affine_matrices.py
domain/transforms/affine_transform.py
domain/transforms/transforms.py
domain/vectors/__init__.py
domain/vectors/vector3d.py
domain/vectors/vector_ops.py
```

## Engine Layer (4 files)
```
engine/execution_loop.py
engine/history_manager.py
engine/picking_system.py
engine/scene_adapter.py
```

## Render Layer (39 files)
```
render/__init__.py
render/buffers/__init__.py
render/buffers/gizmo_buffers.py
render/cameras/__init__.py
render/cameras/camera.py
render/cameras/camera_controls.py
render/cameras/camera_core.py
render/cameras/camera_projection.py
render/gizmos/__init__.py
render/gizmos/gizmos.py
render/gizmos/gizmo_cubic_grid.py
render/gizmos/gizmo_draw_lines.py
render/gizmos/gizmo_draw_points.py
render/gizmos/gizmo_draw_triangles.py
render/gizmos/gizmo_draw_volume.py
render/gizmos/gizmo_planar_grid.py
render/gizmos/gizmo_vector_details.py
render/gizmos/gizmo_vector_visuals.py
render/renderers/__init__.py
render/renderers/renderer.py
render/renderers/renderer_axes.py
render/renderers/renderer_cubic_faces.py
render/renderers/renderer_environment.py
render/renderers/renderer_image.py
render/renderers/renderer_linear_algebra.py
render/renderers/renderer_vectors.py
render/renderers/labels/__init__.py
render/renderers/labels/label_axes.py
render/renderers/labels/label_grid.py
render/renderers/labels/label_vectors.py
render/renderers/labels/labels.py
render/shaders/__init__.py
render/shaders/gizmo_programs.py
render/viewconfigs/__init__.py
render/viewconfigs/viewconfig.py
render/viewconfigs/viewconfig_axis.py
render/viewconfigs/viewconfig_core.py
render/viewconfigs/viewconfig_cubic.py
render/viewconfigs/viewconfig_grid_basis.py
```

## UI Layer (60 files)
```
ui/__init__.py
ui/layout/__init__.py
ui/layout/workspace.py
ui/themes/__init__.py
ui/themes/theme_manager.py
ui/utils/__init__.py
ui/utils/imgui_compat.py
ui/inspectors/__init__.py
ui/inspectors/inspector.py
ui/inspectors/inspector_computed.py
ui/inspectors/inspector_coordinates.py
ui/inspectors/inspector_header.py
ui/inspectors/inspector_properties.py
ui/inspectors/inspector_transform_history.py
ui/panels/__init__.py
ui/panels/images/__init__.py
ui/panels/images/images_tab.py
ui/panels/images/images_tab_constants.py
ui/panels/images/images_convolution_section.py
ui/panels/images/images_educational_section.py
ui/panels/images/images_info_section.py
ui/panels/images/images_result_section.py
ui/panels/images/images_source_section.py
ui/panels/images/images_transform_section.py
ui/panels/sidebar/__init__.py
ui/panels/sidebar/sidebar.py
ui/panels/sidebar/sidebar_export.py
ui/panels/sidebar/sidebar_images_info.py
ui/panels/sidebar/sidebar_images_ops.py
ui/panels/sidebar/sidebar_images_result.py
ui/panels/sidebar/sidebar_images_section.py
ui/panels/sidebar/sidebar_images_source.py
ui/panels/sidebar/sidebar_images_transform.py
ui/panels/sidebar/sidebar_linear_system_ops.py
ui/panels/sidebar/sidebar_linear_systems_section.py
ui/panels/sidebar/sidebar_matrix_ops.py
ui/panels/sidebar/sidebar_matrix_section.py
ui/panels/sidebar/sidebar_render.py
ui/panels/sidebar/sidebar_state.py
ui/panels/sidebar/sidebar_utils.py
ui/panels/sidebar/sidebar_vector_creation.py
ui/panels/sidebar/sidebar_vector_list.py
ui/panels/sidebar/sidebar_vector_operations.py
ui/panels/sidebar/sidebar_vision.py
ui/panels/sidebar/sidebar_visualization_section.py
ui/panels/timeline/__init__.py
ui/panels/timeline/timeline_panel.py
ui/panels/tool_palette/__init__.py
ui/panels/tool_palette/tool_palette.py
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
ui/toolbars/__init__.py
ui/toolbars/toolbar.py
```

---

# END OF INVENTORY

This document provides a complete factual inventory of the CVLA codebase as it exists. No architectural recommendations or refactoring suggestions have been included per the original request.

**Key Takeaways:**
1. The application uses Redux-style state management with immutable AppState
2. All domain data flows through the Store/Reducer pattern
3. UI is built with ImGui in a Photoshop-style layout
4. Rendering is done via ModernGL with a SceneAdapter bridge
5. Domain logic (images, vectors, transforms) is pure and side-effect free
6. Camera and ViewConfig are the only objects mutated directly (not via Redux)
7. The codebase has 168 Python files across 6 major layers
