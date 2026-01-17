# CVLA Architecture Audit Report

**Auditor:** Senior Software Architect + Code Reviewer
**Date:** 2026-01-17
**Scope:** Identify what is unfinished, broken, or rough - NO new features

---

# 1. CURRENT STATE SUMMARY

## 1.1 What Works

| System | Status | Notes |
|--------|--------|-------|
| **Redux State Flow** | STABLE | Single source of truth, immutable AppState, actions/reducers work correctly |
| **Vector CRUD** | STABLE | Add, delete, update, select, duplicate all functional |
| **Matrix CRUD** | STABLE | Add, delete, update cells, select all functional |
| **Matrix Transforms** | STABLE | ApplyMatrixToSelected, ApplyMatrixToAll work correctly |
| **Image Loading** | STABLE | File load, sample patterns, downsampling work |
| **Kernel Convolution** | STABLE | All 13 kernels apply correctly |
| **Affine Transforms** | STABLE | Rotation, scale, flip work |
| **3D Rendering** | STABLE | Vectors, grid, axes, camera orbit/zoom |
| **Undo/Redo** | STABLE | History maintained for domain operations |
| **Tab Navigation** | STABLE | vectors/matrices/systems/images tabs work |
| **Theme Switching** | STABLE | dark/light/high-contrast themes work |

## 1.2 What Is Stable (but incomplete)

| System | Status | Notes |
|--------|--------|-------|
| **Educational Pipeline** | PARTIAL | Steps created but timeline displays empty content |
| **Linear Systems** | PARTIAL | Solve works, but equation editor is sidebar-local |
| **Image Visualization** | PARTIAL | Image displays on grid but render modes not fully wired |
| **Selectors** | COMPLETE | get_selected_vector, get_selected_matrix exist but matrix not used |

## 1.3 What Is Fragile

| System | Risk Level | Issue |
|--------|------------|-------|
| **Tool Palette** | HIGH | Visual-only, tools don't change behavior |
| **Inspector** | MEDIUM | Only handles vectors, ignores matrices |
| **Image Result Section** | MEDIUM | Null dereference if processed_image becomes None |
| **Error Handling** | MEDIUM | Silent failures in reducers |
| **History Exclusions** | LOW | Some view settings bloat undo stack |
| **OpenGL Resources** | LOW | Never cleaned up, potential memory leaks |

---

# 2. INCOMPLETE / BROKEN ITEMS (Checklist)

## 2.1 CRITICAL - UI Wiring Broken

### [ ] Tool Palette Does Nothing (HIGH)

**Files:** `ui/panels/tool_palette/tool_palette.py`, `app/app_handlers.py`

**Symptom:** Clicking tools (Move, Rotate, Add Vector, Add Matrix, Image, Pipeline) only changes highlight - no behavior change.

**Why broken:** `app_handlers.py:on_mouse_button` and `on_mouse_move` never check `state.active_tool`. The tool selection is purely cosmetic.

**Affected tools (7 total, 6 broken):**
- `Select` - Works (default behavior)
- `Move` - Does nothing (orbits camera instead of moving objects)
- `Rotate` - Does nothing (no rotation behavior implemented)
- `Add Vector` - Does nothing (should click to add vector)
- `Add Matrix` - Does nothing (should click to add matrix)
- `Image` - Does nothing (no image-specific interaction)
- `Pipeline` - Does nothing (no pipeline-specific interaction)

---

### [ ] Inspector Ignores Matrices (MEDIUM)

**Files:** `ui/inspectors/inspector.py:32`

**Symptom:** Selecting a matrix shows "No selection" in inspector.

**Why broken:** Inspector only calls `get_selected_vector()` (line 32), never `get_selected_matrix()`. The selector exists but is unused.

**Code evidence:**
```python
selected_vector = get_selected_vector(state)  # Only this is checked
# No: selected_matrix = get_selected_matrix(state)
```

---

### [ ] Potential Null Dereference in Image Sections (MEDIUM)

**Files:**
- `ui/panels/images/images_result_section.py:22-25`
- `ui/panels/images/images_info_section.py:24-32`

**Symptom:** If `state.processed_image` or `state.current_image` becomes None during render, app crashes.

**Why fragile:** Parent guard (images_tab.py) checks before calling section, but no local validation. Race condition possible if state changes mid-frame.

---

## 2.2 MEDIUM - State Flow Gaps

### [ ] Dead State Fields (10 fields)

**File:** `state/app_state.py`

| Field | Line | Status | Replacement |
|-------|------|--------|-------------|
| `show_heatmap` | 121 | DEAD | Use `image_color_mode` |
| `show_channels` | 122 | DEAD | Never implemented |
| `image_pipeline` | 57 | DEAD | Replaced by `pipeline_steps` |
| `active_pipeline_index` | 58 | DEAD | Replaced by `pipeline_step_index` |
| `micro_op` | 61 | DEAD | Never used |
| `micro_step_index` | 59 | DEAD | Never used |
| `micro_step_total` | 60 | DEAD | Never used |
| `image_step_index` | 63 | DEAD | Never used |
| `image_step_total` | 64 | DEAD | Never used |
| `input_mnist_index` | 104 | DEAD | MNIST never implemented |

**Impact:** State bloat, confusion about which fields are active.

---

### [ ] Missing History Exclusions (10 actions)

**File:** `state/reducers/reducer_history.py:44-52`

**Symptom:** View/render settings create undo history entries.

**Actions that should be excluded but aren't:**
1. `SetImageNormalizeMean`
2. `SetImageNormalizeStd`
3. `SetImageColorMode`
4. `SetImageRenderScale`
5. `SetImageRenderMode`
6. `SetImagePreviewResolution`
7. `ToggleImageGridOverlay`
8. `ToggleImageDownsample`
9. `ToggleImageOnGrid`
10. `SetActiveImageTab`

**Impact:** Undo stack fills with trivial view changes.

---

### [ ] Silent Failure in Reducers (3 locations)

**Files:**
- `state/reducers/reducer_image_kernel.py:47-49`
- `state/reducers/reducer_image_transform.py` (similar pattern)
- `state/reducers/reducer_image_preprocess.py` (similar pattern)

**Symptom:** Operations fail silently, user sees no error.

**Pattern:**
```python
except Exception as e:
    print(f"ApplyKernel error: {e}")  # Console only
    return state  # No status update
```

**Correct pattern (in reducer_image_load.py):**
```python
except Exception as e:
    return replace(state,
        image_status=f"Failed: {e}",
        image_status_level="error",
    )
```

---

## 2.3 LOW - Resource Management

### [ ] No OpenGL Resource Cleanup

**Files:**
- `render/gizmos/gizmos.py` - VBOs/VAOs created but never released
- `render/renderers/renderer.py` - Image plane cache accumulates
- `app/app.py` - No cleanup on exit

**Symptom:** Memory leak over extended use, especially with repeated image operations.

**Missing:**
- `__del__` methods or context managers
- Cache invalidation strategy
- Explicit `glDeleteBuffers` / `vao.release()` calls

---

### [ ] Image Cache Invalidation Not Verified

**File:** `render/renderers/renderer.py:34`

**Symptom:** Changing `image_render_scale` or `image_render_mode` may not invalidate cached image plane batches.

**Risk:** Stale rendering after settings change.

---

# 3. ARCHITECTURAL DEBT

## 3.1 Intent vs Implementation Divergence

### Tool System Design Intent
**Intent:** Tool palette should change interaction mode (move objects, add vectors on click, etc.)

**Reality:** Tool palette is a stateless visual indicator. The `active_tool` field exists in state but is never read by any handler.

**Divergence location:** `app/app_handlers.py` should dispatch different actions based on `state.active_tool` but doesn't.

---

### Inspector Design Intent
**Intent:** Inspector should show details for any selected object (vector, matrix, plane, image).

**Reality:** Inspector only handles vectors. Selector infrastructure exists for matrices but is unused.

**Divergence location:** `ui/inspectors/inspector.py:32-66` - hardcoded to vector-only workflow.

---

### Educational Pipeline Design Intent
**Intent:** Pipeline should provide step-through educational visualization of operations.

**Reality:** Pipeline steps are created but timeline shows only titles. No visualization of kernel sliding, convolution math, or intermediate states.

**Divergence location:** `EducationalStep` has fields for `input_data`, `output_data`, `kernel_values` but timeline doesn't render them.

---

## 3.2 Dual Systems Coexisting

### Color Palette Duplication
**Locations:**
- `state/selectors/__init__.py:49-58` - `COLOR_PALETTE`, `get_next_color()`
- `ui/panels/sidebar/sidebar_state.py:42-52` - `self.color_palette`, `self.next_color_idx`

**Problem:** Two color cycling systems exist. Sidebar has local palette that shadows state selector.

---

### Equation Storage Split
**Locations:**
- `state/app_state.py:96-101` - `input_equations`, `input_equation_count` in state
- `ui/panels/sidebar/sidebar_state.py` - `self.equation_input` local to sidebar

**Problem:** State fields exist but sidebar uses local storage. Linear system definition is domain data but inaccessible through Store.

---

## 3.3 Legacy Cruft

### App.selected (Dead Code)
**File:** `app/app.py:111` - `self.selected = None`

**Status:** Never read or written elsewhere. Pre-Redux legacy.

---

### Sidebar.scene (Dead Code)
**File:** `ui/panels/sidebar/sidebar_render.py:29` - `self.scene = None`

**Status:** Set to None every frame, never used.

---

# 4. MISSING WIRES

## 4.1 UI -> State Gaps

| UI Element | Expected Dispatch | Actual Behavior |
|------------|------------------|-----------------|
| Tool Palette: Move | Should enable move mode | Only highlights button |
| Tool Palette: Rotate | Should enable rotate mode | Only highlights button |
| Tool Palette: Add Vector | Should enable click-to-add | Only highlights button |
| Tool Palette: Add Matrix | Should enable matrix mode | Only highlights button |
| Inspector: Matrix selection | Should show matrix details | Shows "No selection" |

## 4.2 State -> Render Gaps

| State Field | Expected Effect | Actual Behavior |
|-------------|----------------|-----------------|
| `active_tool` | Changes mouse behavior | Not read by handlers |
| `show_heatmap` | Enables heatmap mode | Dead field |
| `show_channels` | Shows channel separation | Dead field |
| `micro_op` | Micro-step visualization | Dead field |

## 4.3 Partially Wired

| Path | Wired | Gap |
|------|-------|-----|
| Vector Create -> State -> Render | YES | Complete |
| Matrix Create -> State -> Render | YES | Complete |
| Matrix Select -> State -> Inspector | PARTIAL | State updates, inspector ignores |
| Image Load -> State -> Render | YES | Complete |
| Pipeline Step Create -> State -> Timeline | PARTIAL | Steps display, no visualization |
| Tool Select -> State -> Handlers | NO | State updates, handlers ignore |

---

# 5. CLEANUP PRIORITIES

## Priority 0 - Must Fix (Broken Functionality)

| # | Item | Risk | Effort | Files |
|---|------|------|--------|-------|
| 1 | Add null guards to image sections | HIGH | 5 min | images_result_section.py, images_info_section.py |
| 2 | Remove 10 dead state fields | MEDIUM | 15 min | app_state.py |
| 3 | Add status updates to failing reducers | MEDIUM | 15 min | reducer_image_kernel.py, reducer_image_transform.py, reducer_image_preprocess.py |

## Priority 1 - Should Fix (Incomplete Features)

| # | Item | Risk | Effort | Files |
|---|------|------|--------|-------|
| 4 | Wire tool palette OR remove tools | LOW | 2 hrs or 5 min | app_handlers.py OR tool_palette.py |
| 5 | Add matrix support to inspector | LOW | 45 min | inspector.py, new inspector_matrix_*.py files |
| 6 | Add missing history exclusions | LOW | 10 min | reducer_history.py |
| 7 | Unify color palette | LOW | 15 min | sidebar_state.py, sidebar_utils.py |

## Priority 2 - Nice to Fix (Technical Debt)

| # | Item | Risk | Effort | Files |
|---|------|------|--------|-------|
| 8 | Remove App.selected | NONE | 1 min | app/app.py |
| 9 | Remove Sidebar.scene | NONE | 1 min | sidebar_render.py |
| 10 | Add OpenGL cleanup | LOW | 30 min | gizmos.py, renderer.py, app.py |
| 11 | Verify cache invalidation | LOW | 20 min | renderer.py |

---

# 6. STOPPING POINT

## What "Done" Looks Like for CVLA v1 (No ML)

### Core Features Complete
- [x] Vector CRUD with 3D visualization
- [x] Matrix CRUD with transformation application
- [x] Image loading, kernels, transforms
- [x] Undo/redo for domain operations
- [x] Camera controls (orbit, zoom, reset)
- [x] Tab-based UI navigation
- [x] Theme switching

### Stability Requirements Met
- [ ] No null dereference crashes
- [ ] All silent failures converted to status messages
- [ ] Dead state fields removed
- [ ] History exclusions complete

### Architectural Consistency
- [ ] Tool palette either functional OR removed
- [ ] Inspector handles all selectable types OR clearly vector-only
- [ ] Single color palette source
- [ ] No dead code in state layer

### Documentation Updated
- [x] CODEBASE_INVENTORY.md current
- [x] ARCHITECTURE_HARDENING.md complete
- [ ] Remove stale references to planes, ribbon
- [ ] Document tool limitations if kept as cosmetic

### What Is NOT Required for v1
- ML/Keras integration (deferred)
- MNIST loading (dead code, remove)
- Micro-step visualization (infrastructure exists but not wired)
- Move/Rotate tools (can be cosmetic for v1)
- Full pipeline educational visualization

---

# APPENDIX A: Files Changed by Recent Refactor

Based on git status, the following have been modified or deleted:

**Modified:**
- `app/app.py`
- `engine/picking_system.py`
- `engine/scene_adapter.py`
- `state/__init__.py`
- `state/actions/__init__.py`
- `state/actions/input_actions.py`
- `state/app_state.py`
- `state/models/__init__.py`
- `state/reducers/reducer_image_kernel.py`
- `state/reducers/reducer_inputs.py`
- `ui/panels/sidebar/sidebar_*.py` (5 files)

**Deleted (per ARCHITECTURE_HARDENING decisions):**
- `domain/pipelines/__init__.py` - Empty placeholder
- `domain/scenes/solvers.py` - Orphaned mixin
- `state/models/plane_model.py` - Unused infrastructure
- `ui/ribbon/` (11 files) - Dead code, Toolbar kept

**Added:**
- `engine/image_adapter.py` - TempImageMatrix extracted
- `docs/ARCHITECTURE_HARDENING.md`
- `docs/CODEBASE_INVENTORY.md`

---

# APPENDIX B: Action/Reducer Coverage Matrix

All 54 actions have corresponding reducer handlers:

| Category | Actions | Reducer | Covered |
|----------|---------|---------|---------|
| Vector | 7 | reducer_vectors.py | YES |
| Matrix | 8 | reducer_matrices.py | YES |
| Image | 8 | reducer_images.py (delegates) | YES |
| Input | 17 | reducer_inputs.py | YES |
| Navigation | 8 | reducer_navigation.py | YES |
| Pipeline | 4 | reducer_pipeline.py | YES |
| History | 2 | reducer_history.py | YES |

No orphaned actions detected.

---

# APPENDIX C: Cross-Reference to ARCHITECTURE_HARDENING.md

| Hardening Item | Status | Notes |
|----------------|--------|-------|
| L1: Camera outside Redux | ACCEPTED | Documented, intentional |
| L2: ViewConfig outside Redux | ACCEPTED | Documented, intentional |
| L3: Sidebar local state | PARTIALLY FIXED | input_equations in state now |
| H1: Ribbon removed | COMPLETE | Files deleted |
| H2: Planes removed | COMPLETE | Files deleted |
| H3: SceneSolversMixin removed | COMPLETE | File deleted |
| H5: domain/pipelines removed | COMPLETE | Directory deleted |
| Patch 2.9: ImageDataAdapter | COMPLETE | engine/image_adapter.py created |

---

END OF AUDIT REPORT
