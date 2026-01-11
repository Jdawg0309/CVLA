Code Reference
==============

Below is a concise mapping of the main modules and functions in this repository. Each entry gives a short description of what the function or method does and how it is typically used in the application.

`main.py`
- `main()` : Application entry point. Instantiates `App` and calls `run()`; wraps startup in a try/except to print tracebacks.

`domain/vectors/vector_ops.py` (utility linear-algebra helpers)
- `add(a, b)`: Elementwise vector addition. Returns numpy array.
- `scale(v, s)`: Multiply vector `v` by scalar `s`.
- `dot(a, b)`: Dot product (returns float).
- `cross(a, b)`: Cross product (returns numpy array).
- `normalize(v)`: Return unit vector for `v` (safe-guards tiny magnitudes).
- `angle_between(a, b, degrees=True)`: Angle between two vectors; optional degrees output.
- `project(v, onto)`: Project vector `v` onto `onto`.
- `reflect(v, normal)`: Reflect `v` across plane with given normal.
- `lerp(a, b, t)`: Linear interpolation between `a` and `b`.
- `distance(a, b)`: Euclidean distance between points.
- `gram_schmidt(vectors)`: Orthonormalize collection of vectors.
- `matrix_multiply(A, B)`: Matrix product A·B.
- `matrix_inverse(A)`: Safe inverse, returns `None` if singular.
- `matrix_determinant(A)`: Determinant of a matrix.
- `eigen_decomposition(A)`: Returns (eigenvalues, eigenvectors) or (None, None) on failure.
- `solve_linear_system(A, b)`: Solve Ax=b using numpy solver (None if singular).
- `qr_decomposition(A)`: Return (Q, R).
- `svd_decomposition(A)`: Return (U, S, Vt).

`domain/vectors/vector3d.py` (Vector3D class)
- `Vector3D.__init__(coords, color, label, visible, metadata)`: Construct a 3D vector and validate length==3.
- `normalize()`: Normalize in-place and record in `history`.
- `scale(factor)`: Scale in-place and record history.
- `magnitude()`: Return vector norm.
- `dot(other)`, `cross(other)`: Vector algebra ops; `cross` returns a new `Vector3D`.
- `angle(other, degrees=True)`: Angle between vectors.
- `project_onto(other)`: Return projection as a new `Vector3D`.
- `transform(matrix)`: Apply matrix to vector and return a new `Vector3D`.
- `reset()`, `copy()`, `to_list()`, `__str__/__repr__()`: Utility methods for state and display.

`engine/scene_adapter.py` (SceneAdapter: AppState -> renderer compatibility)
- `SceneAdapter.__init__(state)`: Build a read-only adapter that exposes vectors/matrices/planes in renderer-friendly shapes.
- `selected_object`, `selection_type`, `preview_matrix`: Selection and preview data derived from AppState.
- `create_scene_from_state(state)`: Convenience factory for adapter creation.

`engine/execution_loop.py` (Frame timing helpers)
- `FrameTimer.tick()`: Return delta time between frames.

`engine/history_manager.py` (Undo/redo helpers)
- `can_undo(state)`, `can_redo(state)`: Simple history checks.

`render/cameras/camera.py` (Camera controls & matrices)
- `Camera.__init__()`: Initialize orbit parameters, viewport, mode flags, cubic-view tweaks.
- `set_viewport(width, height)`: Update aspect and viewport size.
- `position()`: Compute camera world position (sphere/2D variants).
- `vp()`: Return cached view-projection matrix (perspective or orthographic depending on mode).
- `_get_2d_up_vector()`: Helper for 2D presets.
- `orbit(dx, dy)`, `pan(dx, dy)`, `zoom(scroll_y)`: Interactive camera controls.
- `set_view_preset(preset)`: Apply named presets (`xy`, `xz`, `yz`, `cube`, `3d_free`).
- `cubic_view_rotation(auto_rotate, speed)`: Optional auto-rotation for demo mode.
- `reset()`, `focus_on_vector(coords)`, `world_to_screen(world_pos, width, height)`, `screen_to_ray(screen_x, screen_y, width, height)`, `get_view_matrix()`, `get_projection_matrix()`: Utility methods for coordinate conversions and camera state.

`render/renderers/renderer.py` (High-level rendering orchestration)
- `Renderer.__init__(ctx, camera, view)`: Create `Gizmos`, initialize rendering flags and GL state.
- `_get_view_projection()`, `update_view(view_config)`: VP caching and view updates.
- `render(scene)`: Top-level render routine. Clears background, draws cube/plane grid, linear algebra visuals, vectors, and selection highlight.
- Private helpers: `_clear_with_gradient()`, `_render_cubic_environment()`, `_render_planar_environment()`, `_render_cube_faces()`, `_render_cube_corner_indicators()`, `_render_3d_axes_with_depths()`, `_draw_axis_cones()`, `_render_linear_algebra_visuals()`, `_render_vectors_with_enhancements()`, `_render_vector_projections()`, `_render_selection_highlight()` — these assemble visuals by calling into `Gizmos` and composing geometries for the scene.

`engine/gizmos.py` (Low-level drawable primitives and shader glue)
- Constructor creates and caches shader programs for lines, triangles, points, and volumes.
- `_create_line_program()`, `_create_triangle_program()`, `_create_point_program()`, `_create_volume_program()`: Build GLSL programs used for different primitive types.
- `_init_buffers()`: Allocate VBOs/VAOs used across draws.
- Core draw calls: `draw_lines(vertices, colors, vp, width, depth)`, `draw_triangles(vertices, normals, colors, vp, ...)`, `draw_points(vertices, colors, vp, size, depth)`, `draw_volume(vertices, colors, vp, opacity, depth)` — convert Python lists to interleaved buffers and issue ModernGL draws.
- High-level helpers: `draw_cubic_grid(...)`, `draw_cube(...)`, `draw_vector_with_details(...)`, `_draw_arrow_head(...)`, `_draw_vector_components(...)`, `draw_vector_span(...)`, `draw_parallelepiped(...)`, `draw_basis_transform(...)`, `draw_grid(...)`, `draw_axes(...)` — convenience methods to build common visuals used by `Renderer`.

`engine/picking_system.py` (Mouse picking utilities)
- `pick_vector(screen_x, screen_y, width, height, camera, vectors, radius_px)`: Find nearest visible vector under the cursor, using `camera.world_to_screen` and simple radius tests.
- `pick_object(screen_x, screen_y, width, height, camera, scene)`: Generic picker wrapper (currently uses vector picking first).
- `ray_intersect_plane(ray_origin, ray_dir, plane_eq)`: Ray-plane intersection utility.
- `get_nearest_point_on_line(point, line_start, line_end)`: Project point onto line segment and clamp.

`render/renderers/labels/labels.py` (2D overlay labels rendered with ImGui draw lists)
- `LabelRenderer.__init__()` : Initialize styling and caches.
- `update_view(viewconfig)`: Update internal view config pointer.
- `world_to_screen(camera, world_pos, width, height)`: Convenience wrapper for camera projection.
- `draw_axes(camera, width, height)`, `draw_grid_numbers(camera, width, height, viewconfig, grid_size, major)`, `draw_vector_labels(camera, vectors, width, height, selected_vector)`: Draw axis labels, grid tick labels, and vector labels using ImGui's background draw list.
- Internal helpers: `_get_axis_label(axis)`, `_get_active_planes()`, `_get_grid_positions(plane, value, grid_size)`.

`render/viewconfigs/viewconfig.py` (View configuration and cubic-view helpers)
- `ViewConfig` encapsulates many display toggles and cubic-view tuning parameters.
- Methods include `_setup_axis_mapping()`, `_setup_cubic_view()`, `update(**kwargs)`, `axis_vectors()`, `axis_label_strings()`, `get_grid_planes()`, `get_cube_corners()`, `get_cube_face_centers()`, `get_cubic_grid_settings()`, `grid_axes()`, `get_grid_normal()`, `get_grid_basis()`, `get_display_settings()`, `clone()`, `__str__()` — used by `Renderer`, `Labels`, and UI components to read or tweak rendering settings.

`app/app.py` (Application lifecycle and GLFW/ImGui glue)
- `dlog(msg)`: Debug logging helper.
- `App.__init__()`: Create GLFW window, setup ModernGL context, ImGui integration, camera, renderer, label renderer, workspace layout, Redux store, and input callbacks.
- Input handlers: `on_key(win, key, scancode, action, mods)`, `on_resize(win, width, height)`, `on_mouse_button(win, btn, action, mods)`, `on_mouse_move(win, x, y)`, `on_scroll(win, xoff, yoff)` — process keyboard and mouse events and drive camera and picking logic.
- `run()`: Main loop: event polling, UI frame, call `workspace.render(...)`, rendering via `SceneAdapter`, label overlays, and presentation.

`app/app_state_bridge.py` (State-to-runtime adapter)
- `build_scene_adapter(state)`: Create a read-only adapter for rendering.

`ui/panels/sidebar/sidebar.py` (Primary operations panel with vector/matrix/system controls)
- `Sidebar` manages input widgets and state used by the UI.
- Helpers and UI widgets: `_get_next_color()`, `_styled_button()`, `_section()`, `_end_section()`, `_input_float3(...)`, `_matrix_input_widget(...)` — small UI building blocks to keep consistent styling and correct per-cell IDs.
- Vector UI: `_render_vector_creation()`, `_render_vector_operations()`, `_render_vector_list()` — read from AppState and dispatch actions.
- Matrix UI and solvers: `_render_matrix_operations()`, `_render_linear_systems()`, `_resize_equations()`, `_solve_linear_system()`, `_add_solution_vectors(solution)` — driven by AppState inputs and actions.
- Export helpers: `_render_export_dialog()`, `_export_json()`, `_export_csv()`, `_export_python()`.

`ui/inspectors/inspector.py` (Right-hand inspector for selected object)
- `Inspector.render(state, dispatch, rect)` orchestrates the inspector window.
- Private methods: `_render_header(vector, dispatch)`, `_render_coordinate_editor(vector, dispatch)`, `_render_properties(vector, dispatch)`, `_render_transform_history(vector, dispatch)`, `_render_computed_properties(vector, state, dispatch)` — used to inspect and edit coordinates, color, label, and computed relations with other vectors.

`ui/toolbars/toolbar.py` (Top toolbar)
- `Toolbar.render(state, dispatch, camera, view_config, app)` draws a Photoshop-style ribbon and theme controls.

`ui/layout/workspace.py` (UI layout orchestration)
- `WorkspaceLayout.render(state, dispatch, camera, view_config, app)` lays out toolbar, tool palette, operations panel, inspector, and timeline.

`ui/panels/tool_palette/tool_palette.py` (Left tool palette)
- `ToolPalette.render(rect, state, dispatch)` draws tool buttons and sets `active_tool`.

`ui/panels/timeline/timeline_panel.py` (Bottom timeline)
- `TimelinePanel.render(rect, state, dispatch)` shows educational steps and navigation controls.

Usage notes
- Most UI functions are callbacks invoked during the ImGui frame (via `App.run()` -> `WorkspaceLayout.render()`).
- Rendering helpers in `render/gizmos/gizmos.py` are called by `Renderer` to produce GPU draws; they accept plain Python lists/ndarrays and a view-projection matrix `vp`.
