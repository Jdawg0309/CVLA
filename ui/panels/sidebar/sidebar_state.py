"""
Sidebar state initialization.

NOTE: Most state has been migrated to AppState. This file only contains
UI-specific state that doesn't need to be in the global store:
- ImGui widget buffers
- Transient UI state
- Window layout

Vector/Matrix/Image data is now in AppState via the Redux store.
"""


def sidebar_init(self):
    # =========================================================================
    # UI LAYOUT STATE (local to sidebar)
    # =========================================================================
    self.active_tab = "vectors"  # Fallback when state is unavailable

    # =========================================================================
    # TRANSIENT INPUT BUFFERS (for ImGui text inputs)
    # =========================================================================

    # =========================================================================
    # UI WIDGET STATE (purely local)
    # =========================================================================
    self.show_equation_editor = False
    self.show_export_dialog = False
    self.vector_list_filter = ""
    self._cell_buffers = {}  # For matrix cell editing
    self._cell_active = set()

    # Matrix selection index (local UI state)
    self.selected_matrix_idx = None

    # Runtime references
    self._state = None  # Current AppState (set each frame)
    self._dispatch = None  # Dispatch function (set each frame)
    self.scale_factor = 1.0

    # =========================================================================
    # OPERATION RESULTS (transient display state)
    # =========================================================================
    self.current_operation = None
    self.operation_result = None
    self.convolution_position = (0, 0)
