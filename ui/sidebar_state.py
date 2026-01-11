"""
Sidebar state initialization.
"""


def sidebar_init(self):
    # Vector creation
    self.vec_input = [1.0, 0.0, 0.0]
    self.vec_name = ""
    self.vec_color = (0.8, 0.2, 0.2)
    self.next_vector_id = 4

    # Matrix input
    self.matrix_input = [[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]]
    self.matrix_name = "A"
    self.matrix_size = 3

    # System of equations
    self.equation_count = 3
    self.equation_input = [
        [1.0, 1.0, 1.0, 0.0],  # x + y + z = 0
        [2.0, -1.0, 0.0, 0.0], # 2x - y = 0
        [0.0, 1.0, -1.0, 0.0]  # y - z = 0
    ]

    # Operation state
    self.current_operation = None
    self.operation_result = None
    self.show_matrix_editor = False
    self.show_equation_editor = False
    self.show_export_dialog = False

    # UI state
    self.window_width = 420
    self.active_tab = "vectors"
    self.vector_list_filter = ""
    self._cell_buffers = {}
    self._cell_active = set()

    # Color palette for auto-colors
    self.color_palette = [
        (0.8, 0.2, 0.2),  # Red
        (0.2, 0.8, 0.2),  # Green
        (0.2, 0.2, 0.8),  # Blue
        (0.8, 0.8, 0.2),  # Yellow
        (0.8, 0.2, 0.8),  # Magenta
        (0.2, 0.8, 0.8),  # Cyan
        (0.8, 0.5, 0.2),  # Orange
        (0.5, 0.2, 0.8),  # Purple
    ]
    self.next_color_idx = 0
    # Preview toggle for matrix editor
    self.preview_matrix_enabled = False
    # Selected matrix index in the scene (for list operations)
    self.selected_matrix_idx = None
    # Runtime references and defaults
    self.scene = None
    self.scale_factor = 1.0

    # Image/Vision state (for ML/CV visualization)
    self.current_image = None  # Current ImageMatrix
    self.processed_image = None  # Result of last operation
    self.selected_kernel = 'sobel_x'  # Default kernel
    self.show_image_matrix = False  # Show matrix values
    self.image_path = ""  # Path input for loading
    self.sample_pattern = 'checkerboard'  # For sample images
    self.sample_size = 32  # Sample image size
    self.transform_rotation = 0.0  # Rotation angle
    self.transform_scale = 1.0  # Scale factor
    self.convolution_position = (0, 0)  # For visualization
