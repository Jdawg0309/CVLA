"""
Enhanced Sidebar with modern UI for linear algebra operations.
"""

from ui.sidebar_export import (
    _render_export_dialog,
    _export_json,
    _export_csv,
    _export_python,
)
from ui.sidebar_images_convolution import _render_image_convolution_section
from ui.sidebar_images_education import _render_image_education_section
from ui.sidebar_images_info import _render_image_info_section
from ui.sidebar_images_ops import _add_image_as_vectors
from ui.sidebar_images_result import _render_image_result_section
from ui.sidebar_images_section import _render_image_operations
from ui.sidebar_images_source import _render_image_source_section
from ui.sidebar_images_transform import _render_image_transform_section
from ui.sidebar_linear_system_ops import (
    _resize_equations,
    _solve_linear_system,
    _add_solution_vectors,
)
from ui.sidebar_linear_systems_section import _render_linear_systems
from ui.sidebar_matrix_ops import (
    _compute_null_space,
    _compute_column_space,
)
from ui.sidebar_matrix_section import _render_matrix_operations
from ui.sidebar_render import render
from ui.sidebar_state import sidebar_init
from ui.sidebar_utils import (
    _get_next_color,
    _styled_button,
    _section,
    _end_section,
    _input_float3,
    _coerce_float,
    _input_number_cell,
    _matrix_input_widget,
)
from ui.sidebar_vector_creation import _render_vector_creation
from ui.sidebar_vector_list import _render_vector_list
from ui.sidebar_vector_operations import _render_vector_operations, _do_vector_algebra
from ui.sidebar_visualization_section import _render_visualization_options


class Sidebar:
    __init__ = sidebar_init

    _get_next_color = _get_next_color
    _styled_button = _styled_button
    _section = _section
    _end_section = _end_section
    _input_float3 = _input_float3
    _coerce_float = _coerce_float
    _input_number_cell = _input_number_cell
    _matrix_input_widget = _matrix_input_widget

    _render_vector_creation = _render_vector_creation
    _render_vector_operations = _render_vector_operations
    _render_vector_list = _render_vector_list

    _do_vector_algebra = _do_vector_algebra

    _render_matrix_operations = _render_matrix_operations
    _compute_null_space = _compute_null_space
    _compute_column_space = _compute_column_space

    _render_linear_systems = _render_linear_systems
    _resize_equations = _resize_equations
    _solve_linear_system = _solve_linear_system
    _add_solution_vectors = _add_solution_vectors

    _render_visualization_options = _render_visualization_options

    _render_export_dialog = _render_export_dialog
    _export_json = _export_json
    _export_csv = _export_csv
    _export_python = _export_python

    _render_image_operations = _render_image_operations
    _render_image_source_section = _render_image_source_section
    _render_image_info_section = _render_image_info_section
    _render_image_convolution_section = _render_image_convolution_section
    _render_image_transform_section = _render_image_transform_section
    _render_image_result_section = _render_image_result_section
    _render_image_education_section = _render_image_education_section
    _add_image_as_vectors = _add_image_as_vectors

    render = render
