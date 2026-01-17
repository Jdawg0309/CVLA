"""
Enhanced Sidebar with modern UI for linear algebra operations.
"""

from ui.panels.sidebar.sidebar_export import (
    _render_export_dialog,
    _export_json,
    _export_csv,
    _export_python,
)
from ui.panels.sidebar.sidebar_linear_system_ops import (
    _resize_equations,
    _solve_linear_system,
    _add_solution_vectors,
)
from ui.panels.sidebar.sidebar_linear_systems_section import _render_linear_systems
from ui.panels.sidebar.sidebar_matrix_ops import (
    _compute_null_space,
    _compute_column_space,
)
from ui.panels.sidebar.sidebar_matrix_section import _render_matrix_operations
from ui.panels.sidebar.sidebar_render import render
from ui.panels.sidebar.sidebar_state import sidebar_init
from ui.panels.sidebar.sidebar_utils import (
    _get_next_color,
    _styled_button,
    _section,
    _end_section,
    _input_float3,
    _input_float_list,
    _coerce_float,
    _input_number_cell,
    _matrix_input_widget,
)
from ui.panels.sidebar.sidebar_vector_creation import _render_vector_creation
from ui.panels.sidebar.sidebar_vector_list import _render_vector_list
from ui.panels.sidebar.sidebar_vector_operations import _render_vector_operations, _do_vector_algebra
from ui.panels.sidebar.sidebar_visualization_section import _render_visualization_options
from ui.panels.sidebar.sidebar_input_section import _render_input_section


class Sidebar:
    __init__ = sidebar_init

    _get_next_color = _get_next_color
    _styled_button = _styled_button
    _section = _section
    _end_section = _end_section
    _input_float3 = _input_float3
    _input_float_list = _input_float_list
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
    _render_input_section = _render_input_section

    _render_export_dialog = _render_export_dialog
    _export_json = _export_json
    _export_csv = _export_csv
    _export_python = _export_python

    render = render
