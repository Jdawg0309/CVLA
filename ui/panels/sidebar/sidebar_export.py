"""
Sidebar export helpers.
"""

import imgui
import json
from state.selectors import get_vectors


def _render_export_dialog(self):
    """Render export dialog."""
    if self.show_export_dialog:
        imgui.open_popup("Export Vectors")

    if imgui.begin_popup_modal("Export Vectors")[0]:
        imgui.text("Export format:")
        imgui.spacing()

        if imgui.button("JSON", width=100):
            self._export_json()
            self.show_export_dialog = False
            imgui.close_current_popup()

        imgui.same_line()

        if imgui.button("CSV", width=100):
            self._export_csv()
            self.show_export_dialog = False
            imgui.close_current_popup()

        imgui.same_line()

        if imgui.button("Python", width=100):
            self._export_python()
            self.show_export_dialog = False
            imgui.close_current_popup()

        imgui.end_popup()


def _export_json(self):
    """Export vectors to JSON."""
    if self._state is None:
        print("JSON Export unavailable (no state).")
        return

    vectors = list(get_vectors(self._state))
    data = {
        'vectors': [
            {
                'label': v.label,
                'coords': list(v.coords),
                'color': v.color,
                'visible': v.visible
            }
            for v in vectors
        ]
    }

    if data['vectors']:
        print("JSON Export (first vector):", json.dumps(data['vectors'][0], indent=2))
    else:
        print("JSON Export: No vectors to export")


def _export_csv(self):
    """Export vectors to CSV."""
    if self._state is None:
        print("CSV Export unavailable (no state).")
        return

    vectors = list(get_vectors(self._state))
    csv_lines = ["Label,X,Y,Z,R,G,B"]
    for v in vectors:
        coords = list(v.coords)
        while len(coords) < 3:
            coords.append(0.0)
        csv_lines.append(
            f'{v.label},{coords[0]},{coords[1]},{coords[2]},'
            f'{v.color[0]},{v.color[1]},{v.color[2]}'
        )

    print("CSV Export (first few lines):")
    for line in csv_lines[:3]:
        print(line)


def _export_python(self):
    """Export vectors as Python code."""
    if self._state is None:
        print("Python Export unavailable (no state).")
        return

    vectors = list(get_vectors(self._state))
    python_code = "# CVLA Vector Export\n"
    python_code += "import numpy as np\n\n"
    python_code += "vectors = [\n"

    for v in vectors:
        python_code += f"    # {v.label}\n"
        python_code += f"    np.array({list(v.coords)}, dtype=np.float32),\n"

    python_code += "]\n"

    print("Python Export ready")
