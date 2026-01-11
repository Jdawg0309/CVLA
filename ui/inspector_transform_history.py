"""
Inspector transform history section.
"""

import imgui


def _render_transform_history(self, vector, dispatch):
    """Render transformation history."""
    if imgui.collapsing_header("Transform History",
                              flags=imgui.TREE_NODE_DEFAULT_OPEN):
        imgui.text_disabled("Transform history is not tracked in AppState.")
