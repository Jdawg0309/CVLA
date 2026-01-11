"""
Inspector transform history section.
"""

import imgui


def _render_transform_history(self, vector):
    """Render transformation history."""
    if imgui.collapsing_header("Transform History",
                              flags=imgui.TREE_NODE_DEFAULT_OPEN):
        if not vector.history:
            imgui.text_disabled("No transformations applied")
        else:
            for i, (op, param) in enumerate(vector.history):
                if op == 'scale':
                    imgui.text(f"{i+1}. Scaled by {param:.4f}")
                elif op == 'normalize':
                    imgui.text(f"{i+1}. Normalized")
                elif op == 'transform':
                    imgui.text(f"{i+1}. Matrix transform")

            imgui.spacing()
            if imgui.button("Clear History", width=-1):
                vector.history.clear()
