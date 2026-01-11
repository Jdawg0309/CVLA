"""
Scene history and undo/redo helpers.
"""

import copy
import numpy as np

from core.vector import Vector3D


class SceneHistoryMixin:
    def snapshot(self) -> dict:
        """Return a deep copy snapshot of the scene state relevant for undo."""
        return {
            'vectors': [
                {
                    'coords': v.coords.copy(),
                    'color': v.color,
                    'label': v.label,
                    'visible': v.visible,
                    'metadata': copy.deepcopy(v.metadata),
                }
                for v in self.vectors
            ],
            'matrices': [
                {
                    'matrix': m['matrix'].copy(),
                    'label': m.get('label', ''),
                    'visible': m.get('visible', True),
                }
                for m in self.matrices
            ],
            'selected_object': None,
            'selection_type': self.selection_type
        }

    def apply_snapshot(self, snap: dict):
        """Restore a snapshot into the scene (non-destructive replacement)."""
        self.vectors.clear()
        for v in snap.get('vectors', []):
            vec = Vector3D(np.array(v['coords'], dtype=np.float32), color=v.get('color', (0.8, 0.2, 0.2)), label=v.get('label', ''))
            vec.visible = v.get('visible', True)
            vec.metadata = copy.deepcopy(v.get('metadata', {}))
            self.vectors.append(vec)

        self.matrices.clear()
        for m in snap.get('matrices', []):
            md = {
                'matrix': np.array(m['matrix'], dtype=np.float32),
                'label': m.get('label', ''),
                'visible': m.get('visible', True),
                'color': (0.8, 0.5, 0.2, 0.6),
                'transformations': []
            }
            self.matrices.append(md)

        self.selected_object = None
        self.selection_type = None

    def push_undo(self):
        try:
            snap = self.snapshot()
            self._undo_stack.append(snap)
            self._redo_stack.clear()
        except Exception:
            pass

    def undo(self):
        if not self._undo_stack:
            return
        try:
            current = self.snapshot()
            last = self._undo_stack.pop()
            self._redo_stack.append(current)
            self.apply_snapshot(last)
        except Exception:
            pass

    def redo(self):
        if not self._redo_stack:
            return
        try:
            current = self.snapshot()
            nxt = self._redo_stack.pop()
            self._undo_stack.append(current)
            self.apply_snapshot(nxt)
        except Exception:
            pass
