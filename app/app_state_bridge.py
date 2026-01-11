"""Bridge helpers between AppState and runtime systems."""

from engine.scene_adapter import SceneAdapter


def build_scene_adapter(state):
    """Create a read-only SceneAdapter for rendering."""
    return SceneAdapter(state)
