"""
CVLA MVP - Application Loop with enhanced cubic view
"""

import glfw
import moderngl
import imgui
from imgui.integrations.glfw import GlfwRenderer

from render.renderer import Renderer
from render.camera import Camera
from engine.labels import LabelRenderer
from render.viewconfig import ViewConfig
from runtime.app_handlers import on_key, on_resize, on_mouse_button, on_mouse_move, on_scroll
from runtime.app_logging import dlog, DEBUG
from runtime.app_run import run
from runtime.app_style import setup_imgui_style

from core.scene import Scene
from core.vector import Vector3D
from ui.sidebar import Sidebar

# New state management
from state import Store, create_initial_state, create_scene_from_state


class App:
    def __init__(self):
        # -----------------------------
        # Window / GL context
        # -----------------------------
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.SAMPLES, 8)  # Increased anti-aliasing
        glfw.window_hint(glfw.RESIZABLE, True)
        glfw.window_hint(glfw.DEPTH_BITS, 24)
        
        self.window = glfw.create_window(1440, 900, "CVLA - 3D Vector Visualizer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # Enable vsync

        # -----------------------------
        # ModernGL
        # -----------------------------
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)
        # Some ModernGL builds / versions don't expose MULTISAMPLE constant.
        # Respect the GLFW window hint for samples and only enable if available.
        if hasattr(moderngl, "MULTISAMPLE"):
            self.ctx.enable(moderngl.MULTISAMPLE)
        else:
            dlog("[App] moderngl.MULTISAMPLE not found; skipping explicit enable")
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # -----------------------------
        # ImGui
        # -----------------------------
        imgui.create_context()
        self.imgui = GlfwRenderer(self.window)
        
        # Configure ImGui style for better visuals
        self._setup_imgui_style()

        # -----------------------------
        # Core systems
        # -----------------------------
        self.camera = Camera()
        self.scene = Scene()
        
        # Initialize view configuration with cubic view
        self.view_config = ViewConfig(
            up_axis="z",
            grid_mode="cube",  # Enable cubic view
            grid_plane="xy",
            grid_size=15,  # Adjusted for cubic view
            major_tick=5,
            minor_tick=1,
            show_grid=True,
            show_axes=True,
            show_labels=True,
            show_cube_faces=True,  # Enable cube faces
            show_cube_corners=True,
            cubic_grid_density=1.0,
            vector_scale=3.0
        )
        
        # Initialize systems with view config
        self.renderer = Renderer(self.ctx, self.camera, view=self.view_config)
        self.labels = LabelRenderer()
        self.labels.update_view(self.view_config)
        self.sidebar = Sidebar()

        # -----------------------------
        # State Management (Redux-style)
        # Single source of truth for Images tab
        # -----------------------------
        self.store = Store(create_initial_state())

        # -----------------------------
        # Enhanced basis vectors for cubic view
        # (Using old Scene for vectors/matrices tabs for now)
        # -----------------------------
        self.scene.add_vector(Vector3D([2, 0, 0], color=(1.0, 0.25, 0.25), label="i"))
        self.scene.add_vector(Vector3D([0, 2, 0], color=(0.25, 1.0, 0.25), label="j"))
        self.scene.add_vector(Vector3D([0, 0, 2], color=(0.35, 0.55, 1.0), label="k"))
        
        # Add some additional vectors for better visualization
        self.scene.add_vector(Vector3D([1, 1, 0], color=(0.8, 0.6, 0.2), label="v₁"))
        self.scene.add_vector(Vector3D([0.5, 1, 1], color=(0.6, 0.2, 0.8), label="v₂"))

        # State
        self.selected = None
        self.rotating = False
        self.last_mouse = None
        self._last_debug_time = 0.0
        # Runtime status
        self.fps = 0.0
        self.show_help = False

        # -----------------------------
        # GLFW callbacks
        # -----------------------------
        glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self.on_mouse_move)
        glfw.set_scroll_callback(self.window, self.on_scroll)
        glfw.set_key_callback(self.window, self.on_key)
        glfw.set_window_size_callback(self.window, self.on_resize)

        dlog("[App] initialized with cubic view")

    _setup_imgui_style = setup_imgui_style
    on_key = on_key
    on_resize = on_resize
    on_mouse_button = on_mouse_button
    on_mouse_move = on_mouse_move
    on_scroll = on_scroll
    run = run
