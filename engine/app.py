"""
CVLA MVP - Application Loop with enhanced cubic view
"""

import os
import time
import glfw
import moderngl
import imgui
from imgui.integrations.glfw import GlfwRenderer

from engine.renderer import Renderer
from engine.camera import Camera
from engine.labels import LabelRenderer
from engine.viewconfig import ViewConfig
from engine.picking import pick_vector

from core.scene import Scene
from core.vector import Vector3D
from ui.sidebar import Sidebar


DEBUG = os.getenv("CVLA_DEBUG") == "1"


def dlog(msg: str):
    if DEBUG:
        print(msg)


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
        # Enhanced basis vectors for cubic view
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
        self.auto_rotate = False
        self.rotation_speed = 0.5
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

    def _setup_imgui_style(self):
        """Setup ImGui style for better visual appearance."""
        style = imgui.get_style()
        # Improve readability: slightly larger global font and more generous spacing
        try:
            # Global font scale increases all ImGui text sizes
            style.font_global_scale = 1.18
        except Exception:
            pass

        style.window_rounding = 6.0
        style.frame_rounding = 4.0
        style.grab_rounding = 4.0
        style.frame_padding = (8, 6)
        style.window_padding = (12, 12)
        style.item_spacing = (10, 8)
        
        colors = style.colors
        # Dark but higher-contrast background colors for readability
        colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.06, 0.06, 0.07, 0.98)
        colors[imgui.COLOR_FRAME_BACKGROUND] = (0.18, 0.18, 0.22, 1.00)
        colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.24, 0.24, 0.28, 1.00)
        colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = (0.28, 0.28, 0.32, 1.00)
        colors[imgui.COLOR_BUTTON] = (0.22, 0.45, 0.82, 0.9)
        colors[imgui.COLOR_BUTTON_HOVERED] = (0.26, 0.55, 0.95, 1.0)
        colors[imgui.COLOR_BUTTON_ACTIVE] = (0.18, 0.38, 0.72, 1.0)
        colors[imgui.COLOR_HEADER] = (0.20, 0.20, 0.24, 0.95)
        colors[imgui.COLOR_HEADER_HOVERED] = (0.26, 0.26, 0.30, 1.0)
        colors[imgui.COLOR_HEADER_ACTIVE] = (0.30, 0.30, 0.34, 1.0)

    def on_key(self, win, key, scancode, action, mods):
        """Handle keyboard input."""
        if action == glfw.PRESS or action == glfw.REPEAT:
            # Toggle auto-rotation with R key
            if key == glfw.KEY_R:
                self.auto_rotate = not self.auto_rotate
                dlog(f"[App] Auto-rotate: {self.auto_rotate}")
            
            # Reset view with space bar
            elif key == glfw.KEY_SPACE:
                self.camera.reset()
                dlog("[App] Camera reset")
            
            # Toggle cubic view with C key
            elif key == glfw.KEY_C:
                if self.view_config.grid_mode == "cube":
                    self.view_config.grid_mode = "plane"
                    self.view_config.grid_plane = "xy"
                else:
                    self.view_config.grid_mode = "cube"
                dlog(f"[App] Grid mode: {self.view_config.grid_mode}")
            
            # Toggle cube faces with F key
            elif key == glfw.KEY_F:
                self.view_config.show_cube_faces = not self.view_config.show_cube_faces
                dlog(f"[App] Cube faces: {self.view_config.show_cube_faces}")
            
            # Toggle vector components with V key
            elif key == glfw.KEY_V:
                self.renderer.show_vector_components = not self.renderer.show_vector_components
                dlog(f"[App] Vector components: {self.renderer.show_vector_components}")

    def on_resize(self, win, width, height):
        """Handle window resize."""
        self.camera.set_viewport(width, height)

    # -------------------------------------------------
    # Input handling (unchanged from original)
    # -------------------------------------------------
    def on_mouse_button(self, win, btn, action, mods):
        x, y = glfw.get_cursor_pos(win)
        io = imgui.get_io()

        if btn == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            if io.want_capture_mouse:
                return

            fb_w, fb_h = glfw.get_framebuffer_size(self.window)
            self.selected = pick_vector(
                screen_x=x,
                screen_y=y,
                width=fb_w,
                height=fb_h,
                camera=self.camera,
                vectors=self.scene.vectors,
                radius_px=20,
            )
            
            if self.selected:
                dlog(f"[App] Selected vector: {self.selected.label}")
                self.camera.focus_on_vector(self.selected.coords)

        if btn == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                if io.want_capture_mouse:
                    return
                self.rotating = True
                self.last_mouse = (x, y)
            elif action == glfw.RELEASE:
                self.rotating = False

    def on_mouse_move(self, win, x, y):
        if not self.rotating:
            return

        lx, ly = self.last_mouse
        dx, dy = x - lx, y - ly
        self.last_mouse = (x, y)

        self.camera.orbit(dx, dy)

        if DEBUG:
            now = time.time()
            if now - self._last_debug_time > 0.15:
                dlog(f"[Camera] orbit θ={self.camera.theta:.2f}, φ={self.camera.phi:.2f}")
                self._last_debug_time = now

    def on_scroll(self, win, xoff, yoff):
        if imgui.get_io().want_capture_mouse:
            return
        self.camera.zoom(yoff)

    # -------------------------------------------------
    # Main loop with cubic view enhancements
    # -------------------------------------------------
    def run(self):
        last_time = time.time()
        
        while not glfw.window_should_close(self.window):
            # Calculate delta time
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            
            glfw.poll_events()
            self.imgui.process_inputs()

            fb_w, fb_h = glfw.get_framebuffer_size(self.window)
            if fb_w <= 0 or fb_h <= 0:
                continue

            self.ctx.viewport = (0, 0, fb_w, fb_h)
            self.camera.set_viewport(fb_w, fb_h)
            
            # Auto-rotation for cubic view
            if self.auto_rotate:
                self.camera.cubic_view_rotation(auto_rotate=True, speed=self.rotation_speed)

            # Update systems with new view config
            self.renderer.update_view(self.view_config)
            self.labels.update_view(self.view_config)

            # UI
            imgui.new_frame()
            self.selected = self.sidebar.render(
                fb_h, self.scene, self.selected, self.camera, self.view_config
            )
            
            # Add cubic view controls to UI
            self._render_cubic_view_controls()

            # 3D Rendering with enhanced cubic view
            self.renderer.render(self.scene)

            # Update FPS (simple instantaneous measure)
            try:
                self.fps = 1.0 / max(delta_time, 1e-6)
            except Exception:
                self.fps = 0.0

            # 2D Overlay (Labels)
            if self.view_config.show_labels:
                self.labels.draw_axes(self.camera, fb_w, fb_h)
                self.labels.draw_grid_numbers(
                    camera=self.camera,
                    width=fb_w,
                    height=fb_h,
                    viewconfig=self.view_config,
                    grid_size=self.view_config.grid_size,
                    major=self.view_config.major_tick
                )
                self.labels.draw_vector_labels(self.camera, self.scene.vectors, fb_w, fb_h, 
                                             selected_vector=self.selected)

            # Present
            imgui.render()
            self.imgui.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)

        # Cleanup
        self.imgui.shutdown()
        glfw.terminate()
        dlog("[App] shutdown")

    def _render_cubic_view_controls(self):
        """Render additional controls for cubic view."""
        imgui.begin("Cubic View Controls", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        
        # Auto-rotation control
        _, self.auto_rotate = imgui.checkbox("Auto-rotate", self.auto_rotate)
        if self.auto_rotate:
            imgui.same_line()
            imgui.push_item_width(100)
            _, self.rotation_speed = imgui.slider_float("Speed", self.rotation_speed, 0.1, 2.0)
            imgui.pop_item_width()
        
        # Cube face toggle
        _, self.view_config.show_cube_faces = imgui.checkbox(
            "Show Cube Faces", self.view_config.show_cube_faces
        )
        
        # Cube corner indicators
        _, self.view_config.show_cube_corners = imgui.checkbox(
            "Show Corner Indicators", self.view_config.show_cube_corners
        )
        
        # Grid density
        imgui.push_item_width(150)
        changed, self.view_config.cubic_grid_density = imgui.slider_float(
            "Grid Density", self.view_config.cubic_grid_density, 0.5, 3.0
        )
        if changed:
            self.view_config._setup_cubic_view()
        imgui.pop_item_width()
        
        # Cube face opacity
        changed, self.view_config.cube_face_opacity = imgui.slider_float(
            "Face Opacity", self.view_config.cube_face_opacity, 0.01, 0.3
        )
        if changed:
            for i in range(len(self.view_config.cube_face_colors)):
                color = list(self.view_config.cube_face_colors[i])
                color[3] = self.view_config.cube_face_opacity
                self.view_config.cube_face_colors[i] = tuple(color)
        
        imgui.separator()
        
        # Quick view presets
        if imgui.button("XY View"):
            self.view_config.grid_mode = "plane"
            self.view_config.grid_plane = "xy"
            self.camera.set_view_preset("xy")
        
        imgui.same_line()
        if imgui.button("XZ View"):
            self.view_config.grid_mode = "plane"
            self.view_config.grid_plane = "xz"
            self.camera.set_view_preset("xz")
        
        imgui.same_line()
        if imgui.button("YZ View"):
            self.view_config.grid_mode = "plane"
            self.view_config.grid_plane = "yz"
            self.camera.set_view_preset("yz")
        
        imgui.same_line()
        if imgui.button("3D Cube"):
            self.view_config.grid_mode = "cube"
            self.camera.set_view_preset("cube")
        
        imgui.end()