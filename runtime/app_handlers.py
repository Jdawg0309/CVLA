"""
App input handlers.

This module handles keyboard, mouse, and window events.
Uses SceneAdapter for picking and dispatches SelectVector actions.
"""

import time
import glfw
import imgui

from engine.picking import pick_vector
from runtime.app_logging import dlog, DEBUG
from graph.scene_adapter import SceneAdapter
from state.actions import SelectVector


def on_key(self, win, key, scancode, action, mods):
    """Handle keyboard input."""
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_R:
            self.view_config.auto_rotate = not getattr(self.view_config, 'auto_rotate', False)
            dlog(f"[App] Auto-rotate: {self.view_config.auto_rotate}")

        elif key == glfw.KEY_SPACE:
            self.camera.reset()
            dlog("[App] Camera reset")

        elif key == glfw.KEY_C:
            if self.view_config.grid_mode == "cube":
                self.view_config.grid_mode = "plane"
                self.view_config.grid_plane = "xy"
            else:
                self.view_config.grid_mode = "cube"
            dlog(f"[App] Grid mode: {self.view_config.grid_mode}")

        elif key == glfw.KEY_F:
            self.view_config.show_cube_faces = not self.view_config.show_cube_faces
            dlog(f"[App] Cube faces: {self.view_config.show_cube_faces}")

        elif key == glfw.KEY_V:
            self.renderer.show_vector_components = not self.renderer.show_vector_components
            dlog(f"[App] Vector components: {self.renderer.show_vector_components}")


def on_resize(self, win, width, height):
    """Handle window resize."""
    self.camera.set_viewport(width, height)


def on_mouse_button(self, win, btn, action, mods):
    x, y = glfw.get_cursor_pos(win)
    io = imgui.get_io()

    if btn == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        if io.want_capture_mouse:
            return

        fb_w, fb_h = glfw.get_framebuffer_size(self.window)

        # Get vectors from state via adapter
        state = self.store.get_state()
        scene_adapter = SceneAdapter(state)

        picked = pick_vector(
            screen_x=x,
            screen_y=y,
            width=fb_w,
            height=fb_h,
            camera=self.camera,
            vectors=scene_adapter.vectors,
            radius_px=20,
        )

        if picked:
            dlog(f"[App] Selected vector: {picked.label}")
            self.camera.focus_on_vector(picked.coords)

            # Find vector ID by label and dispatch selection
            for v in state.vectors:
                if v.label == picked.label:
                    self.store.dispatch(SelectVector(id=v.id))
                    break

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
