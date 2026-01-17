"""
App input handlers.

This module handles keyboard, mouse, and window events.
Uses SceneAdapter for picking and dispatches SelectVector actions.
"""

import time
import glfw
import imgui
from engine.picking_system import pick_vector
from app.app_logging import dlog, DEBUG
from app.app_state_bridge import build_scene_adapter
from state.actions import SelectVector, AddVector, AddMatrix, SetSelectedPixel


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
    if hasattr(self, "imgui") and self.imgui is not None:
        self.imgui.keyboard_callback(win, key, scancode, action, mods)


def on_resize(self, win, width, height):
    """Handle window resize."""
    self.camera.set_viewport(width, height)


def on_mouse_button(self, win, btn, action, mods):
    x, y = glfw.get_cursor_pos(win)
    io = imgui.get_io()

    state = self.store.get_state()
    active_tool = getattr(state, "active_tool", "select")

    if btn == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        if io.want_capture_mouse:
            return

        fb_w, fb_h = glfw.get_framebuffer_size(self.window)

        if active_tool == "select":
            # Get vectors from state via adapter
            scene_adapter = build_scene_adapter(state)

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
            return

        if active_tool == "move":
            self.panning = True
            self.last_mouse = (x, y)
            return

        if active_tool == "rotate":
            self.rotating = True
            self.last_mouse = (x, y)
            return

        if active_tool == "add_vector":
            origin, direction = self.camera.screen_to_ray(x, y, fb_w, fb_h)
            plane = self.view_config.grid_plane if self.view_config.grid_mode == "plane" else "xy"
            axis_index = {"xy": 2, "xz": 1, "yz": 0}.get(plane, 2)
            denom = direction[axis_index]
            if abs(denom) < 1e-6:
                return
            t = (0.0 - origin[axis_index]) / denom
            if t < 0:
                return
            hit = origin + direction * t
            coords = (float(hit[0]), float(hit[1]), float(hit[2]))
            self.store.dispatch(AddVector(
                coords=coords,
                color=(0.8, 0.2, 0.2),
                label="",
            ))
            return

        if active_tool == "image":
            if not state.show_image_on_grid:
                return
            image = state.current_image
            if state.active_image_tab != "raw" and state.processed_image is not None:
                image = state.processed_image
            if image is None:
                return
            origin, direction = self.camera.screen_to_ray(x, y, fb_w, fb_h)
            denom = direction[2]
            if abs(denom) < 1e-6:
                return
            t = (0.0 - origin[2]) / denom
            if t < 0:
                return
            hit = origin + direction * t
            scale = max(0.001, float(state.image_render_scale))
            col = int((hit[0] / scale) + (image.width / 2.0))
            row = int((image.height / 2.0) - (hit[1] / scale))
            if 0 <= row < image.height and 0 <= col < image.width:
                self.store.dispatch(SetSelectedPixel(row=row, col=col))
            return

        if active_tool == "add_matrix":
            matrix = tuple(tuple(row) for row in state.input_matrix)
            self.store.dispatch(AddMatrix(values=matrix, label=state.input_matrix_label))
            return

    if btn == glfw.MOUSE_BUTTON_RIGHT:
        if action == glfw.PRESS:
            if io.want_capture_mouse:
                return
            self.rotating = True
            self.last_mouse = (x, y)
        elif action == glfw.RELEASE:
            self.rotating = False

    if btn == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
        self.rotating = False
        self.panning = False


def on_mouse_move(self, win, x, y):
    if not self.rotating and not self.panning:
        return

    lx, ly = self.last_mouse
    dx, dy = x - lx, y - ly
    self.last_mouse = (x, y)

    if self.panning:
        self.camera.pan(dx, dy)
    else:
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
