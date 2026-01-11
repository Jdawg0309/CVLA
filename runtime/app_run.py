"""
App main loop.
"""

import time
import glfw
import imgui

from runtime.app_logging import dlog


def run(self):
    last_time = time.time()

    while not glfw.window_should_close(self.window):
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

        if getattr(self.view_config, 'auto_rotate', False):
            speed = getattr(self.view_config, 'rotation_speed', 0.5)
            self.camera.cubic_view_rotation(auto_rotate=True, speed=speed)

        self.renderer.update_view(self.view_config)
        self.labels.update_view(self.view_config)

        state = self.store.get_state()
        imgui.new_frame()
        self.selected = self.sidebar.render(
            fb_h, self.scene, self.selected, self.camera, self.view_config,
            state=state,
            dispatch=self.store.dispatch
        )

        render_image = state.processed_image or state.current_image
        self.renderer.render(
            self.scene,
            image_data=render_image,
            show_image_on_grid=state.show_image_on_grid,
            image_render_scale=state.image_render_scale,
            image_color_mode=state.image_color_mode,
        )

        try:
            self.fps = 1.0 / max(delta_time, 1e-6)
        except Exception:
            self.fps = 0.0

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

        imgui.render()
        self.imgui.render(imgui.get_draw_data())
        glfw.swap_buffers(self.window)

    self.imgui.shutdown()
    glfw.terminate()
    dlog("[App] shutdown")
