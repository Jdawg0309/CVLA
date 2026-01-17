"""
App main loop.

This module contains the main render loop. It uses SceneAdapter to convert
the immutable AppState into a format the renderer understands.
"""

import glfw
import imgui

from app.app_logging import dlog
from app.app_state_bridge import build_scene_adapter
from engine.execution_loop import FrameTimer
from state import get_current_step


def run(self):
    timer = FrameTimer()

    while not glfw.window_should_close(self.window):
        delta_time = timer.tick()

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

        # Get current state and create scene adapter for rendering
        state = self.store.get_state()
        scene_adapter = build_scene_adapter(state)

        imgui.new_frame()

        # UI workspace layout
        self.workspace.render(
            state=state,
            dispatch=self.store.dispatch,
            camera=self.camera,
            view_config=self.view_config,
            app=self,
        )

        # Render 3D scene using adapter (vectors from state)
        render_image = state.processed_image
        color_source = None
        if state.active_image_tab == "raw" or state.processed_image is None:
            render_image = state.current_image
        else:
            if state.current_image and state.image_color_mode == "rgb":
                color_source = state.current_image
            current_step = get_current_step(state)
            if current_step is not None:
                if current_step.output_data is not None:
                    render_image = current_step.output_data
                elif current_step.input_data is not None:
                    render_image = current_step.input_data
        self.renderer.render(
            scene_adapter,
            image_data=render_image,
            show_image_on_grid=state.show_image_on_grid,
            image_render_scale=state.image_render_scale,
            image_color_mode=state.image_color_mode,
            image_color_source=color_source,
            image_render_mode=state.image_render_mode,
            show_image_grid_overlay=state.show_image_grid_overlay,
        )

        try:
            self.fps = 1.0 / max(delta_time, 1e-6)
        except Exception:
            self.fps = 0.0

        # Draw labels using vectors from adapter
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
            self.labels.draw_vector_labels(
                self.camera, scene_adapter.vectors, fb_w, fb_h,
                selected_vector=scene_adapter.selected_object
            )

        imgui.render()
        self.imgui.render(imgui.get_draw_data())
        glfw.swap_buffers(self.window)

    self.imgui.shutdown()
    glfw.terminate()
    dlog("[App] shutdown")
