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

        state = self.store.get_state()
        view_snapshot = (
            state.view_preset,
            state.view_up_axis,
            state.view_grid_mode,
            state.view_grid_plane,
            state.view_show_grid,
            state.view_show_axes,
            state.view_show_labels,
            state.view_grid_size,
            state.view_base_major_tick,
            state.view_base_minor_tick,
            state.view_major_tick,
            state.view_minor_tick,
            state.view_auto_rotate,
            state.view_rotation_speed,
            state.view_show_cube_faces,
            state.view_show_cube_corners,
            state.view_cubic_grid_density,
            state.view_cube_face_opacity,
            state.view_mode_2d,
        )
        if view_snapshot != self._last_view_state:
            if state.view_preset != self._last_view_preset:
                self.camera.set_view_preset(state.view_preset)
                self._last_view_preset = state.view_preset

            self.view_config._base_major_tick = int(state.view_base_major_tick)
            self.view_config._base_minor_tick = int(state.view_base_minor_tick)
            self.view_config.update(
                up_axis=state.view_up_axis,
                grid_mode=state.view_grid_mode,
                grid_plane=state.view_grid_plane,
                grid_size=state.view_grid_size,
                show_grid=state.view_show_grid,
                show_axes=state.view_show_axes,
                show_labels=state.view_show_labels,
                show_cube_faces=state.view_show_cube_faces,
                show_cube_corners=state.view_show_cube_corners,
                cubic_grid_density=state.view_cubic_grid_density,
                cube_face_opacity=state.view_cube_face_opacity,
                auto_rotate=state.view_auto_rotate,
                rotation_speed=state.view_rotation_speed,
            )
            self.view_config.major_tick = int(state.view_major_tick)
            self.view_config.minor_tick = int(state.view_minor_tick)
            self.view_config._base_major_tick = int(state.view_base_major_tick)
            self.view_config._base_minor_tick = int(state.view_base_minor_tick)
            self.view_config.cube_face_colors = [
                (color[0], color[1], color[2], state.view_cube_face_opacity)
                for color in self.view_config.cube_face_colors
            ]
            self.camera.mode_2d = state.view_mode_2d
            self._last_view_state = view_snapshot

        if getattr(self.view_config, 'auto_rotate', False):
            speed = getattr(self.view_config, 'rotation_speed', 0.5)
            self.camera.cubic_view_rotation(auto_rotate=True, speed=speed)

        self.renderer.update_view(self.view_config)
        self.labels.update_view(self.view_config)

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
        # Determine which image to render
        render_image = None
        color_source = None
        if state.active_image_tab == "raw" or state.processed_image is None:
            render_image = state.current_image
        else:
            # Default to processed_image when operations have been applied
            render_image = state.processed_image
            if state.current_image and state.image_color_mode == "rgb":
                color_source = state.current_image
            # Override with pipeline step if available
            current_step = get_current_step(state)
            if current_step is not None:
                if current_step.output_data is not None:
                    render_image = current_step.output_data
                elif current_step.input_data is not None:
                    render_image = current_step.input_data
        # Debug: Log image rendering info
        if render_image is not None and hasattr(render_image, 'id'):
            _img_id = getattr(render_image, 'id', 'unknown')[:8]
            _img_name = getattr(render_image, 'name', 'unknown')
            # Uncomment to debug: print(f"[CVLA] Rendering image: {_img_name} (id={_img_id})")

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
