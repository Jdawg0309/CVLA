"""View tab - visualization settings and camera controls."""

import imgui
from typing import Optional, Callable, Any

from ui.ribbon.ribbon_tab import RibbonTab
from ui.ribbon.ribbon_group import RibbonGroup
from ui.ribbon.ribbon_button import RibbonButton


class ViewTab(RibbonTab):
    """View tab with grid, axes, camera, and display settings."""

    def __init__(self):
        groups = [
            RibbonGroup("Presets", [
                RibbonButton("Front", "Fr", tooltip="Front view"),
                RibbonButton("Top", "Top", tooltip="Top view"),
                RibbonButton("Right", "Rt", tooltip="Right view"),
                RibbonButton("Iso", "3D", tooltip="Isometric view"),
            ]),
            RibbonGroup("Grid", [
                RibbonButton("Grid", "Grd", tooltip="Toggle grid", is_toggle=True),
                RibbonButton("Axes", "Ax", tooltip="Toggle axes", is_toggle=True),
                RibbonButton("Labels", "Lbl", tooltip="Toggle labels", is_toggle=True),
            ]),
            RibbonGroup("Display", [
                RibbonButton("Image", "Img", tooltip="Toggle image on grid", is_toggle=True),
                RibbonButton("Cubic", "Cub", tooltip="Cubic view mode", is_toggle=True),
            ]),
        ]
        super().__init__(groups)

    def render(
        self,
        state: Any,
        dispatch: Optional[Callable] = None,
        camera: Any = None,
        view_config: Any = None,
    ) -> None:
        """Render View tab with visualization controls."""
        # Camera presets
        imgui.begin_group()

        if imgui.button("Fr\nFront", 50, 56) and camera:
            camera.set_preset("front")

        if imgui.is_item_hovered():
            imgui.set_tooltip("Front view (looking at XY plane)")

        imgui.same_line()

        if imgui.button("Top\nTop", 50, 56) and camera:
            camera.set_preset("top")

        if imgui.is_item_hovered():
            imgui.set_tooltip("Top view (looking down at XZ plane)")

        imgui.same_line()

        if imgui.button("Rt\nRight", 50, 56) and camera:
            camera.set_preset("right")

        if imgui.is_item_hovered():
            imgui.set_tooltip("Right view (looking at YZ plane)")

        imgui.same_line()

        if imgui.button("3D\nIso", 50, 56) and camera:
            camera.set_preset("isometric")

        if imgui.is_item_hovered():
            imgui.set_tooltip("Isometric 3D view")

        imgui.spacing()
        imgui.text_disabled("Presets")
        imgui.end_group()

        imgui.same_line()
        imgui.dummy(16, 1)
        imgui.same_line()

        # Grid/Axes/Labels toggles
        imgui.begin_group()

        show_grid = view_config.show_grid if view_config else True
        if show_grid:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.5, 0.7, 1.0)

        if imgui.button("Grd\nGrid", 50, 56) and view_config:
            view_config.update(show_grid=not show_grid)

        if show_grid:
            imgui.pop_style_color()

        if imgui.is_item_hovered():
            imgui.set_tooltip("Toggle coordinate grid")

        imgui.same_line()

        show_axes = view_config.show_axes if view_config else True
        if show_axes:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.5, 0.7, 1.0)

        if imgui.button("Ax\nAxes", 50, 56) and view_config:
            view_config.update(show_axes=not show_axes)

        if show_axes:
            imgui.pop_style_color()

        if imgui.is_item_hovered():
            imgui.set_tooltip("Toggle coordinate axes")

        imgui.same_line()

        show_labels = view_config.show_labels if view_config else True
        if show_labels:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.5, 0.7, 1.0)

        if imgui.button("Lbl\nLabels", 50, 56) and view_config:
            view_config.update(show_labels=not show_labels)

        if show_labels:
            imgui.pop_style_color()

        if imgui.is_item_hovered():
            imgui.set_tooltip("Toggle vector labels")

        imgui.spacing()
        imgui.text_disabled("Grid")
        imgui.end_group()

        imgui.same_line()
        imgui.dummy(16, 1)
        imgui.same_line()

        # Display options
        imgui.begin_group()

        show_image = state.show_image_on_grid if state else True
        if show_image:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.5, 0.7, 1.0)

        if imgui.button("Img\nImage", 50, 56):
            if state:
                from state.actions import ToggleImageOnGrid
                if dispatch:
                    dispatch(ToggleImageOnGrid())

        if show_image:
            imgui.pop_style_color()

        if imgui.is_item_hovered():
            imgui.set_tooltip("Toggle image display on grid")

        imgui.same_line()

        cubic_mode = view_config.cubic_mode if view_config and hasattr(view_config, 'cubic_mode') else False
        if cubic_mode:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.5, 0.7, 1.0)

        if imgui.button("Cub\nCubic", 50, 56) and view_config:
            if hasattr(view_config, 'toggle_cubic_mode'):
                view_config.toggle_cubic_mode()

        if cubic_mode:
            imgui.pop_style_color()

        if imgui.is_item_hovered():
            imgui.set_tooltip("Toggle cubic view mode")

        imgui.spacing()
        imgui.text_disabled("Display")
        imgui.end_group()

        imgui.same_line()
        imgui.dummy(16, 1)
        imgui.same_line()

        # Camera info and controls
        imgui.begin_group()
        imgui.text("Camera:")

        if camera:
            imgui.text_disabled(f"Dist: {camera.distance:.1f}")

            imgui.push_item_width(80)
            changed, new_dist = imgui.slider_float("##dist", camera.distance, 2.0, 20.0, "%.1f")
            imgui.pop_item_width()
            if changed:
                camera.distance = new_dist

            if imgui.button("Reset", 60, 20):
                camera.reset()

        imgui.spacing()
        imgui.text_disabled("Camera")
        imgui.end_group()

        imgui.same_line()
        imgui.dummy(16, 1)
        imgui.same_line()

        # Auto-rotate
        imgui.begin_group()

        auto_rotate = view_config.auto_rotate if view_config and hasattr(view_config, 'auto_rotate') else False
        if auto_rotate:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.5, 0.7, 1.0)

        if imgui.button("Auto\nRotate", 56, 56):
            if view_config and hasattr(view_config, 'auto_rotate'):
                view_config.update(auto_rotate=not auto_rotate)

        if auto_rotate:
            imgui.pop_style_color()

        if imgui.is_item_hovered():
            imgui.set_tooltip("Toggle automatic rotation")

        imgui.spacing()
        imgui.text_disabled("Animate")
        imgui.end_group()
