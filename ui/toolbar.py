"""
Top toolbar for quick access to common operations
"""

import imgui
import numpy as np


class Toolbar:
    def __init__(self):
        self.show_camera_info = False
        self.show_stats = True
        self.show_quick_actions = True
        
    def render(self, scene, camera, view_config, app):
        """Render toolbar at top of screen."""
        # Set window at top
        imgui.set_next_window_position(10, 10)
        imgui.set_next_window_size(imgui.get_io().display_size.x - 20, 40)
        
        # Transparent, no background
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0, 0, 0, 0)
        imgui.push_style_color(imgui.COLOR_BORDER, 0, 0, 0, 0)
        
        if imgui.begin("Toolbar", 
                      flags=imgui.WINDOW_NO_TITLE_BAR | 
                            imgui.WINDOW_NO_RESIZE |
                            imgui.WINDOW_NO_MOVE |
                            imgui.WINDOW_NO_SCROLLBAR |
                            imgui.WINDOW_NO_SAVED_SETTINGS):
            
            # Left side: Application title and mode
            imgui.text_colored("CVLA", 0.9, 0.9, 1.0, 1.0)
            imgui.same_line()
            imgui.text_disabled("|")
            imgui.same_line()
            
            mode_text = "2D" if camera.mode_2d else "3D"
            imgui.text(f"Mode: {mode_text}")
            imgui.same_line()
            imgui.text_disabled("|")
            imgui.same_line()
            
            # View info
            view_text = f"View: {view_config.grid_plane.upper()}" if view_config.grid_mode == "plane" else "View: 3D"
            imgui.text(view_text)
            
            # Middle: Quick action buttons
            imgui.same_line(imgui.get_io().display_size.x / 2 - 100)
            
            quick_actions = [
                ("🎯", "Focus", lambda: camera.focus_on_vector(
                    scene.selected_object.coords if scene.selected_object else None
                )),
                ("🔄", "Reset", camera.reset),
                ("📐", "Grid", lambda: view_config.update(show_grid=not view_config.show_grid)),
                ("📏", "Axes", lambda: view_config.update(show_axes=not view_config.show_axes)),
                ("🏷️", "Labels", lambda: view_config.update(show_labels=not view_config.show_labels)),
            ]
            
            for icon, tooltip, action in quick_actions:
                if imgui.button(icon):
                    action()
                
                if imgui.is_item_hovered():
                    imgui.set_tooltip(tooltip)
                
                imgui.same_line()
            
            # Right side: Stats and info
            imgui.same_line(imgui.get_io().display_size.x - 200)
            
            # FPS
            if hasattr(app, 'fps'):
                fps_color = (0.2, 0.8, 0.2, 1.0) if app.fps > 30 else (0.8, 0.2, 0.2, 1.0)
                imgui.text_colored(f"{app.fps:.0f} FPS", *fps_color)
                imgui.same_line()
            
            # Vector count
            imgui.text(f"Vectors: {len(scene.vectors)}")
            imgui.same_line()
            
            # Help button
            if imgui.button("?"):
                app.show_help = not app.show_help
            
        imgui.end()
        imgui.pop_style_color(2)
