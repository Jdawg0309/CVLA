import os
import time
import numpy as np
from pyrr import Matrix44, Vector3

DEBUG = os.getenv("CVLA_DEBUG") == "1"


class Camera:
    def __init__(self):
        # Orbit parameters
        self.radius = 15.0  # Increased for better cubic view
        self.theta = np.pi / 4  # azimuthal angle
        self.phi = np.pi / 3    # polar angle

        # Camera target (center of the world)
        self.target = np.zeros(3, dtype=np.float32)

        # Viewport
        self.width = 1440
        self.height = 900
        self.aspect = self.width / self.height

        # Mode flags
        self.mode_2d = False
        self.view_preset = "cube"  # "xy", "xz", "yz", "cube", "3d_free"
        self.lock_up_vector = True

        # Ortho scale controls the "zoom" in 2D
        self.ortho_scale = 10.0  # Increased for cubic view

        # Camera sensitivity
        self.orbit_sensitivity = 0.005
        self.pan_sensitivity = 0.01
        self.zoom_sensitivity = 0.90

        # Up vector for 3D mode
        self.up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Smoothing
        self.target_smooth = self.target.copy()
        self.smooth_factor = 0.1

        # Cubic view specific
        self.cubic_mode = True  # Enable enhanced cubic navigation
        self.cubic_rotation_speed = 1.0
        self.last_cubic_rotation = 0.0

        if DEBUG:
            print("[Camera] initialized")

    def set_viewport(self, width: int, height: int):
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.aspect = self.width / self.height

    # -----------------------------
    # Camera positioning
    # -----------------------------
    def position(self) -> np.ndarray:
        """Returns camera position in world space."""
        if self.mode_2d:
            # In 2D mode, camera position depends on the view preset
            if self.view_preset == "xy":
                # Looking down at XY plane from positive Z
                return np.array([0.0, 0.0, self.radius], dtype=np.float32)
            elif self.view_preset == "xz":
                # Looking at XZ plane from positive Y
                return np.array([0.0, self.radius, 0.0], dtype=np.float32)
            elif self.view_preset == "yz":
                # Looking at YZ plane from positive X
                return np.array([self.radius, 0.0, 0.0], dtype=np.float32)
            else:  # cube or default
                # Default to XY view
                return np.array([0.0, 0.0, self.radius], dtype=np.float32)

        # 3D mode: spherical coordinates
        # Clamp phi to avoid singularities
        self.phi = float(np.clip(self.phi, 0.10, np.pi - 0.10))

        x = self.radius * np.sin(self.phi) * np.cos(self.theta)
        y = self.radius * np.cos(self.phi)
        z = self.radius * np.sin(self.phi) * np.sin(self.theta)
        
        pos = np.array([x, y, z], dtype=np.float32)
        
        # Apply target offset
        pos += self.target_smooth
        
        return pos

    # -----------------------------
    # Matrices
    # -----------------------------
    def vp(self) -> np.ndarray:
        """Returns ViewProjection matrix (float32 4x4)."""
        pos = self.position()
        target = self.target_smooth

        if self.mode_2d:
            # 2D mode: orthographic projection
            up = self._get_2d_up_vector()
            view = Matrix44.look_at(pos, target, up)

            s = float(self.ortho_scale)
            # Ortho bounds respect aspect ratio
            left = -s * self.aspect
            right = s * self.aspect
            bottom = -s
            top = s
            near, far = 0.1, 500.0
            proj = Matrix44.orthogonal_projection(left, right, bottom, top, near, far)
            return np.array(proj * view, dtype=np.float32)

        # 3D mode: perspective projection
        # Adjust FOV based on cubic mode for better viewing
        fov = 45.0 if self.cubic_mode and self.view_preset == "cube" else 50.0
        view = Matrix44.look_at(pos, target, self.up_vector)
        near, far = 0.1, 500.0
        proj = Matrix44.perspective_projection(fov, self.aspect, near, far)
        return np.array(proj * view, dtype=np.float32)

    def _get_2d_up_vector(self):
        """Get the up vector for 2D mode based on view preset."""
        if self.view_preset == "xy":
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Y up
        elif self.view_preset == "xz":
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Z up
        elif self.view_preset == "yz":
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Z up
        else:  # cube or default
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Y up

    # -----------------------------
    # Controls - Enhanced for cubic view
    # -----------------------------
    def orbit(self, dx: float, dy: float):
        """Orbit around target with cubic view enhancements."""
        if self.mode_2d:
            return
        
        # Apply sensitivity with cubic mode multiplier
        sensitivity = self.orbit_sensitivity
        if self.cubic_mode and self.view_preset == "cube":
            sensitivity *= 1.5  # More responsive in cubic view
        
        self.theta += float(dx) * sensitivity
        self.phi -= float(dy) * sensitivity
        
        # Keep phi in safe range
        self.phi = float(np.clip(self.phi, 0.10, np.pi - 0.10))
        
        # Keep theta in reasonable range
        if self.theta > 2 * np.pi:
            self.theta -= 2 * np.pi
        elif self.theta < -2 * np.pi:
            self.theta += 2 * np.pi
        
        # Update cubic rotation tracking
        if self.cubic_mode:
            self.last_cubic_rotation = time.time()

    def pan(self, dx: float, dy: float):
        """Pan the camera (move target)."""
        if self.mode_2d:
            return
            
        # Calculate camera right and up vectors
        pos = self.position()
        forward = Vector3(self.target) - Vector3(pos)
        forward.normalize()
        
        right = Vector3(forward).cross(Vector3(self.up_vector))
        right.normalize()
        
        up = Vector3(right).cross(Vector3(forward))
        up.normalize()
        
        # Calculate pan amount based on current distance
        pan_speed = self.radius * self.pan_sensitivity * 0.01
        
        # Apply pan
        self.target[0] -= float(right.x) * dx * pan_speed
        self.target[1] += float(up.y) * dy * pan_speed
        self.target[2] -= float(right.z) * dx * pan_speed
        
        # Smooth target movement
        self.target_smooth = self.target_smooth * (1 - self.smooth_factor) + self.target * self.smooth_factor

    def zoom(self, scroll_y: float):
        """Zoom in/out with cubic view adjustments."""
        amt = float(scroll_y)
        
        # Adjust sensitivity based on mode
        zoom_sensitivity = self.zoom_sensitivity
        if self.cubic_mode and self.view_preset == "cube":
            zoom_sensitivity = 0.93  # Slower zoom for cubic view

        if self.mode_2d:
            # In 2D mode, change orthographic scale
            self.ortho_scale *= (zoom_sensitivity ** amt)
            self.ortho_scale = float(np.clip(self.ortho_scale, 1.0, 100.0))
        else:
            # In 3D mode, change radius
            self.radius *= (zoom_sensitivity ** amt)
            self.radius = float(np.clip(self.radius, 1.0, 500.0))

    def set_view_preset(self, preset: str):
        """Set the view preset with enhanced cubic view positioning."""
        self.view_preset = preset.lower()
        
        if preset == "xy":
            self.theta = np.pi / 4
            self.phi = np.pi / 2 - 0.01  # Almost top-down
            self.up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            self.cubic_mode = False
        elif preset == "xz":
            self.theta = 0
            self.phi = np.pi / 2
            self.up_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self.cubic_mode = False
        elif preset == "yz":
            self.theta = np.pi / 2
            self.phi = np.pi / 2
            self.up_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self.cubic_mode = False
        elif preset == "cube":
            # Optimal cubic view angles
            self.theta = np.pi / 4 + np.pi / 8  # Slight adjustment for better 3D view
            self.phi = np.pi / 4  # More angled view
            self.up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            self.cubic_mode = True
            self.mode_2d = False
        elif preset == "3d_free":
            # Free 3D view
            self.theta = np.pi / 4
            self.phi = np.pi / 3
            self.up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            self.cubic_mode = True
            self.mode_2d = False
        
        # Reset zoom/radius
        self.radius = 15.0 if self.cubic_mode else 12.0
        self.ortho_scale = 10.0 if self.cubic_mode else 8.0
        self.target = np.zeros(3, dtype=np.float32)
        self.target_smooth = self.target.copy()

    def cubic_view_rotation(self, auto_rotate=False, speed=0.5):
        """Perform automatic rotation for cubic view demonstration."""
        if auto_rotate and self.cubic_mode and self.view_preset == "cube":
            import time
            current_time = time.time()
            delta = current_time - self.last_cubic_rotation
            self.last_cubic_rotation = current_time
            
            # Slow automatic rotation
            self.theta += delta * speed * 0.5
            if self.theta > 2 * np.pi:
                self.theta -= 2 * np.pi

    def reset(self):
        """Reset camera to default position."""
        self.radius = 15.0
        self.theta = np.pi / 4 + np.pi / 8  # Better cubic view angle
        self.phi = np.pi / 4
        self.ortho_scale = 10.0
        self.target = np.zeros(3, dtype=np.float32)
        self.target_smooth = self.target.copy()
        self.up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.mode_2d = False
        self.view_preset = "cube"
        self.cubic_mode = True

    def focus_on_vector(self, vector_coords):
        """Focus camera on a specific vector with cubic view adjustments."""
        if vector_coords is not None:
            # Set target to the vector's tip
            self.target = np.array(vector_coords, dtype=np.float32) * 0.5
            self.target_smooth = self.target.copy()
            
            # Adjust radius to fit the vector in view
            vector_length = np.linalg.norm(vector_coords)
            self.radius = max(4.0, vector_length * 2.0)  # More space in cubic view
            
            # In cubic view, maintain good viewing angle
            if self.cubic_mode:
                self.phi = np.pi / 4  # Maintain good 3D angle
                self.theta = np.pi / 4 + np.pi / 8

    def world_to_screen(self, world_pos, width, height):
        """Convert world coordinates to screen coordinates."""
        vp = self.vp()
        p = np.array([*world_pos, 1.0], dtype=np.float32)
        clip = vp @ p

        if abs(clip[3]) < 1e-6:
            return None

        ndc = clip[:3] / clip[3]
        
        # Check if behind camera
        if ndc[2] < -1.0:
            return None
            
        x = (ndc[0] * 0.5 + 0.5) * width
        y = (1.0 - (ndc[1] * 0.5 + 0.5)) * height
        return (x, y, ndc[2])

    def screen_to_ray(self, screen_x, screen_y, width, height):
        """Convert screen coordinates to a ray in world space."""
        # Normalized device coordinates
        x = (2.0 * screen_x) / width - 1.0
        y = 1.0 - (2.0 * screen_y) / height
        z = 1.0
        
        # Clip coordinates
        ray_clip = np.array([x, y, -1.0, 1.0], dtype=np.float32)
        
        # Eye coordinates
        vp = self.vp()
        inv_proj_view = np.linalg.inv(vp)
        ray_eye = inv_proj_view @ ray_clip
        ray_eye = ray_eye / ray_eye[3]
        ray_eye[2] = -1.0
        ray_eye[3] = 0.0
        
        # World coordinates
        ray_world = inv_proj_view @ ray_eye
        ray_world = ray_world[:3]
        ray_world = ray_world / np.linalg.norm(ray_world)
        
        return self.position(), ray_world
    
    def get_view_matrix(self):
        """Get just the view matrix (without projection)."""
        pos = self.position()
        target = self.target_smooth
        return np.array(Matrix44.look_at(pos, target, self.up_vector), dtype=np.float32)
    
    def get_projection_matrix(self):
        """Get just the projection matrix."""
        if self.mode_2d:
            s = float(self.ortho_scale)
            left = -s * self.aspect
            right = s * self.aspect
            bottom = -s
            top = s
            near, far = 0.1, 500.0
            return np.array(Matrix44.orthogonal_projection(left, right, bottom, top, near, far), 
                          dtype=np.float32)
        else:
            fov = 45.0 if self.cubic_mode and self.view_preset == "cube" else 50.0
            near, far = 0.1, 500.0
            return np.array(Matrix44.perspective_projection(fov, self.aspect, near, far), 
                          dtype=np.float32)