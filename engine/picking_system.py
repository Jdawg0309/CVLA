"""
Enhanced picking system for vectors, planes, and other objects
"""

import numpy as np


def pick_vector(screen_x, screen_y, width, height, camera, vectors, radius_px=25):
    """
    Pick a vector by screen coordinates with improved accuracy.
    
    Args:
        screen_x, screen_y: Screen coordinates
        width, height: Viewport dimensions
        camera: Camera object
        vectors: List of Vector3D objects
        radius_px: Pick radius in pixels
    
    Returns:
        Selected Vector3D or None
    """
    if not vectors:
        return None
    
    best_vector = None
    best_distance = float('inf')
    radius_sq = radius_px * radius_px
    
    for vector in vectors:
        if not vector.visible:
            continue
        
        # Get vector tip position
        screen_pos = camera.world_to_screen(vector.coords, width, height)
        if screen_pos is None:
            continue  # Behind camera or invalid
        
        x, y, depth = screen_pos
        
        # Calculate squared distance
        dx = x - screen_x
        dy = y - screen_y
        distance_sq = dx * dx + dy * dy
        
        # Check if within pick radius and closer than previous best
        if distance_sq < radius_sq and distance_sq < best_distance:
            # Additional check: prefer vectors not obscured by depth
            if depth > -0.5:  # Not too far behind
                best_vector = vector
                best_distance = distance_sq
    
    return best_vector


def pick_object(screen_x, screen_y, width, height, camera, scene):
    """
    Pick any object in the scene (vector, plane, etc.).
    
    Returns:
        (object, type) or (None, None)
    """
    # First try to pick a vector
    vector = pick_vector(screen_x, screen_y, width, height, camera, scene.vectors)
    if vector:
        return vector, 'vector'
    
    # Could add plane picking here in the future
    # For now, return None
    return None, None


def ray_intersect_plane(ray_origin, ray_dir, plane_eq):
    """
    Check if a ray intersects a plane.
    
    Args:
        ray_origin: Ray starting point
        ray_dir: Ray direction (normalized)
        plane_eq: Plane equation [a, b, c, d]
    
    Returns:
        Intersection point or None
    """
    a, b, c, d = plane_eq
    normal = np.array([a, b, c])
    
    denom = np.dot(normal, ray_dir)
    if abs(denom) < 1e-6:
        return None  # Ray is parallel to plane
    
    # Distance from ray origin to plane
    t = -(np.dot(normal, ray_origin) + d) / denom
    
    if t < 0:
        return None  # Intersection behind ray origin
    
    return ray_origin + ray_dir * t


def get_nearest_point_on_line(point, line_start, line_end):
    """
    Find the nearest point on a line segment.
    
    Args:
        point: The point
        line_start: Line start point
        line_end: Line end point
    
    Returns:
        Nearest point on line segment
    """
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-6:
        return line_start
    
    line_dir = line_vec / line_len
    point_vec = point - line_start
    projection = np.dot(point_vec, line_dir)
    
    # Clamp to line segment
    projection = np.clip(projection, 0, line_len)
    
    return line_start + line_dir * projection