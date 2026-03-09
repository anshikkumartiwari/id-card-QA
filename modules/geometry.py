import numpy as np
import math

def analyze(card_quad: list) -> dict:
    """
    Evaluate the geometric properties of the detected ID card.
    Specifically checks the aspect ratio and estimates rotation based on the quad points.

    Expected aspect ratio: 89.5 / 54.0 ≈ 1.657

    Parameters
    ----------
    card_quad : list of 4 points [[x,y], [x,y], [x,y], [x,y]]
                Represents the corners of the detected ID card.

    Returns
    -------
    dict with:
        geometry_adequate: bool
        aspect_ratio: float
        is_ratio_valid: bool
        rotation_detected: bool
        message: str
    """
    if card_quad is None or len(card_quad) != 4:
        return {
            "geometry_adequate": False,
            "aspect_ratio": 0.0,
            "is_ratio_valid": False,
            "rotation_detected": False,
            "message": "Card not detected or invalid quad."
        }

    pts = np.array(card_quad, dtype="float32")
    tl, tr, br, bl = pts

    # Calculate widths and heights
    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    width = (width_top + width_bot) / 2.0

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height = (height_left + height_right) / 2.0

    if height == 0 or width == 0:
        return {
            "geometry_adequate": False,
            "aspect_ratio": 0.0,
            "is_ratio_valid": False,
            "rotation_detected": False,
            "message": "Invalid card dimensions (zero)."
        }

    # ID cards are typically landscape (width > height)
    # But if an image is uploaded in portrait, we just take long/short
    long_edge = max(width, height)
    short_edge = min(width, height)
    
    aspect_ratio = long_edge / short_edge
    
    expected_ratio = 89.5 / 54.0  # ~1.657
    tolerance = 0.10 # Stricter tolerance for aspect ratio

    is_ratio_valid = bool(abs(aspect_ratio - expected_ratio) <= tolerance)

    # Calculate 2D in-plane rotation angle (tilt)
    edges = [
        (tl, tr),
        (tr, br),
        (br, bl),
        (bl, tl)
    ]
    max_len = 0
    longest_edge = None
    for p1, p2 in edges:
        length = np.linalg.norm(p2 - p1)
        if length > max_len:
            max_len = length
            longest_edge = (p1, p2)

    p1, p2 = longest_edge
    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    angle_deg = math.degrees(math.atan2(dy, dx))
    
    # Tilt is deviation from nearest horizontal/vertical axis
    tilt = abs(angle_deg) % 90
    if tilt > 45:
        tilt = 90 - tilt
        
    rotation_degrees = round(float(tilt), 1)
    
    # Consider > 5.0 degrees as significant rotation
    rotation_detected = bool(rotation_degrees > 5.0)

    # Perspective check
    width_diff_ratio = float(abs(width_top - width_bot) / max(width_top, width_bot))
    height_diff_ratio = float(abs(height_left - height_right) / max(height_left, height_right))
    perspective_distortion = bool(width_diff_ratio > 0.15 or height_diff_ratio > 0.15)

    message = "Good geometry."
    if not is_ratio_valid:
        message = f"Aspect ratio ({aspect_ratio:.2f}) deviates from expected {expected_ratio:.2f}."
    elif rotation_detected:
        message = f"Card is rotated by ~{rotation_degrees}°."
    elif perspective_distortion:
        message = "Significant perspective 3D distortion detected."

    geometry_adequate = bool(is_ratio_valid and not rotation_detected and not perspective_distortion)

    return {
        "geometry_adequate": geometry_adequate,
        "aspect_ratio": round(float(aspect_ratio), 3),
        "is_ratio_valid": is_ratio_valid,
        "rotation_degrees": rotation_degrees,
        "rotation_detected": rotation_detected,
        "message": message
    }
