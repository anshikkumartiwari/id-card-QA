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
            "aspect_ratio": 0.0,
            "rotation_degrees": 0.0,
            "width_diff_ratio": 0.0,
            "height_diff_ratio": 0.0
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
            "aspect_ratio": 0.0,
            "rotation_degrees": 0.0,
            "width_diff_ratio": 0.0,
            "height_diff_ratio": 0.0
        }

    # ID cards are typically landscape (width > height)
    # But if an image is uploaded in portrait, we just take long/short
    long_edge = max(width, height)
    short_edge = min(width, height)
    
    aspect_ratio = long_edge / short_edge
    
    aspect_ratio = long_edge / short_edge

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
    
    # Perspective check
    width_diff_ratio = float(abs(width_top - width_bot) / max(width_top, width_bot))
    height_diff_ratio = float(abs(height_left - height_right) / max(height_left, height_right))

    return {
        "aspect_ratio": round(float(aspect_ratio), 3),
        "rotation_degrees": rotation_degrees,
        "width_diff_ratio": round(width_diff_ratio, 4),
        "height_diff_ratio": round(height_diff_ratio, 4)
    }
