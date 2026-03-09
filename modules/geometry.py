import numpy as np

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
    tolerance = 0.55 # Using the same tolerance from card_detection.py

    is_ratio_valid = bool(abs(aspect_ratio - expected_ratio) <= tolerance)

    # Estimate rotation / sheer
    # If the top/bottom widths or left/right heights differ significantly, 
    # the card is likely rotated in 3D space (perspective distortion).
    width_diff_ratio = float(abs(width_top - width_bot) / max(width_top, width_bot))
    height_diff_ratio = float(abs(height_left - height_right) / max(height_left, height_right))

    # Threshold for perspective rotation
    # If opposite edges differ by more than ~15%, flag as rotated
    rotation_detected = bool(width_diff_ratio > 0.15 or height_diff_ratio > 0.15)

    message = "Good geometry."
    if not is_ratio_valid:
        message = f"Aspect ratio ({aspect_ratio:.2f}) deviates from expected {expected_ratio:.2f}."
    elif rotation_detected:
        message = "Significant perspective rotation detected."

    geometry_adequate = bool(is_ratio_valid and not rotation_detected)

    return {
        "geometry_adequate": geometry_adequate,
        "aspect_ratio": round(float(aspect_ratio), 3),
        "is_ratio_valid": is_ratio_valid,
        "rotation_detected": rotation_detected,
        "message": message
    }
