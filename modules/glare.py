import cv2
import numpy as np


def _extract_quad_region_bgr(image_bgr: np.ndarray, quad) -> np.ndarray:
    pts = np.array(quad, dtype="float32")
    tl, tr, br, bl = pts

    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    max_w = int(max(width_top, width_bot))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_h = int(max(height_left, height_right))

    if max_w <= 0 or max_h <= 0:
        return image_bgr

    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image_bgr, M, (max_w, max_h))


def analyze(image_bgr: np.ndarray, card_quad=None, face_bbox=None) -> dict:
    """
    Detect glare in the image, specifically localized to the card region if available.
    Also calculates if glare overlaps with the given face bouncing box.
    
    Parameters
    ----------
    image_bgr : Full BGR image
    card_quad : Optional card quadrilateral [[x,y]...]
    face_bbox : Optional face bounding box (x1, y1, x2, y2)
    
    Returns
    -------
    dict with:
        glare_detected : bool
        glare_percentage : float
        glare_on_face : bool
        glare_percentage_face : float
    """
    if card_quad is not None:
        region = _extract_quad_region_bgr(image_bgr, card_quad)
    else:
        region = image_bgr
        
    if region is None or region.size == 0:
        return {
            "glare_detected": False,
            "glare_percentage": 0.0,
            "glare_on_face": False,
            "glare_percentage_face": 0.0
        }

    # Convert to grayscale to find bright spots
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Apply a light blur to reduce noise
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold for glare (very bright pixels, e.g., > 240)
    # The threshold might need tuning. 240 is a common threshold for blown-out highlights.
    threshold_value = 240
    _, glare_mask = cv2.threshold(gray_blurred, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Calculate percentage of glare pixels
    glare_pixels = cv2.countNonZero(glare_mask)
    total_pixels = region.shape[0] * region.shape[1]
    
    glare_percentage = (glare_pixels / float(total_pixels)) * 100.0 if total_pixels > 0 else 0.0
    
    # If any pixel in the card area is glare, we flag it. 
    glare_detected = bool(glare_percentage > 0.0)
    
    # Calculate glare on face if bbox is provided
    glare_on_face = False
    glare_percentage_face = 0.0
    
    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        ch, cw = region.shape[:2]

        fx1 = max(0, min(x1, cw))
        fy1 = max(0, min(y1, ch))
        fx2 = max(0, min(x2, cw))
        fy2 = max(0, min(y2, ch))
        
        face_mask = glare_mask[fy1:fy2, fx1:fx2]
        if face_mask is not None and face_mask.size > 0:
            face_pixels = cv2.countNonZero(face_mask)
            total_face_pixels = face_mask.size
            if total_face_pixels > 0:
                glare_percentage_face = (face_pixels / float(total_face_pixels)) * 100.0
                glare_on_face = bool(glare_percentage_face > 0.0)
    
    return {
        "glare_detected": glare_detected,
        "glare_percentage": round(glare_percentage, 4),
        "glare_on_face": glare_on_face,
        "glare_percentage_face": round(glare_percentage_face, 4)
    }
