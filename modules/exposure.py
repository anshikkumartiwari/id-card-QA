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
    Evaluate the exposure (brightness) of the ID card and the face.
    
    Parameters
    ----------
    image_bgr : Full BGR image
    card_quad : Optional card quadrilateral [[x,y]...]
    face_bbox : Optional face bounding box (x1, y1, x2, y2)
    
    Returns
    -------
    dict with:
        exposure_adequate : bool
        mean_brightness_card : float
        mean_brightness_face : float
        status : str ('good', 'underexposed', 'overexposed')
    """
    if card_quad is not None:
        region = _extract_quad_region_bgr(image_bgr, card_quad)
    else:
        region = image_bgr
        
    if region is None or region.size == 0:
        return {
            "exposure_adequate": False,
            "mean_brightness_card": 0.0,
            "mean_brightness_face": 0.0,
            "status": "unknown"
        }

    # Convert to grayscale to evaluate brightness
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    mean_brightness_card = np.mean(gray)
    
    # Calculate brightness on face if bbox is provided
    mean_brightness_face = 0.0
    
    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        ch, cw = region.shape[:2]

        fx1 = max(0, min(x1, cw))
        fy1 = max(0, min(y1, ch))
        fx2 = max(0, min(x2, cw))
        fy2 = max(0, min(y2, ch))
        
        face_roi = gray[fy1:fy2, fx1:fx2]
        if face_roi is not None and face_roi.size > 0:
            mean_brightness_face = np.mean(face_roi)
    else:
        # fallback to general card brightness if face is not found
        mean_brightness_face = mean_brightness_card
        
    # Determine exposure status based on typical brightness thresholds
    # ~0 = black, ~255 = white
    if mean_brightness_card < 60:
        status = "underexposed"
        exposure_adequate = False
    elif mean_brightness_card > 210:
        status = "overexposed"
        exposure_adequate = False
    else:
        status = "good"
        exposure_adequate = True

    return {
        "exposure_adequate": exposure_adequate,
        "mean_brightness_card": round(float(mean_brightness_card), 2),
        "mean_brightness_face": round(float(mean_brightness_face), 2),
        "status": status
    }
