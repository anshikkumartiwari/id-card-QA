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


def _estimate_noise(image: np.ndarray) -> float:
    """
    Estimate image noise using Laplacian variance or a similar metric.
    Here we use a common method: compute the standard deviation of the 
    laplacian (high pass filter). Higher values usually mean MORE detail/edges,
    but computing the variance of the residuals after median filtering 
    often gives a better 'noise' estimate.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Median filter removes noise but keeps edges
    filtered = cv2.medianBlur(gray, 3)
    
    # The absolute difference is the "noise" (residuals)
    residual = cv2.absdiff(gray, filtered)
    
    # Calculate the mean of the residuals. 
    # Higher mean -> more noise/grain.
    noise_level = np.mean(residual)
    return float(noise_level)


def analyze(image_bgr: np.ndarray, card_quad=None, face_bbox=None) -> dict:
    """
    Detect noise levels in the image, specifically localized to the card region and face region.
    
    Parameters
    ----------
    image_bgr : Full BGR image
    card_quad : Optional card quadrilateral [[x,y]...]
    face_bbox : Optional face bounding box (x1, y1, x2, y2)
    
    Returns
    -------
    dict with:
        noise_level_global : float
        noise_level_card : float
        noise_level_face : float
        noise_detected : bool
    """
    
    noise_global = _estimate_noise(image_bgr)
    
    noise_card = 0.0
    region = None
    if card_quad is not None:
        region = _extract_quad_region_bgr(image_bgr, card_quad)
        if region is not None and region.size > 0:
            noise_card = _estimate_noise(region)
    else:
        region = image_bgr
        noise_card = noise_global

    noise_face = 0.0
    if face_bbox is not None and region is not None and region.size > 0:
        x1, y1, x2, y2 = face_bbox
        ch, cw = region.shape[:2]

        fx1 = max(0, min(x1, cw))
        fy1 = max(0, min(y1, ch))
        fx2 = max(0, min(x2, cw))
        fy2 = max(0, min(y2, ch))
        
        face_crop = region[fy1:fy2, fx1:fx2]
        if face_crop is not None and face_crop.size > 0:
            noise_face = _estimate_noise(face_crop)

    # Threshold for what is considered "noisy" (needs empirical tuning)
    # A residual mean > 5.0 is often visibly noisy
    noise_detected = bool(noise_card > 5.0 or noise_face > 5.0)

    return {
        "noise_level_global": round(noise_global, 4),
        "noise_level_card": round(noise_card, 4),
        "noise_level_face": round(noise_face, 4),
        "noise_detected": noise_detected
    }
