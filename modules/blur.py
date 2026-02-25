import cv2
import numpy as np


# --- Compute focus score robust to defocus + motion blur
def _focus_score(gray_region: np.ndarray) -> float:
    if gray_region is None or gray_region.size == 0:
        return 0.0

    # Light smoothing to suppress sensor noise spikes
    gray_region = cv2.GaussianBlur(gray_region, (3, 3), 0)

    # Laplacian variance (defocus sensitivity)
    lap = cv2.Laplacian(gray_region, cv2.CV_64F)
    lap_var = lap.var()

    # Sobel gradients
    gx = cv2.Sobel(gray_region, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)

    grad_energy = np.mean(gx**2 + gy**2)

    # Detect directional imbalance (motion blur)
    mean_abs_gx = np.mean(np.abs(gx))
    mean_abs_gy = np.mean(np.abs(gy))

    ratio = max(mean_abs_gx, mean_abs_gy) / (min(mean_abs_gx, mean_abs_gy) + 1e-6)
    anisotropy_penalty = 1.0 / ratio

    ten_adjusted = grad_energy * anisotropy_penalty

    return float(0.55 * lap_var + 0.45 * ten_adjusted)


# --- Warp quadrilateral card to flat rectangle
def _extract_quad_region(gray: np.ndarray, quad) -> np.ndarray:
    pts = np.array(quad, dtype="float32")
    tl, tr, br, bl = pts

    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    max_w = int(max(width_top, width_bot))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_h = int(max(height_left, height_right))

    if max_w <= 0 or max_h <= 0:
        return gray

    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(gray, M, (max_w, max_h))


# --- Main API
def analyze(gray_full: np.ndarray, card_quad=None, face_bbox=None) -> dict:
    """
    Zone-aware blur analysis.

    Parameters
    ----------
    gray_full : Full grayscale image
    card_quad : Optional card quadrilateral [[x,y]...]
    face_bbox : Optional (x1, y1, x2, y2)

    Returns
    -------
    {
        blur_score_global,
        blur_score_card_zone,
        blur_score_face_zone,
        blur_composite
    }
    """

    # Global (diagnostic only, NOT used in composite)
    blur_score_global = _focus_score(gray_full)

    # --- Card zone
    blur_score_card_zone = None
    card_region = None

    if card_quad is not None:
        card_region = _extract_quad_region(gray_full, card_quad)
        if card_region is not None and card_region.size > 0:
            blur_score_card_zone = _focus_score(card_region)

    # --- Face zone
    blur_score_face_zone = None

    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox

        if card_region is not None:
            ch, cw = card_region.shape

            fx1 = max(0, min(x1, cw))
            fy1 = max(0, min(y1, ch))
            fx2 = max(0, min(x2, cw))
            fy2 = max(0, min(y2, ch))

            face_crop = card_region[fy1:fy2, fx1:fx2]
        else:
            h, w = gray_full.shape
            fx1 = max(0, min(x1, w))
            fy1 = max(0, min(y1, h))
            fx2 = max(0, min(x2, w))
            fy2 = max(0, min(y2, h))

            face_crop = gray_full[fy1:fy2, fx1:fx2]

        if face_crop is not None and face_crop.size > 0:
            blur_score_face_zone = _focus_score(face_crop)

    # --- Composite (NO global influence)
    if blur_score_card_zone is not None and blur_score_face_zone is not None:
        blur_composite = (
            0.6 * blur_score_card_zone +
            0.4 * blur_score_face_zone
        )
    elif blur_score_card_zone is not None:
        blur_composite = blur_score_card_zone
    elif blur_score_face_zone is not None:
        blur_composite = blur_score_face_zone
    else:
        blur_composite = blur_score_global  # fallback only

    return {
        "blur_score_global": round(blur_score_global, 4),
        "blur_score_card_zone": round(blur_score_card_zone, 4) if blur_score_card_zone is not None else None,
        "blur_score_face_zone": round(blur_score_face_zone, 4) if blur_score_face_zone is not None else None,
        "blur_composite": round(blur_composite, 4)
    }