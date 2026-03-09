"""
card_detect.py — Enhanced ID-card quadrilateral detector with structural color validation.

Detection pipeline:
  1. Aggressive image downscaling (multiple scales)
  2. Multi-channel preprocessing (gray, V, S, L, CLAHE)
  3. Multi-strategy edge detection (Canny, Sobel, Scharr, Laplacian)
  4. **NEW: Color-based segmentation (blue band + white card body)**
  5. Morphological consolidation
  6. Contour extraction with area filtering
  7. Ramer–Douglas–Peucker polygon approximation (multi-epsilon)
  8. Geometry heuristics (4 corners, convexity, angles, area, aspect ratio)
  9. **NEW: Structural validation (blue bottom band + white side margins)**
 10. Perspective / homography validation
 11. Hough-line intersection fallback
 12. Best-candidate selection via composite score
"""

import cv2
import numpy as np
import math
from itertools import combinations

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "expected_aspect_ratio": 89.5 / 54.0,  # ~1.657  (long / short)
    "aspect_ratio_tolerance": 0.55,
    "min_area_ratio": 0.10,      # card ≥ 1.5 % of frame
    "max_area_ratio": 0.98,       # card < 98 % of frame
    "score_accept": 0.40,         # minimum composite score
    "min_angle_deg": 60.0,        # each interior angle ≥ 60°
    "max_angle_deg": 135.0,       # each interior angle ≤ 135°
    "min_solidity": 0.75,
    "min_extent": 0.50,
    
    # Structural priors (from config.py)
    "bottom_band_rel_height": 10.0 / 89.5,   # ≈ 0.1117
    "side_margin_rel_width": 5.0 / 54.0,     # ≈ 0.0926
    "side_margin_deep_rel_width": 10.0 / 54.0,  # ≈ 0.185
    "margin_widen_y_rel_range": (20.0 / 89.5, 60.0 / 89.5),
    
    # Color thresholds
    "bottom_band_min_blue_over_red": 1.15,
    "bottom_band_min_saturation": 35,
    "bottom_band_coverage_ratio": 0.70,
    "side_margin_min_brightness": 200,
    "side_margin_max_saturation": 50,
    "side_margin_max_gradient": 10.0,
}

# Processing scales — we try several downscale heights
_PROCESSING_HEIGHTS = [500, 700, 1000]


# ===================================================================
#  Geometry helpers
# ===================================================================

def _order_points(pts):
    """Order 4 points as TL, TR, BR, BL """
    pts = np.array(pts, dtype="float32").reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]   # TL has smallest x+y
    rect[2] = pts[np.argmax(s)]   # BR has largest  x+y
    rect[1] = pts[np.argmin(d)]   # TR has smallest x-y
    rect[3] = pts[np.argmax(d)]   # BL has largest  x-y
    return rect


def _quad_perimeter(pts):
    p = 0.0
    for i in range(4):
        p += np.linalg.norm(pts[i] - pts[(i + 1) % 4])
    return p


def _angle_at_vertex(a, b, c):
    """Angle (degrees) at vertex b formed by segments a→b and c→b."""
    v1 = a - b
    v2 = c - b
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def _interior_angles(pts):
    """Return the 4 interior angles of an ordered quad."""
    angles = []
    for i in range(4):
        a = pts[(i - 1) % 4]
        b = pts[i]
        c = pts[(i + 1) % 4]
        angles.append(_angle_at_vertex(a, b, c))
    return angles


def _is_convex(pts):
    """Check if ordered quad is convex via cross-product sign consistency."""
    signs = []
    for i in range(4):
        o = pts[i]
        a = pts[(i + 1) % 4] - o
        b = pts[(i + 2) % 4] - o
        cross = a[0] * b[1] - a[1] * b[0]
        signs.append(cross)
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def _validate_quad_geometry(pts, img_w, img_h):
    """
    Hard geometric filter.  Returns False if the quad is clearly not a card.
    """
    area = cv2.contourArea(pts.astype(np.float32))
    img_area = float(img_w * img_h)
    ratio = area / img_area

    if ratio < THRESHOLDS["min_area_ratio"] or ratio > THRESHOLDS["max_area_ratio"]:
        return False

    if not _is_convex(pts):
        return False

    angles = _interior_angles(pts)
    for a in angles:
        if a < THRESHOLDS["min_angle_deg"] or a > THRESHOLDS["max_angle_deg"]:
            return False

    # Aspect ratio check (generous for perspective)
    tl, tr, br, bl = pts
    w1 = np.linalg.norm(tr - tl)
    w2 = np.linalg.norm(br - bl)
    h1 = np.linalg.norm(bl - tl)
    h2 = np.linalg.norm(br - tr)
    avg_w = (w1 + w2) / 2.0
    avg_h = (h1 + h2) / 2.0
    if min(avg_w, avg_h) < 1:
        return False
    aspect = max(avg_w, avg_h) / min(avg_w, avg_h)
    target = THRESHOLDS["expected_aspect_ratio"]
    if abs(aspect - target) / target > THRESHOLDS["aspect_ratio_tolerance"]:
        return False

    return True


def _homography_reprojection_error(pts):
    """
    Warp the quad to a canonical rectangle and measure reprojection error.
    Low error → the quad is close to a true perspective-projected rectangle.
    Returns normalised error in [0, 1] range (lower is better).
    """
    pts = pts.astype(np.float32)
    tl, tr, br, bl = pts
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if w < 10 or h < 10:
        return 1.0
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    H, mask = cv2.findHomography(pts, dst, cv2.RANSAC, 5.0)
    if H is None:
        return 1.0
    projected = cv2.perspectiveTransform(pts.reshape(1, -1, 2), H).reshape(4, 2)
    err = np.mean(np.linalg.norm(projected - dst, axis=1))
    normalised = err / (math.sqrt(w * w + h * h) + 1e-9)
    return float(np.clip(normalised, 0.0, 1.0))


# ===================================================================
#  NEW: Structural color validation helpers
# ===================================================================

def _verify_blue_bottom_band(image_bgr, quad):
    """
    Check if the bottom region of the quad has a blue band.
    Returns a score [0..1] indicating confidence.
    """
    try:
        # Warp quad to rectangle
        pts = _order_points(quad)
        tl, tr, br, bl = pts
        w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        
        if w < 50 or h < 50:
            return 0.0
        
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image_bgr, M, (w, h), 
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
        
        # Extract bottom band region (expected ~11% of card height)
        band_rel_height = THRESHOLDS["bottom_band_rel_height"]
        band_height = max(5, int(h * band_rel_height * 1.5))  # Allow some tolerance
        band_y_start = h - band_height
        bottom_band = warped[band_y_start:h, :, :]
        
        if bottom_band.size == 0:
            return 0.0
        
        # Convert to HSV and check for blue characteristics
        hsv = cv2.cvtColor(bottom_band, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # Blue hue range: 100-130 in OpenCV (180 scale)
        blue_mask = cv2.inRange(hsv, np.array([90, 30, 30]), np.array([135, 255, 255]))
        blue_ratio = np.count_nonzero(blue_mask) / float(blue_mask.size)
        
        # Also check BGR: blue > red
        b_ch, g_ch, r_ch = bottom_band[:, :, 0], bottom_band[:, :, 1], bottom_band[:, :, 2]
        blue_over_red = np.mean(b_ch.astype(float) / (r_ch.astype(float) + 1.0))
        
        # Check saturation (blue should be reasonably saturated)
        mean_saturation = np.mean(s_ch)
        
        # Combine factors
        score = 0.0
        if blue_ratio > 0.3:  # At least 30% blue pixels
            score += 0.4 * min(1.0, blue_ratio / 0.7)
        
        if blue_over_red > THRESHOLDS["bottom_band_min_blue_over_red"]:
            score += 0.3
        
        if mean_saturation > THRESHOLDS["bottom_band_min_saturation"]:
            score += 0.3
        
        return min(1.0, score)
    
    except Exception:
        return 0.0


def _verify_white_side_margins(image_bgr, quad):
    """
    Check if the card has white side margins as expected.
    Returns a score [0..1].
    """
    try:
        pts = _order_points(quad)
        tl, tr, br, bl = pts
        w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        
        if w < 50 or h < 50:
            return 0.0
        
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image_bgr, M, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
        
        # Sample left and right margins
        margin_rel_width = THRESHOLDS["side_margin_rel_width"]
        margin_width = max(3, int(w * margin_rel_width * 1.2))
        
        left_margin = warped[:, :margin_width, :]
        right_margin = warped[:, -margin_width:, :]
        
        if left_margin.size == 0 or right_margin.size == 0:
            return 0.0
        
        # Check brightness (should be high for white)
        gray_left = cv2.cvtColor(left_margin, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_margin, cv2.COLOR_BGR2GRAY)
        
        brightness_left = np.mean(gray_left)
        brightness_right = np.mean(gray_right)
        
        # Check saturation (should be low for white/gray)
        hsv_left = cv2.cvtColor(left_margin, cv2.COLOR_BGR2HSV)
        hsv_right = cv2.cvtColor(right_margin, cv2.COLOR_BGR2HSV)
        
        saturation_left = np.mean(hsv_left[:, :, 1])
        saturation_right = np.mean(hsv_right[:, :, 1])
        
        score = 0.0
        
        # Both margins should be bright
        min_brightness = THRESHOLDS["side_margin_min_brightness"]
        if brightness_left > min_brightness * 0.8:
            score += 0.25
        if brightness_right > min_brightness * 0.8:
            score += 0.25
        
        # Both margins should have low saturation
        max_saturation = THRESHOLDS["side_margin_max_saturation"]
        if saturation_left < max_saturation * 1.2:
            score += 0.25
        if saturation_right < max_saturation * 1.2:
            score += 0.25
        
        return score
    
    except Exception:
        return 0.0


def _detect_blue_band_region(image_bgr):
    """
    NEW: Color-based segmentation to find blue regions (bottom band).
    Returns a binary mask.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Blue hue range (adjusted for robustness)
    lower_blue = np.array([90, 30, 30])
    upper_blue = np.array([135, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Also check BGR ratio
    b_ch = image_bgr[:, :, 0].astype(float)
    r_ch = image_bgr[:, :, 2].astype(float)
    g_ch = image_bgr[:, :, 1].astype(float)
    
    # Blue should be stronger than red
    blue_dominant = (b_ch > r_ch * 1.1) & (b_ch > 50)
    blue_mask2 = np.uint8(blue_dominant) * 255
    
    # Combine masks
    combined = cv2.bitwise_or(blue_mask, blue_mask2)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return combined


def _detect_white_card_region(image_bgr):
    """
    NEW: Color-based segmentation to find white/light regions (card body).
    Returns a binary mask.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # White regions: high brightness, low saturation
    bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    low_sat_mask = cv2.threshold(hsv[:, :, 1], 60, 255, cv2.THRESH_BINARY_INV)[1]
    
    white_mask = cv2.bitwise_and(bright_mask, low_sat_mask)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return white_mask


# ===================================================================
#  Enhanced scoring with structural validation
# ===================================================================

def _score_candidate(pts, img_w, img_h, image_bgr=None):
    """
    Composite score [0..1] — higher means more card-like.
    NOW INCLUDES structural color validation!
    """
    pts = np.array(pts, dtype=np.float32)
    tl, tr, br, bl = pts

    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)

    if max(w_top, w_bot) < 1 or max(h_left, h_right) < 1:
        return 0.0

    # 1. Side symmetry (0..1)
    w_sym = min(w_top, w_bot) / max(w_top, w_bot)
    h_sym = min(h_left, h_right) / max(h_left, h_right)
    sym_score = (w_sym + h_sym) / 2.0

    # 2. Aspect ratio (0..1)
    avg_w = (w_top + w_bot) / 2.0
    avg_h = (h_left + h_right) / 2.0
    if min(avg_w, avg_h) < 1:
        return 0.0
    aspect = max(avg_w, avg_h) / min(avg_w, avg_h)
    target = THRESHOLDS["expected_aspect_ratio"]
    dev = abs(aspect - target) / target
    aspect_score = max(0.0, 1.0 - dev * 2.0)

    # 3. Angle quality — ideal 90° at each corner
    angles = _interior_angles(pts)
    angle_devs = [abs(a - 90.0) for a in angles]
    angle_score = max(0.0, 1.0 - (sum(angle_devs) / (4.0 * 45.0)))

    # 4. Solidity
    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)
    quad_area = cv2.contourArea(pts)
    solidity = quad_area / hull_area if hull_area > 0 else 0.0
    solidity_score = np.clip(solidity, 0.0, 1.0)

    # 5. Homography reprojection (low error is good)
    reproj = _homography_reprojection_error(pts)
    homog_score = 1.0 - reproj

    # 6. Area preference
    img_area = float(img_w * img_h)
    area_ratio = quad_area / img_area
    if area_ratio < THRESHOLDS["min_area_ratio"]:
        area_score = 0.0
    elif area_ratio > THRESHOLDS["max_area_ratio"]:
        area_score = 0.0
    else:
        area_score = max(0.0, 1.0 - abs(area_ratio - 0.25) * 2.0)
        area_score = max(area_score, 0.3)

    # 7. NEW: Blue bottom band verification
    blue_band_score = 0.0
    if image_bgr is not None:
        blue_band_score = _verify_blue_bottom_band(image_bgr, pts)

    # 8. NEW: White side margin verification
    white_margin_score = 0.0
    if image_bgr is not None:
        white_margin_score = _verify_white_side_margins(image_bgr, pts)

    # Weighted combination (adjusted weights to incorporate structural features)
    score = (
        sym_score         * 0.12 +
        aspect_score      * 0.20 +
        angle_score       * 0.15 +
        solidity_score    * 0.08 +
        homog_score       * 0.15 +
        area_score        * 0.08 +
        blue_band_score   * 0.12 +  # NEW: Blue band validation
        white_margin_score * 0.10    # NEW: White margin validation
    )
    return float(np.clip(score, 0.0, 1.0))


# ===================================================================
#  Edge / binary image generation (enhanced with color segmentation)
# ===================================================================

def _generate_edge_maps(resized_bgr):
    """
    Produce a list of binary/edge images using every useful strategy.
    NOW INCLUDES color-based segmentation for card-specific features!
    """
    results = []
    h, w = resized_bgr.shape[:2]

    # Proportional kernel size
    k = max(3, int(round(min(w, h) / 150)))
    if k % 2 == 0:
        k += 1
    k_big = k * 2 + 1

    # --- Pre-process channels ------------------------------------------------
    bilateral = cv2.bilateralFilter(resized_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(bilateral, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2]
    s_ch = hsv[:, :, 1]

    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]

    # CLAHE on L and gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    l_clahe = clahe.apply(l_ch)

    morph_close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_big, k_big))
    morph_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    channels = [gray, gray_clahe, v_ch, l_ch, l_clahe]

    # --- NEW: Color-based segmentation strategies -----------------------------
    # 1. Blue band detection
    blue_mask = _detect_blue_band_region(bilateral)
    results.append(blue_mask)
    
    # Expand blue regions to find full card
    blue_dilated = cv2.dilate(blue_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k*3, k*3)), iterations=3)
    results.append(blue_dilated)
    
    # 2. White card body detection
    white_mask = _detect_white_card_region(bilateral)
    results.append(white_mask)
    
    # 3. Combined color mask (white body OR blue band)
    combined_color = cv2.bitwise_or(white_mask, blue_dilated)
    results.append(combined_color)
    
    # 4. Edge of color regions
    color_edges = cv2.Canny(combined_color, 50, 150)
    color_edges = cv2.dilate(color_edges, morph_small, iterations=2)
    results.append(color_edges)

    # --- Strategy 1: Auto Canny on each channel (tight + loose sigma) ---------
    for ch in channels:
        for sigma in (0.25, 0.33, 0.50):
            edg = _auto_canny(ch, sigma)
            edg = cv2.dilate(edg, morph_small, iterations=1)
            edg = cv2.morphologyEx(edg, cv2.MORPH_CLOSE, morph_close_k, iterations=1)
            results.append(edg)

    # --- Strategy 2: Fixed Canny thresholds -----------------------------------
    for ch in [gray, gray_clahe]:
        for lo, hi in [(30, 100), (50, 150), (75, 200)]:
            blurred = cv2.GaussianBlur(ch, (5, 5), 0)
            edg = cv2.Canny(blurred, lo, hi)
            edg = cv2.dilate(edg, morph_small, iterations=1)
            edg = cv2.morphologyEx(edg, cv2.MORPH_CLOSE, morph_close_k, iterations=1)
            results.append(edg)

    # --- Strategy 3: Sobel magnitude ------------------------------------------
    for ch in [gray_clahe, l_clahe]:
        sob_x = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3)
        sob_y = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sob_x ** 2 + sob_y ** 2)
        mag = np.uint8(np.clip(mag / mag.max() * 255, 0, 255)) if mag.max() > 0 else np.zeros_like(ch)
        _, bw = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, morph_close_k, iterations=1)
        results.append(bw)

    # --- Strategy 4: Scharr magnitude -----------------------------------------
    for ch in [gray_clahe, v_ch]:
        sc_x = cv2.Scharr(ch, cv2.CV_64F, 1, 0)
        sc_y = cv2.Scharr(ch, cv2.CV_64F, 0, 1)
        mag = np.sqrt(sc_x ** 2 + sc_y ** 2)
        mag = np.uint8(np.clip(mag / (mag.max() + 1e-9) * 255, 0, 255))
        _, bw = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, morph_close_k, iterations=1)
        results.append(bw)

    # --- Strategy 5: Laplacian ------------------------------------------------
    for ch in [gray_clahe]:
        lap = cv2.Laplacian(ch, cv2.CV_64F, ksize=3)
        lap = np.uint8(np.clip(np.abs(lap) / (np.abs(lap).max() + 1e-9) * 255, 0, 255))
        _, bw = cv2.threshold(lap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, morph_close_k, iterations=1)
        results.append(bw)

    # --- Strategy 6: Adaptive threshold (MEAN + GAUSSIAN) ---------------------
    for ch in [gray, gray_clahe, v_ch, l_clahe]:
        blurred = cv2.GaussianBlur(ch, (k | 1, k | 1), 0)
        for method in [cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C]:
            for block in [11, 21, 31]:
                at = cv2.adaptiveThreshold(blurred, 255, method,
                                           cv2.THRESH_BINARY_INV, block, 2)
                at = cv2.morphologyEx(at, cv2.MORPH_CLOSE, morph_close_k, iterations=2)
                results.append(at)

    # --- Strategy 7: Otsu on multiple channels --------------------------------
    for ch in [gray, v_ch, l_ch, l_clahe]:
        blurred = cv2.GaussianBlur(ch, (5, 5), 0)
        _, ot = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ot = cv2.morphologyEx(ot, cv2.MORPH_CLOSE, morph_close_k, iterations=2)
        results.append(ot)
        _, ot2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ot2 = cv2.morphologyEx(ot2, cv2.MORPH_CLOSE, morph_close_k, iterations=2)
        results.append(ot2)

    # --- Strategy 8: Low-saturation mask (white card) -------------------------
    for thr in [40, 60, 80]:
        _, sat_mask = cv2.threshold(s_ch, thr, 255, cv2.THRESH_BINARY_INV)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, morph_close_k, iterations=2)
        results.append(sat_mask)

    # --- Strategy 9: Color-distance from dominant border ----------------------
    border_pixels = np.concatenate([
        resized_bgr[0, :], resized_bgr[-1, :],
        resized_bgr[:, 0], resized_bgr[:, -1]
    ])
    mean_border = np.mean(border_pixels, axis=0).astype(np.float32)
    diff = np.linalg.norm(resized_bgr.astype(np.float32) - mean_border, axis=2)
    diff_norm = np.uint8(np.clip(diff / (diff.max() + 1e-9) * 255, 0, 255))
    _, cd_mask = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cd_mask = cv2.morphologyEx(cd_mask, cv2.MORPH_CLOSE, morph_close_k, iterations=2)
    results.append(cd_mask)

    # --- Strategy 10: Combined best edges + adaptive --------------------------
    best_canny = _auto_canny(gray_clahe, 0.33)
    best_canny = cv2.dilate(best_canny, morph_small, iterations=1)
    best_adapt = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray_clahe, (k | 1, k | 1), 0),
        255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    combined = cv2.bitwise_or(best_canny, best_adapt)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (k * 3, k * 3)),
                                iterations=2)
    results.append(combined)

    return results


def _auto_canny(gray, sigma=0.33):
    v = np.median(gray)
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lo, hi)


# ===================================================================
#  Hough-line fallback
# ===================================================================

def _line_intersection(line1, line2):
    """Compute intersection of two lines given as (rho, theta)."""
    rho1, theta1 = line1
    rho2, theta2 = line2
    ct1, st1 = math.cos(theta1), math.sin(theta1)
    ct2, st2 = math.cos(theta2), math.sin(theta2)
    det = ct1 * st2 - ct2 * st1
    if abs(det) < 1e-8:
        return None
    x = (st2 * rho1 - st1 * rho2) / det
    y = (ct1 * rho2 - ct2 * rho1) / det
    return (x, y)


def _hough_quad_candidates(edge_img, img_w, img_h, max_candidates=5):
    """
    Use Hough to find 4 dominant lines and form quad candidates.
    """
    candidates = []

    lines = cv2.HoughLines(edge_img, 1, np.pi / 180, threshold=min(img_w, img_h) // 4)
    if lines is None or len(lines) < 4:
        return candidates

    lines = lines[:, 0, :]

    horiz = []
    vert = []
    for rho, theta in lines:
        deg = math.degrees(theta) % 180
        if deg < 30 or deg > 150:
            vert.append((rho, theta))
        elif 60 < deg < 120:
            horiz.append((rho, theta))

    def _dedup(group, min_rho_gap=20):
        if not group:
            return group
        group = sorted(group, key=lambda x: x[0])
        deduped = [group[0]]
        for r, t in group[1:]:
            if abs(r - deduped[-1][0]) > min_rho_gap:
                deduped.append((r, t))
        return deduped

    horiz = _dedup(horiz, min_rho_gap=img_h * 0.05)
    vert = _dedup(vert, min_rho_gap=img_w * 0.05)

    if len(horiz) < 2 or len(vert) < 2:
        return candidates

    for (h1, h2) in combinations(horiz[:6], 2):
        for (v1, v2) in combinations(vert[:6], 2):
            corners = []
            for hl in (h1, h2):
                for vl in (v1, v2):
                    pt = _line_intersection(hl, vl)
                    if pt is None:
                        break
                    corners.append(pt)
            if len(corners) == 4:
                pts = _order_points(np.array(corners, dtype=np.float32))
                if _validate_quad_geometry(pts, img_w, img_h):
                    candidates.append(pts)
                    if len(candidates) >= max_candidates:
                        return candidates
    return candidates


# ===================================================================
#  Contour-based quad extraction
# ===================================================================

def _extract_quads_from_binary(binary, img_w, img_h):
    """
    Find contours, approximate to quads, validate geometry.
    """
    quads = []
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    img_area = float(img_w * img_h)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * THRESHOLDS["min_area_ratio"]:
            continue
        if area > img_area * THRESHOLDS["max_area_ratio"]:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri < 1:
            continue

        found_quad = None
        for eps_mult in [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.08]:
            approx = cv2.approxPolyDP(cnt, eps_mult * peri, True)
            n = len(approx)
            if n == 4:
                candidate = approx.reshape(4, 2).astype(np.float32)
                ordered = _order_points(candidate)
                if _validate_quad_geometry(ordered, img_w, img_h):
                    found_quad = ordered
                    break

        if found_quad is None:
            hull = cv2.convexHull(cnt)
            hull_approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
            if len(hull_approx) == 4:
                candidate = hull_approx.reshape(4, 2).astype(np.float32)
                ordered = _order_points(candidate)
                if _validate_quad_geometry(ordered, img_w, img_h):
                    found_quad = ordered

        if found_quad is None:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.float32)
            ordered = _order_points(box)
            if _validate_quad_geometry(ordered, img_w, img_h):
                found_quad = ordered

        if found_quad is not None:
            quads.append((found_quad, area))

    return quads


# ===================================================================
#  Main detection
# ===================================================================

def detect(image_bgr):
    """
    Detect ID card quadrilateral in image.

    Returns dict:
        card_detected       : bool
        quadrilateral       : list of 4 [x,y] points (int) or None
        aspect_ratio_deviation : float or None
        card_area_ratio     : float or None
        confidence          : float or None
    """
    if image_bgr is None:
        return _empty_result()

    orig_h, orig_w = image_bgr.shape[:2]
    if orig_h < 50 or orig_w < 50:
        return _empty_result()

    best_score = -1.0
    best_quad = None

    # Process at multiple scales
    for target_h in _PROCESSING_HEIGHTS:
        if orig_h <= target_h:
            scale = 1.0
            resized = image_bgr.copy()
        else:
            scale = orig_h / float(target_h)
            new_w = int(orig_w / scale)
            new_h = int(orig_h / scale)
            if new_w < 50 or new_h < 50:
                continue
            resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        rh, rw = resized.shape[:2]

        # Generate all binary/edge maps (now includes color-based ones)
        binaries = _generate_edge_maps(resized)

        # Extract quad candidates
        all_quads = []
        for bw in binaries:
            quads = _extract_quads_from_binary(bw, rw, rh)
            all_quads.extend(quads)

        # Hough-line fallback
        gray_r = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        edge_for_hough = cv2.Canny(clahe.apply(gray_r), 50, 150)
        edge_for_hough = cv2.dilate(edge_for_hough,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                     iterations=1)
        hough_quads = _hough_quad_candidates(edge_for_hough, rw, rh)
        for hq in hough_quads:
            area = cv2.contourArea(hq)
            all_quads.append((hq, area))

        # Score every candidate (NOW with color validation!)
        for (quad, area) in all_quads:
            sc = _score_candidate(quad, rw, rh, resized)

            # Bonus for high solidity
            hull = cv2.convexHull(quad.astype(np.float32))
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity > 0.92:
                sc = min(1.0, sc + 0.03)

            # Penalty for low solidity/extent
            x, y, wb, hb = cv2.boundingRect(quad.astype(np.int32))
            extent = area / (wb * hb) if (wb * hb) > 0 else 0
            if solidity < THRESHOLDS["min_solidity"] or extent < THRESHOLDS["min_extent"]:
                sc *= 0.75

            if sc > best_score or (abs(sc - best_score) < 0.01 and area > 0):
                if sc > best_score:
                    best_score = sc
                    best_quad = (quad * scale).astype(np.float32)

    # Final decision
    if best_quad is not None and best_score >= THRESHOLDS["score_accept"]:
        quad_int = best_quad.astype(int)
        quad_int[:, 0] = np.clip(quad_int[:, 0], 0, orig_w - 1)
        quad_int[:, 1] = np.clip(quad_int[:, 1], 0, orig_h - 1)

        tl, tr, br, bl = quad_int.astype(np.float32)
        width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
        if min(width, height) > 0:
            aspect = float(max(width, height) / min(width, height))
        else:
            aspect = 0.0
        deviation = abs(aspect - THRESHOLDS["expected_aspect_ratio"])

        card_area = cv2.contourArea(quad_int)
        card_area_ratio = round(float(card_area) / float(orig_w * orig_h), 4)

        return {
            "card_detected": True,
            "quadrilateral": quad_int.tolist(),
            "aspect_ratio_deviation": round(float(deviation), 4),
            "card_area_ratio": card_area_ratio,
            "confidence": round(best_score, 4),
        }

    return _empty_result()


def _empty_result():
    return {
        "card_detected": False,
        "quadrilateral": None,
        "aspect_ratio_deviation": None,
        "card_area_ratio": None,
        "confidence": None,
    }


# ===================================================================
#  Visualization
# ===================================================================

def draw_bounding_box(image_bgr, quad, score=None):
    """Draw the detected quadrilateral on the image for visualization."""
    if quad is None:
        return image_bgr.copy()
    annotated = image_bgr.copy()
    pts = np.array(quad, np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated, [pts], True, (0, 255, 0), 3, cv2.LINE_AA)

    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for (x, y), c, lbl in zip(quad, colors, labels):
        cv2.circle(annotated, (int(x), int(y)), 8, c, -1)
        cv2.putText(annotated, lbl, (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

    if score is not None:
        cv2.putText(annotated, f"Score: {score:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return annotated


# ===================================================================
#  Perspective correction utility
# ===================================================================

def extract_card(image_bgr, quad, output_width=856, output_height=540):
    """
    Warp the detected card region to a top-down rectangular view.
    """
    if quad is None:
        return None
    pts = _order_points(np.array(quad, dtype=np.float32))
    dst = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image_bgr, M, (output_width, output_height),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    return warped


# ===================================================================
#  CLI / standalone test
# ===================================================================
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_card.jpg"
    img = cv2.imread(path)
    if img is None:
        print(f"Cannot read image: {path}")
        sys.exit(1)

    result = detect(img)
    print(result)

    if result["card_detected"]:
        out = draw_bounding_box(img, result["quadrilateral"], result.get("confidence"))
        cv2.imwrite("detected_output.jpg", out)
        print("Saved: detected_output.jpg")

        warped = extract_card(img, result["quadrilateral"])
        if warped is not None:
            cv2.imwrite("card_extracted.jpg", warped)
            print("Saved: card_extracted.jpg")
    else:
        print("No card detected.")