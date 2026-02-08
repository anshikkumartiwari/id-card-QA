import cv2
import numpy as np

# Standard ID-1 card ratio (85.60mm / 53.98mm) approx 1.586
THRESHOLDS = {
    "expected_aspect_ratio": 1.586,
    "min_area_ratio": 0.02,   # Card must be at least 2% of screen (allow smaller crops)
    "max_area_ratio": 0.98,   # Card must be less than 98% of screen
    "score_accept": 0.45      # final score threshold to accept detection
}

def _order_points(pts):
    """
    Order points as Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    """
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect

def _auto_canny(img, sigma=0.33):
    """
    Auto Canny thresholds based on median of image.
    """
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)

def _score_candidate(pts, img_w, img_h):
    """
    Score a quadrilateral candidate by how card-like it is (0..1).
    Uses rectangularity, aspect similarity, solidity and area proximity.
    """
    pts = np.array(pts, dtype=np.float32)
    tl, tr, br, bl = pts
    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    if max(width_top, width_bot) == 0 or max(height_left, height_right) == 0:
        return 0.0

    # symmetry of opposite sides
    w_sym = min(width_top, width_bot) / max(width_top, width_bot)
    h_sym = min(height_left, height_right) / max(height_left, height_right)
    rect_sym = (w_sym + h_sym) / 2.0

    # aspect ratio (longer_side / shorter_side)
    avg_w = (width_top + width_bot) / 2.0
    avg_h = (height_left + height_right) / 2.0
    if min(avg_w, avg_h) == 0:
        return 0.0
    aspect = max(avg_w, avg_h) / min(avg_w, avg_h)
    target = THRESHOLDS["expected_aspect_ratio"]
    deviation = abs(aspect - target) / target  # normalized deviation
    aspect_score = max(0.0, 1.0 - (deviation * 2.5))  # tolerant to perspective

    # area ratio (compared to image)
    quad_area = cv2.contourArea(pts)
    image_area = float(img_w * img_h)
    area_ratio = quad_area / image_area
    # prefer area in reasonable range, penalize extremes
    if area_ratio <= 0:
        return 0.0
    area_pref = 1.0 - abs(np.log(area_ratio / ((THRESHOLDS["min_area_ratio"] + THRESHOLDS["max_area_ratio"]) / 2.0) + 1e-9))
    area_score = np.clip(area_pref, 0.0, 1.0)

    # solidity (area / hull area) to avoid highly concave shapes
    hull = cv2.convexHull(pts.astype(np.float32))
    hull_area = cv2.contourArea(hull) if hull is not None else 0
    solidity = quad_area / hull_area if hull_area > 0 else 0.0
    solidity_score = np.clip(solidity, 0.0, 1.0)

    # combined score (weights tuned for robustness)
    score = (rect_sym * 0.3) + (aspect_score * 0.35) + (solidity_score * 0.2) + (area_score * 0.15)
    return float(np.clip(score, 0.0, 1.0))

def _get_processing_candidates(image_bgr):
    """
    Create multiple binary candidates using color channels, edge maps, and thresholding.
    """
    candidates = []
    h, w = image_bgr.shape[:2]

    # small bilateral to preserve edges while reducing texture noise
    bilateral = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bilateral, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    # kernel sizes proportional to image
    k = max(3, int(round(min(w, h) / 200)))  # small kernel base
    if k % 2 == 0: k += 1

    # 1) Auto Canny on grayscale (edges)
    edges = _auto_canny(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=1)
    candidates.append(edges)

    # 2) Adaptive threshold on gray (inverted so card on light bg becomes foreground)
    g_blur = cv2.GaussianBlur(gray, (k|1, k|1), 0)
    adapt = cv2.adaptiveThreshold(g_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k*2+1, k*2+1)), iterations=2)
    candidates.append(adapt)

    # 3) Otsu on V channel (works when card color differs in value)
    _, otsu = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k*2+1, k*2+1)), iterations=2)
    candidates.append(otsu)

    # 4) Low-saturation mask (useful for white card on textured colored background)
    _, sat_mask = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidates.append(sat_mask)

    # 5) Combine edges + adapt to fill holes
    combined = cv2.bitwise_or(edges, adapt)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k*3, k*3)), iterations=1)
    candidates.append(combined)

    return candidates

def detect(image_bgr):
    """
    Detect ID card quadrilateral in image.
    Returns a dict: card_detected(bool), quadrilateral(list of 4 pts int), aspect_ratio_deviation, card_area_ratio
    """
    if image_bgr is None:
        return {"card_detected": False, "quadrilateral": None, "aspect_ratio_deviation": None, "card_area_ratio": None}

    orig_h, orig_w = image_bgr.shape[:2]
    target_h = 700
    scale = orig_h / target_h if orig_h > target_h else 1.0
    resized = cv2.resize(image_bgr, (int(orig_w / scale), int(orig_h / scale))) if scale != 1.0 else image_bgr.copy()
    rh, rw = resized.shape[:2]
    area_resized = float(rw * rh)

    binary_images = _get_processing_candidates(resized)

    best_score = -1.0
    best_quad = None
    best_area = 0.0

    for binary in binary_images:
        # find contours and keep only external ones
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= 0:
                continue
            # area filter relative to resized
            if area < (area_resized * THRESHOLDS["min_area_ratio"]): 
                continue
            if area > (area_resized * THRESHOLDS["max_area_ratio"]):
                continue

            # approximate to reduce noise
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)

            candidate_quad = None
            if len(approx) == 4:
                candidate_quad = approx.reshape(4, 2)
            elif len(approx) > 4:
                # fit min area rect to get robust 4-corner box (handles rounded corners)
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                candidate_quad = np.array(box, dtype=np.float32)
            else:
                # small polygons (triangles etc.) ignore
                continue

            ordered = _order_points(candidate_quad)
            score = _score_candidate(ordered, rw, rh)

            # small extra sanity: check solidity and extent to cull noise
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull) if hull is not None else 0
            solidity = area / hull_area if hull_area > 0 else 0
            x, y, wb, hb = cv2.boundingRect(cnt)
            extent = area / (wb * hb) if (wb * hb) > 0 else 0

            # penalize if extremely concave or poor extent
            if solidity < 0.7 or extent < 0.45:
                score *= 0.8

            # prefer larger area if scores are equal-ish
            if (score > best_score) or (abs(score - best_score) < 1e-6 and area > best_area):
                best_score = float(score)
                best_quad = ordered
                best_area = float(area)

    if best_quad is not None and best_score >= THRESHOLDS["score_accept"]:
        # scale quad back to original image
        quad_scaled = (best_quad * scale).astype(int)
        tl, tr, br, bl = quad_scaled.astype(np.float32)
        width = np.linalg.norm(tr - tl)
        height = np.linalg.norm(bl - tl)
        aspect = float(max(width, height) / min(width, height)) if min(width, height) > 0 else 0.0
        deviation = abs(aspect - THRESHOLDS["expected_aspect_ratio"])
        quad_list = quad_scaled.tolist()

        # compute card area ratio on original resolution using contour area of scaled quad
        card_area = cv2.contourArea(np.array(quad_scaled, dtype=np.int32))
        card_area_ratio = round(float(card_area) / float(orig_w * orig_h), 4)

        return {
            "card_detected": True,
            "quadrilateral": quad_list,
            "aspect_ratio_deviation": round(float(deviation), 4),
            "card_area_ratio": card_area_ratio
        }

    return {
        "card_detected": False,
        "quadrilateral": None,
        "aspect_ratio_deviation": None,
        "card_area_ratio": None
    }

def draw_bounding_box(image_bgr, quad):
    """
    Draw the detected quadrilateral on the image for visualization.
    """
    if quad is None:
        return image_bgr.copy()
    annotated = image_bgr.copy()
    pts = np.array(quad, np.int32).reshape((-1,1,2))
    cv2.polylines(annotated, [pts], True, (0,255,0), 3, cv2.LINE_AA)
    # draw corner markers (TL, TR, BR, BL)
    colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255)]
    for (x, y), c in zip(quad, colors):
        cv2.circle(annotated, (int(x), int(y)), 8, c, -1)
    return annotated

# Example usage (reference):
# img = cv2.imread("/mnt/data/plain_white_background_no_defocus_no_glare_all_clear.jpg")
# res = detect(img)
# print(res)
# if res["card_detected"]:
#     out = draw_bounding_box(img, res["quadrilateral"])
#     cv2.imwrite("detected.jpg", out)
