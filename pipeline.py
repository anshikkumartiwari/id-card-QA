"""
pipeline.py
-----------
Orchestrates all QA submodule calls in sequence.
app.py passes the decoded image here; this module calls each check,
collects results into a flags dict, and returns everything to app.py.
"""

import cv2
import numpy as np

from config import THRESHOLDS
from modules import resolution, card_detection, face, blur, glare, noise, exposure, geometry


def _extract_card_region(image_bgr: np.ndarray, quad: list) -> np.ndarray:
    """
    Perspective-warp the detected card quadrilateral into a flat rectangle
    and return it as a grayscale image (used for face detection).
    """
    pts = np.array(quad, dtype="float32")
    tl, tr, br, bl = pts

    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    max_w = int(max(width_top, width_bot))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_h = int(max(height_left, height_right))

    if max_w == 0 or max_h == 0:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image_bgr, M, (max_w, max_h))
    return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)


def run_pipeline(image_bgr: np.ndarray) -> dict:
    """
    Run the complete QA pipeline on an uploaded image.

    Order of checks:
        1. Resolution  (cheapest — fail fast)
        2. Card detection
        3. Face detection  (on rectified card)
        4. Blur / focus analysis  (full image + card quad + face bbox)

    Returns
    -------
    dict with keys:
        resolution       – resolution module result dict
        card_detection   – card detection module result dict
        face             – face detection module result dict
        blur             – blur module result dict
        annotated_image  – BGR image with overlays (numpy array)
    """

    # ── Step 1: Resolution check ──────────────────────────────────────
    res_result = resolution.analyze(image_bgr)
    res_result["resolution_adequate"] = bool(
        res_result["long_edge"] >= THRESHOLDS["min_long_edge"]
        and res_result["short_edge"] >= THRESHOLDS["min_short_edge"]
    )

    # ── Step 2: Card detection ────────────────────────────────────────
    card_result = card_detection.detect(image_bgr)
    card_score = card_result.get("confidence")
    if card_score is None:
        card_score = -1.0
    card_result["card_detected"] = bool(card_score >= THRESHOLDS["min_detection_score"])

    # ── Step 3: Annotated image ───────────────────────────────────────
    if card_result["card_detected"]:
        annotated = card_detection.draw_bounding_box(
            image_bgr, card_result["quadrilateral"],
            score=card_result.get("confidence")
        )
    else:
        annotated = image_bgr.copy()

    # ── Step 4: Extract card region (for face detection) ──────────────
    card_quad = card_result["quadrilateral"]  # list of 4 [x,y] or None
    if card_result["card_detected"] and card_quad is not None:
        gray_card = _extract_card_region(image_bgr, card_quad)
    else:
        gray_card = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # ── Step 5: Face detection ────────────────────────────────────────
    face_result = face.detect_face(gray_card)
    face_result["detected"] = bool(face_result["bbox"] is not None)

    # Draw face bbox on the annotated image if detected
    if face_result["detected"] and face_result["bbox"] is not None:
        fx1, fy1, fx2, fy2 = face_result["bbox"]

        if card_result["card_detected"] and card_quad is not None:
            # Map face bbox from rectified card space back to original image
            quad = np.array(card_quad, dtype="float32")
            tl, tr, br, bl = quad
            cw = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
            ch = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
            if cw > 0 and ch > 0:
                src = np.array([[0, 0], [cw-1, 0], [cw-1, ch-1], [0, ch-1]], dtype="float32")
                M_inv = cv2.getPerspectiveTransform(src, quad)
                corners = np.array([[[fx1, fy1]], [[fx2, fy2]]], dtype="float32")
                mapped = cv2.perspectiveTransform(corners, M_inv)
                p1 = tuple(mapped[0][0].astype(int))
                p2 = tuple(mapped[1][0].astype(int))
                cv2.rectangle(annotated, p1, p2, (255, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (255, 0, 255), 2, cv2.LINE_AA)

    # ── Step 6: Blur / focus analysis ─────────────────────────────────
    gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_bbox = face_result["bbox"] if face_result["detected"] else None
    blur_result = blur.analyze(gray_full, card_quad=card_quad, face_bbox=face_bbox)

    # ── Step 7: Glare analysis ────────────────────────────────────────
    glare_result = glare.analyze(image_bgr, card_quad=card_quad, face_bbox=face_bbox)
    glare_result["glare_detected"] = bool(glare_result["glare_percentage"] > THRESHOLDS["glare_max_percentage"])
    glare_result["glare_on_face"] = bool(glare_result["glare_percentage_face"] > THRESHOLDS["glare_max_percentage"])

    # ── Step 8: Noise analysis ────────────────────────────────────────
    noise_result = noise.analyze(image_bgr, card_quad=card_quad, face_bbox=face_bbox)
    noise_result["noise_detected"] = bool(
        noise_result["noise_level_card"] > THRESHOLDS["noise_max_level"] or
        noise_result["noise_level_face"] > THRESHOLDS["noise_max_level"]
    )

    # ── Step 9: Exposure analysis ─────────────────────────────────────
    exposure_result = exposure.analyze(image_bgr, card_quad=card_quad, face_bbox=face_bbox)
    mbc = exposure_result.get("mean_brightness_card", 0.0)
    if mbc < THRESHOLDS["exposure_min_brightness"]:
        exposure_result["status"] = "underexposed"
        exposure_result["exposure_adequate"] = False
    elif mbc > THRESHOLDS["exposure_max_brightness"]:
        exposure_result["status"] = "overexposed"
        exposure_result["exposure_adequate"] = False
    else:
        exposure_result["status"] = "good"
        exposure_result["exposure_adequate"] = True

    # ── Step 10: Geometry analysis ────────────────────────────────────
    geometry_result = geometry.analyze(card_quad=card_quad)
    
    aspect_ratio = geometry_result.get("aspect_ratio", 0.0)
    expected_ratio = THRESHOLDS["geometry_aspect_ratio_expected"]
    tol = THRESHOLDS["geometry_aspect_ratio_tolerance"]
    is_ratio_valid = bool(abs(aspect_ratio - expected_ratio) <= tol)
    
    rot = geometry_result.get("rotation_degrees", 0.0)
    rotation_detected = bool(rot > THRESHOLDS["geometry_max_rotation_degrees"])
    
    wd_ratio = geometry_result.get("width_diff_ratio", 0.0)
    hd_ratio = geometry_result.get("height_diff_ratio", 0.0)
    max_persp = THRESHOLDS["geometry_max_perspective_distortion"]
    perspective_distortion = bool(wd_ratio > max_persp or hd_ratio > max_persp)
    
    geometry_result["is_ratio_valid"] = is_ratio_valid
    geometry_result["rotation_detected"] = rotation_detected
    
    msg = "Good geometry."
    if not is_ratio_valid:
        msg = f"Aspect ratio ({aspect_ratio:.2f}) deviates from expected {expected_ratio:.2f}."
    elif rotation_detected:
        msg = f"Card is rotated by ~{rot}°."
    elif perspective_distortion:
        msg = "Significant perspective 3D distortion detected."
        
    geometry_result["message"] = msg
    geometry_result["geometry_adequate"] = bool(is_ratio_valid and not rotation_detected and not perspective_distortion)

    # ── Compile flags ─────────────────────────────────────────────────
    return {
        "resolution": res_result,
        "card_detection": card_result,
        "face": face_result,
        "blur": blur_result,
        "glare": glare_result,
        "noise": noise_result,
        "exposure": exposure_result,
        "geometry": geometry_result,
        "annotated_image": annotated,
    }