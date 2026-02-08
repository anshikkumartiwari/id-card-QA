import numpy as np
from config import THRESHOLDS


def analyze(image_bgr: np.ndarray) -> dict:
    """
    Check if image resolution meets minimum requirements for ID card processing.
    This is the cheapest possible check — run first, fail fast.
    """
    if image_bgr is None or image_bgr.size == 0:
        return {
            "image_width": 0,
            "image_height": 0,
            "long_edge": 0,
            "short_edge": 0,
            "resolution_adequate": False
        }

    h, w = image_bgr.shape[:2]
    long_edge = max(w, h)
    short_edge = min(w, h)

    adequate = (
        long_edge >= THRESHOLDS["min_long_edge"]
        and short_edge >= THRESHOLDS["min_short_edge"]
    )

    return {
        "image_width": int(w),
        "image_height": int(h),
        "long_edge": int(long_edge),
        "short_edge": int(short_edge),
        "resolution_adequate": bool(adequate)
    }