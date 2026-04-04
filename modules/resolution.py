import numpy as np

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
            "short_edge": 0
        }

    h, w = image_bgr.shape[:2]
    long_edge = max(w, h)
    short_edge = min(w, h)

    return {
        "image_width": int(w),
        "image_height": int(h),
        "long_edge": int(long_edge),
        "short_edge": int(short_edge)
    }


# Example usage (reference):
# import cv2
# img = cv2.imread("/mnt/data/sample_id.jpg")
# res = analyze(img)
# print(res)
#     print("Image resolution stats:", res)