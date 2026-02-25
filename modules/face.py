import cv2
import numpy as np
from config import THRESHOLDS


# --- Load face detector (Haar cascade) ---
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face(gray_card: np.ndarray) -> dict:
    """
    gray_card: grayscale rectified card
    Returns:
      - detected: bool
      - bbox: (x1, y1, x2, y2) or None
      - confidence: float (placeholder)
    """
    h, w = gray_card.shape

    # Detect faces
    faces = _face_cascade.detectMultiScale(
        gray_card,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(int(0.2 * w), int(0.2 * h)),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return {
            "detected": False,
            "bbox": None,
            "confidence": 0.0
        }

    # Pick the largest face
    x, y, rw, rh = max(faces, key=lambda f: f[2] * f[3])

    # Confidence placeholder (can be improved later)
    confidence = 0.95

    return {
        "detected": True,
        "bbox": (int(x), int(y), int(x + rw), int(y + rh)),
        "confidence": float(confidence)
    }


# Example usage (reference):
# import cv2
# img = cv2.imread("card.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# res = detect_face(gray)
# print(res)
