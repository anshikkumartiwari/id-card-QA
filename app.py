import os
import uuid

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

from config import TEMP_FOLDER
from modules import resolution, card_detection

app = Flask(__name__)

# Ensure temp folder exists on startup
os.makedirs(TEMP_FOLDER, exist_ok=True)


@app.route("/")
def index():
    """Serve the main UI page."""
    return render_template("index.html")


@app.route("/assess", methods=["POST"])
def assess():
    """
    Receive an uploaded image, run resolution + card detection,
    save annotated image to temp/, return JSON results.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Decode image from upload bytes
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return jsonify({"error": "Could not decode image"}), 400

    # --- Step 1: Resolution check ---
    res_result = resolution.analyze(image_bgr)

    # --- Step 2: Card detection ---
    card_result = card_detection.detect(image_bgr)

    # --- Draw bounding box if card detected ---
    if card_result["card_detected"]:
        annotated = card_detection.draw_bounding_box(
            image_bgr, card_result["quadrilateral"]
        )
    else:
        annotated = image_bgr.copy()

    # --- Save processed image to temp/ ---
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(TEMP_FOLDER, filename)
    cv2.imwrite(filepath, annotated)

    # --- Build response ---
    response = {
        "resolution": res_result,
        "card_detection": card_result,
        "processed_image_url": f"/temp/{filename}"
    }

    return jsonify(response)


@app.route("/temp/<filename>")
def serve_temp(filename):
    """Serve processed images from the temp folder."""
    return send_from_directory(TEMP_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)