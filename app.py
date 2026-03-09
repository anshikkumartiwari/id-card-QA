import os
import uuid

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

from config import TEMP_FOLDER
from pipeline import run_pipeline

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
    Receive an uploaded image, run the full QA pipeline,
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

    # --- Run the full QA pipeline ---
    results = run_pipeline(image_bgr)

    # --- Save annotated image to temp/ ---
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(TEMP_FOLDER, filename)
    cv2.imwrite(filepath, results["annotated_image"])

    # --- Build response (everything except the numpy array) ---
    response = {
        "resolution": results["resolution"],
        "card_detection": results["card_detection"],
        "face": results["face"],
        "blur": results["blur"],
        "glare": results["glare"],
        "noise": results["noise"],
        "exposure": results["exposure"],
        "processed_image_url": f"/temp/{filename}",
    }

    return jsonify(response)


@app.route("/temp/<filename>")
def serve_temp(filename):
    """Serve processed images from the temp folder."""
    return send_from_directory(TEMP_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)