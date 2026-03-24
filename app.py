import os
import uuid

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

from config import TEMP_FOLDER
from pipeline import run_pipeline
from utils.db import init_db, log_assessment, log_feedback

app = Flask(__name__)

# Ensure temp folder exists on startup
os.makedirs(TEMP_FOLDER, exist_ok=True)
init_db()


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

    # --- Compute Quality Score and Decision ---
    res = results.get("resolution", {})
    card = results.get("card_detection", {})
    fce = results.get("face", {})
    blur = results.get("blur", {})
    glr = results.get("glare", {})
    ns = results.get("noise", {})
    exp = results.get("exposure", {})
    geom = results.get("geometry", {})

    pass_resolution = res.get("resolution_adequate", False)
    pass_card = card.get("card_detected", False)
    pass_face = fce.get("detected", False)
    pass_blur = blur.get("blur_composite", 0) > 80
    pass_glare = not glr.get("glare_detected", True) if glr else True
    pass_noise = not ns.get("noise_detected", True) if ns else True
    pass_exposure = exp.get("exposure_adequate", False) if exp else False
    pass_geometry = geom.get("geometry_adequate", False) if geom else False

    is_accepted = pass_resolution and pass_card and pass_face and pass_blur and pass_glare and pass_noise and pass_exposure and pass_geometry
    decision = "ACCEPT" if is_accepted else "REJECT"
    
    blur_score = blur.get("blur_composite") or 0.0
    glare_score = glr.get("glare_percentage") or 0.0
    quality_score = max(0.0, min(100.0, blur_score - (glare_score * 0.5)))

    # --- Log Assesment ---
    assessment_data = {
        "filename": file.filename,
        "decision": decision,
        "quality_score": round(quality_score, 2),
        "pass_resolution": pass_resolution,
        "pass_card": pass_card,
        "pass_face": pass_face,
        "pass_blur": pass_blur,
        "pass_glare": pass_glare,
        "pass_noise": pass_noise,
        "pass_exposure": pass_exposure,
        "pass_geometry": pass_geometry
    }
    assessment_id = log_assessment(assessment_data)

    # --- Build response (everything except the numpy array) ---
    response = {
        "assessment_id": assessment_id,
        "decision": decision,
        "quality_score": round(quality_score, 2),
        "resolution": results["resolution"],
        "card_detection": results["card_detection"],
        "face": results["face"],
        "blur": results["blur"],
        "glare": results["glare"],
        "noise": results["noise"],
        "exposure": results["exposure"],
        "geometry": results["geometry"],
        "processed_image_url": f"/temp/{filename}",
    }

    return jsonify(response)


@app.route("/feedback", methods=["POST"])
def feedback():
    """Receive user feedback if the model made a wrong prediction for specific attributes."""
    data = request.json
    if not data or "assessment_id" not in data:
        return jsonify({"error": "Missing assessment_id"}), 400
    
    assessment_id = data["assessment_id"]
    attributes = data.get("attributes", [])
    
    if not isinstance(attributes, list):
        # Fallback for single attribute support if needed
        attribute = data.get("attribute")
        is_wrong = data.get("is_wrong", True)
        if attribute:
            log_feedback(assessment_id, attribute, is_wrong)
    else:
        for attr in attributes:
            log_feedback(assessment_id, attr, True)
            
    return jsonify({"success": True})


@app.route("/temp/<filename>")
def serve_temp(filename):
    """Serve processed images from the temp folder."""
    return send_from_directory(TEMP_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)