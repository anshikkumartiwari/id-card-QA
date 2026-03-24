import sqlite3
import random
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.db import init_db, log_assessment, log_feedback

FILENAMES = [
    "WhatsApp Image 2026-03-01 at 09.14.22 AM",
    "WhatsApp Image 2026-03-02 at 07.58.10 PM",
    "WhatsApp Image 2026-03-03 at 10.32.47 AM",
    "WhatsApp Image 2026-03-04 at 06.11.05 PM",
    "WhatsApp Image 2026-03-05 at 01.49.33 AM",
    "WhatsApp Image 2026-03-06 at 08.23.59 PM",
    "WhatsApp Image 2026-03-07 at 03.07.18 PM",
    "WhatsApp Image 2026-03-08 at 11.55.02 AM",
    "WhatsApp Image 2026-03-09 at 05.44.27 PM",
    "WhatsApp Image 2026-03-10 at 09.02.36 AM",
    "WhatsApp Image 2026-03-11 at 12.38.41 AM",
    "WhatsApp Image 2026-03-12 at 04.16.09 PM",
    "WhatsApp Image 2026-03-13 at 07.27.58 AM",
    "WhatsApp Image 2026-03-14 at 02.51.13 PM",
    "WhatsApp Image 2026-03-15 at 10.05.49 PM",
    "WhatsApp Image 2026-03-16 at 06.33.21 AM",
    "WhatsApp Image 2026-03-17 at 08.47.55 PM",
    "WhatsApp Image 2026-03-18 at 11.19.30 AM",
    "WhatsApp Image 2026-03-19 at 03.26.44 PM",
    "WhatsApp Image 2026-03-20 at 09.58.07 PM"
]

# Target Accuracies: (Probability of being WRONG = 1 - Accuracy)
# resolution: 99% -> 0.01
# card detection: 88% -> 0.12
# face detection: 98% -> 0.02
# blur focus: 98% -> 0.02
# glare: 96% -> 0.04
# noise: 94% -> 0.06
# exposure: 95% -> 0.05
# geometry: 89% -> 0.11
ERROR_PROBS = {
    "resolution": 0.01,
    "card": 0.12,
    "face": 0.02,
    "blur": 0.02,
    "glare": 0.04,
    "noise": 0.06,
    "exposure": 0.05,
    "geometry": 0.11
}

def generate_mock_data():
    # Remove old db if exists
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'qa_assessments.db')
    if os.path.exists(db_path):
        os.remove(db_path)
        
    init_db()
    
    for filename in FILENAMES:
        # Simulate base pipeline passes (assume pipeline is mostly right, so we just randomly set passes)
        # Real accuracy is measured by feedback!
        pass_res = random.random() > 0.05
        pass_crd = random.random() > 0.05
        pass_fce = random.random() > 0.05
        pass_blr = random.random() > 0.10
        pass_glr = random.random() > 0.10
        pass_nse = random.random() > 0.10
        pass_exp = random.random() > 0.05
        pass_geo = random.random() > 0.10
        
        is_accepted = all([pass_res, pass_crd, pass_fce, pass_blr, pass_glr, pass_nse, pass_exp, pass_geo])
        decision = "ACCEPT" if is_accepted else "REJECT"
        quality_score = random.uniform(70.0, 100.0) if is_accepted else random.uniform(30.0, 69.0)
        
        data = {
            "filename": filename,
            "decision": decision,
            "quality_score": round(quality_score, 2),
            "pass_resolution": pass_res,
            "pass_card": pass_crd,
            "pass_face": pass_fce,
            "pass_blur": pass_blr,
            "pass_glare": pass_glr,
            "pass_noise": pass_nse,
            "pass_exposure": pass_exp,
            "pass_geometry": pass_geo
        }
        
        assessment_id = log_assessment(data)
        
        # Now apply the feedback according to ERROR_PROBS
        for attr, err_prob in ERROR_PROBS.items():
            if random.random() < err_prob:
                log_feedback(assessment_id, attr, is_wrong=True)
            
    print(f"Successfully generated {len(FILENAMES)} sample records targeting requested accuracies.")

if __name__ == "__main__":
    generate_mock_data()
