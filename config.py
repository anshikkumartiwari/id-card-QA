# config.py
THRESHOLDS = {
    # Resolution gate (px)
    "min_long_edge": 900,
    "min_short_edge": 550,

    # Card physical dimensions (mm) -> used to compute aspect as long/short
    # The card measurements you supplied: length = 89.5 mm, width = 54 mm
    "card_length_mm": 89.5,
    "card_width_mm": 54.0,
    "expected_aspect_ratio": float(89.5 / 54.0),   # long / short ≈ 1.6574
    "aspect_ratio_tolerance": 0.55,

    # Structural priors (relative to card physical dims)
    # bottom colored strip thickness as fraction of card long-edge (height)
    "bottom_band_rel_height": 10.0 / 89.5,   # ≈ 0.1117 (1 cm / 8.95 cm) blue band at bottom
    # side white margin usual width as fraction of card short-edge (width)
    "side_margin_rel_width": 5.0 / 54.0,     # ≈ 0.0926 (0.5 cm / 5.4 cm)
    # when margins go deeper (2nd state), width up to:
    "side_margin_deep_rel_width": 10.0 / 54.0,  # ≈ 0.185 (1 cm / 5.4 cm)
    # vertical region (relative to card long-edge) where margins widen (2cm..6cm)
    "margin_widen_y_rel_range": (20.0 / 89.5, 60.0 / 89.5),  # ≈ (0.224, 0.670)

    # Color / texture thresholds (tunable)
    "bottom_band_min_blue_over_red": 1.15,
    "bottom_band_min_saturation": 35,       # saturation must be reasonably high
    "bottom_band_coverage_ratio": 0.70,     # how continuous the blue band should be horizontally

    "side_margin_min_brightness": 200,      # side margin should be bright (white)
    "side_margin_max_saturation": 50,       # side margin should be low saturation (near-white)
    "side_margin_max_gradient": 10.0,       # low gradient energy allowed in margins

    # Final detection gating
    "min_detection_score": 0.45,
}

TEMP_FOLDER = "temp"
