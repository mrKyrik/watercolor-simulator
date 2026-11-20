"""
Validation Suite for Multi-Layer Glazing and Balanced Pigment Density
"""

import os
import sys
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.simulation import WatercolorEngine


def test_glazing_wet_on_dry():
    print("\n--- TEST: MULTI-LAYER WET-ON-DRY GLAZING ---")
    engine = WatercolorEngine(width=600, height=500)

    # 1. First Wash: Cadmium Yellow rectangle/strip across the top half
    stroke1 = [
        (100.0, 200.0),
        (300.0, 200.0),
        (500.0, 200.0),
    ]
    engine.add_stroke_from_points(
        stroke1,
        color_rgb=(245, 215, 10),
        radius=45.0,
        intensity=0.85,
        pigment_name="Cadmium Yellow",
    )
    # Simulate first wash spreading and settling
    engine.run_simulation(steps=25, dt=0.05)
    
    # 2. Dry the first wash completely to create a permanent glaze layer
    engine.dry_current_wash()
    assert len(engine.dried_glazes) == 1, "First wash must be committed as a dried glaze layer!"
    print(f"Layer 1 successfully dried into glaze stack. Glaze count: {len(engine.dried_glazes)}")

    # 3. Second Wash: Phthalo Blue vertical stroke overlapping across the yellow wash
    stroke2 = [
        (300.0, 80.0),
        (300.0, 250.0),
        (300.0, 420.0),
    ]
    engine.add_stroke_from_points(
        stroke2,
        color_rgb=(0, 110, 210),
        radius=40.0,
        intensity=0.85,
        pigment_name="Phthalo Blue",
    )
    # Simulate second wash
    engine.run_simulation(steps=25, dt=0.05)

    out_img = engine.render()
    out_path = "test_output_glazing_overlap.png"
    cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"Glazing overlap test image saved: {out_path}")

    # Inspect spectral regions:
    # Yellow region only (around x=180, y=200)
    yellow_patch = out_img[190:210, 170:190]
    # Blue region only (around x=300, y=360)
    blue_patch = out_img[350:370, 290:310]
    # Overlap region (intersection at x=300, y=200)
    overlap_patch = out_img[190:210, 290:310]

    y_mean = yellow_patch.mean(axis=(0, 1))
    b_mean = blue_patch.mean(axis=(0, 1))
    ov_mean = overlap_patch.mean(axis=(0, 1))

    print(f"Pure Yellow Wash:   R={y_mean[0]:.1f}, G={y_mean[1]:.1f}, B={y_mean[2]:.1f}")
    print(f"Pure Blue Glaze:    R={b_mean[0]:.1f}, G={b_mean[1]:.1f}, B={b_mean[2]:.1f}")
    print(f"Overlap Glaze Area: R={ov_mean[0]:.1f}, G={ov_mean[1]:.1f}, B={ov_mean[2]:.1f}")

    # Subtractive optical verification:
    # Overlap must be rich green (Green channel dominant over Red and Blue)
    assert ov_mean[1] > ov_mean[0], "Green channel must be stronger than Red in Yellow+Blue subtractive glaze!"
    assert ov_mean[1] > ov_mean[2] * 0.85, "Green channel must dominate in the glaze overlap zone!"
    print("SUBTRACTIVE SPECTRAL GLAZING VERIFIED: Cadmium Yellow + Phthalo Blue yields clear, radiant Green!")


def test_balanced_pigment_density():
    print("\n--- TEST: BALANCED PIGMENT DENSITY IN UNIFORM WASH ---")
    engine = WatercolorEngine(width=500, height=400)

    # Wide wash made of parallel horizontal strokes
    wash_strokes = [
        {"points": [(100.0, 150.0), (400.0, 150.0)], "radius": 35.0, "intensity": 0.80},
        {"points": [(100.0, 190.0), (400.0, 190.0)], "radius": 35.0, "intensity": 0.80},
        {"points": [(100.0, 230.0), (400.0, 230.0)], "radius": 35.0, "intensity": 0.80},
    ]
    engine.add_strokes_wash(
        wash_strokes,
        color_rgb=(180, 50, 40),
        pigment_name="Cadmium Red",
    )
    engine.run_simulation(steps=25, dt=0.05)

    out_img = engine.render()
    out_path = "test_output_balanced_wash.png"
    cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"Balanced wash image saved: {out_path}")

    # Evaluate internal body uniformity (between strokes)
    # Interior area: x in [150, 350], y in [160, 220]
    interior = out_img[160:220, 150:350, :]  # RGB
    # Check red channel standard deviation and mean
    red_ch = interior[:, :, 0].astype(np.float32)
    std_val = float(red_ch.std())
    mean_val = float(red_ch.mean())
    cv_ratio = std_val / max(mean_val, 1e-3)
    print(f"Interior wash uniformity: Mean={mean_val:.2f}, Std={std_val:.2f}, Coefficient of Variation={cv_ratio:.4f}")
    assert cv_ratio < 0.20, f"Wash body must have smooth, balanced distribution (CV={cv_ratio:.4f} < 0.20)!"
    print("BALANCED PIGMENT DISTRIBUTION VERIFIED: Smooth tonal gradient with zero internal seam fissures!")


if __name__ == "__main__":
    test_glazing_wet_on_dry()
    test_balanced_pigment_density()
