import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import WaterColorCanvas


def test_high_pigment_pixelation():
    print("\n--- TEST: HIGH PIGMENT EDGE PIXELATION AND ANTI-ALIASING TEST ---")
    wc = WaterColorCanvas(width=600, height=300)
    wc.generate_paper_texture(scale=140.0, octaves=5, intensity=0.15)

    # Intense, deep blue brush stroke
    stroke_points = [
        (80, 150),
        (200, 145),
        (350, 155),
        (520, 150),
    ]
    wc.add_stroke(
        points=stroke_points,
        color=(15, 45, 130),
        base_radius=32.0,
        intensity=1.0,
        edge_darkening=0.75,
        dry_brush=0.35,
        bristle_strength=0.35,
        paper_interaction=0.12,
    )

    wc.simulate_flow(iterations=10, gravity_strength=0.05, gravity_angle=90.0)
    wc.apply_drying_shift()

    arr = wc.canvas.astype(np.uint8)
    out_path = "test_output_high_pigment_smooth.png"
    cv2.imwrite(out_path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    print(f"Test output saved: {out_path}")

    # Edge profile spectrometry (Vertical slice at x=300)
    # Vertical slice passing through upper edge (y=100..130)
    slice_r = arr[100:130, 300, 0].astype(float)
    diffs = np.abs(np.diff(slice_r))
    max_jump = np.max(diffs)
    
    print(f"Upper edge vertical slice (Red Channel):\n{slice_r}")
    print(f"Maximum single-pixel color jump: {max_jump:.1f}")

    # Success criteria:
    # 1. No artificial cliff jump greater than 120 in a single pixel.
    # 2. Intermediate halftone transition pixels must exist.
    intermediate_pixels = np.sum((slice_r > 30) & (slice_r < 210))
    print(f"Detected halftone intermediate transition pixels: {intermediate_pixels}")

    assert max_jump < 120, f"Harsh color jump still exists on edge! Max jump: {max_jump}"
    assert intermediate_pixels >= 2, f"Insufficient intermediate transition pixels! Found: {intermediate_pixels}"
    print(">>> SUCCESS: High pigment edge smoothly and organically anti-aliased, pixel staircasing eliminated!\n")


if __name__ == "__main__":
    test_high_pigment_pixelation()
