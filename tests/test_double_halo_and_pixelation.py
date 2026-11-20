import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import WaterColorCanvas


def test_halo_and_pixelation():
    print("\n--- TEST: DOUBLE HALO AND PIXELATION ELIMINATION TEST ---")
    wc = WaterColorCanvas(width=800, height=600)
    wc.generate_paper_texture(scale=150.0, octaves=5, intensity=0.15)

    # 1. Intense, dark blue brush stroke (Wide and curved)
    stroke_1 = [
        (150, 450),
        (250, 380),
        (380, 260),
        (520, 180),
        (650, 160),
    ]
    wc.add_stroke(
        points=stroke_1,
        color=(20, 70, 170),
        base_radius=32.0,
        intensity=0.95,
        edge_darkening=0.75,
        dry_brush=0.35,
        bristle_strength=0.35,
        paper_interaction=0.12,
    )

    # 2. Intersection test: Second stroke intersecting stroke 1 (Tests intersection over-darkening)
    stroke_2 = [
        (420, 100),
        (400, 240),
        (380, 380),
        (360, 520),
    ]
    wc.add_stroke(
        points=stroke_2,
        color=(20, 70, 170),
        base_radius=30.0,
        intensity=0.95,
        edge_darkening=0.75,
        dry_brush=0.35,
        bristle_strength=0.35,
        paper_interaction=0.12,
    )

    wc.simulate_flow(iterations=12, gravity_strength=0.08, gravity_angle=90.0)
    wc.apply_drying_shift()

    out_path = "test_output_anti_halo_smooth.png"
    wc.save_image(out_path)
    print(f"Test output saved: {out_path}")


if __name__ == "__main__":
    test_halo_and_pixelation()
