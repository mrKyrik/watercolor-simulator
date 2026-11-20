import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import WaterColorCanvas


def test_parallel_strokes_coalescence():
    print("\n--- TEST: PARALLEL STROKES AND WET-IN-WET COALESCENCE TEST ---")
    wc = WaterColorCanvas(width=700, height=450)
    wc.generate_paper_texture(scale=150.0, octaves=5, intensity=0.15)

    # 4 naturally overlapping parallel brush strokes (Flat wash fill)
    y_coords = [150, 178, 206, 234]
    color = (25, 80, 175)

    stroke_list = []
    for i, y in enumerate(y_coords):
        pts = [
            (80, y + 5),
            (220, y - 4),
            (360, y + 6),
            (500, y - 5),
            (620, y + 2),
        ]
        stroke_list.append({
            "points": pts,
            "color": color,
            "radius": 28.0,
            "intensity": 0.90,
            "edge_darkening": 0.75,
            "dry_brush": 0.30,
            "bristle_strength": 0.30,
            "paper_interaction": 0.12,
        })

    wc.add_strokes_wash(stroke_list, color=color)

    # Fluid levels out coalesced pool and homogenizes free pigment
    wc.simulate_flow(iterations=15, gravity_strength=0.06, gravity_angle=90.0)
    wc.apply_drying_shift()

    out_path = "test_output_parallel_coalescence.png"
    wc.save_image(out_path)
    print(f"Parallel stroke coalescence output saved: {out_path}")


if __name__ == "__main__":
    test_parallel_strokes_coalescence()
