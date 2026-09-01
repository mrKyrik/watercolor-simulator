import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import WaterColorCanvas


def test_parallel_strokes_coalescence():
    print("\n--- TEST: PARALEL DARBELER VE ISLAK-ISLAK BİRLEŞME (COALESCENCE) TESTİ ---")
    # Kullanıcının deneyindeki gibi birleşik dikdörtgen dolgu testi
    wc = WaterColorCanvas(width=700, height=450)
    wc.generate_paper_texture(scale=150.0, octaves=5, intensity=0.15)

    # Doğal örtüşen 4 paralel fırça darbesi (Düz bir dolgu yıkaması)
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

    # Sıvı akışkanı birleşen gölü seviyeler ve serbest pigmenti homojenleştirir
    wc.simulate_flow(iterations=15, gravity_strength=0.06, gravity_angle=90.0)
    wc.apply_drying_shift()

    out_path = "test_output_parallel_coalescence.png"
    wc.save_image(out_path)
    print(f"Paralel darbe birleşme çıktısı kaydedildi: {out_path}")


if __name__ == "__main__":
    test_parallel_strokes_coalescence()
