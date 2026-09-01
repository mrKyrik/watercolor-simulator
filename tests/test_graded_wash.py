import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import WaterColorCanvas


def test_graded_wash():
    print("\n--- TEST: KULLANICININ DENEDİĞİ ÇOK RENKLİ GRADED WASH TESTİ ---")
    wc = WaterColorCanvas(width=700, height=550)
    wc.generate_paper_texture(scale=150.0, octaves=5, intensity=0.15)

    # Kullanıcının arayüz ekran görüntüsündeki 4 renk ve koordinatlar:
    # 1. #AAFFFF -> (170, 255, 255)
    # 2. #00FFFF -> (0, 255, 255)
    # 3. #00AEFF -> (0, 174, 255)
    # 4. #0084FF -> (0, 132, 255)
    color_steps = [
        ((170, 255, 255), 150),
        ((0, 255, 255), 200),
        ((0, 174, 255), 250),
        ((0, 132, 255), 300),
    ]

    stroke_list = []
    for col, y in color_steps:
        pts = [
            (80, y + 4),
            (220, y - 5),
            (360, y + 6),
            (500, y - 4),
            (620, y + 3),
        ]
        stroke_list.append({
            "points": pts,
            "color": col,
            "radius": 45.0,  # Geniş ve birbirine tam temas eden fırça
            "intensity": 0.85,
            "edge_darkening": 0.60,
            "dry_brush": 0.30,
            "bristle_strength": 0.30,
            "paper_interaction": 0.12,
        })

    print("Çok-renkli birleşik akışkan alanı (Unified Multi-Color Wash) ekleniyor...")
    wc.add_strokes_wash(stroke_list)

    # Sıvı akışkanı seviyeleme ve güçlendirilmiş ıslak difüzyon
    wc.simulate_flow(iterations=15, gravity_strength=0.06, gravity_angle=90.0)
    wc.apply_drying_shift()

    out_path = "test_output_graded_wash_smooth.png"
    wc.save_image(out_path)
    print(f"Graded Wash test çıktısı kaydedildi: {out_path}")


if __name__ == "__main__":
    test_graded_wash()
