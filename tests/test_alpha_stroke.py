import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import WaterColorCanvas


def test_alpha_stroke():
    print("\n--- TEST: KULLANICININ ÇİZDİĞİ ALPHA (a) FIRÇA DARBESİ TESTİ ---")
    wc = WaterColorCanvas(width=700, height=550)
    wc.generate_paper_texture(scale=150.0, octaves=5, intensity=0.15)

    # Kullanıcının çizdiği alfa döngüsü güzergahı
    # Sağ alttan başlar, yukarı çıkar, sola ilmek atar, ortadan kesip sağ alta iner
    alpha_points = [
        (480, 440),
        (420, 320),
        (370, 220),
        (310, 140),
        (220, 150),
        (160, 220),
        (150, 320),
        (210, 390),
        (310, 380),
        (370, 290),
        (440, 180),
    ]

    wc.add_stroke(
        points=alpha_points,
        color=(30, 95, 185),
        base_radius=28.0,
        intensity=0.85,
        edge_darkening=0.75,
        dry_brush=0.45,
        bristle_strength=0.40,
        paper_interaction=0.15,
    )

    wc.simulate_flow(iterations=12, gravity_strength=0.1, gravity_angle=90.0)
    wc.apply_drying_shift()

    out_path = "test_output_alpha_hybrid.png"
    wc.save_image(out_path)
    print(f"Yeni hibrit alfa darbesi kaydedildi: {out_path}")


if __name__ == "__main__":
    test_alpha_stroke()
