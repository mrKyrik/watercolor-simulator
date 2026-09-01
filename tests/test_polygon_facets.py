import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import WaterColorCanvas


def test_polygon_facets():
    print("\n--- TEST: ÖNİZLEME (0.4) VS TAM ÇIKTI (1.0) VE BEYAZ POLİGONLAR ---")

    # Kullanıcının çizdiği gibi bir blok oluşturan 7 darbe:
    base_strokes = []
    # Yatay ve dikey zig-zag darbeler
    y_levels = [150, 200, 250, 300, 350]
    for y in y_levels:
        pts = [(150, y), (300, y + 10), (500, y - 10), (700, y + 5), (850, y)]
        base_strokes.append({
            "points": pts,
            "color": (0, 0, 127),
            "radius": 73.0,
            "intensity": 1.0,
            "edge_darkening": 0.60,
            "dry_brush": 0.45,
            "bristle_strength": 0.35,
            "paper_interaction": 0.12,
        })
    # İki dikey darbe
    base_strokes.append({
        "points": [(250, 120), (250, 250), (260, 380)],
        "color": (0, 0, 127),
        "radius": 73.0,
        "intensity": 1.0,
        "edge_darkening": 0.60,
        "dry_brush": 0.45,
        "bristle_strength": 0.35,
        "paper_interaction": 0.12,
    })
    base_strokes.append({
        "points": [(750, 120), (740, 250), (750, 380)],
        "color": (0, 0, 127),
        "radius": 73.0,
        "intensity": 1.0,
        "edge_darkening": 0.60,
        "dry_brush": 0.45,
        "bristle_strength": 0.35,
        "paper_interaction": 0.12,
    })

    # 1. ÖNİZLEME (scale = 0.4)
    scale_prev = 0.4
    w_p, h_p = int(1000 * scale_prev), int(800 * scale_prev)
    wc_prev = WaterColorCanvas(width=w_p, height=h_p)
    wc_prev.generate_paper_texture(scale=200, octaves=6, intensity=0.12)
    scaled_p_strokes = []
    for st in base_strokes:
        s_pts = [(p[0] * scale_prev, p[1] * scale_prev) for p in st["points"]]
        st_c = dict(st)
        st_c["points"] = s_pts
        st_c["radius"] = max(3.0, st["radius"] * scale_prev)
        scaled_p_strokes.append(st_c)
    wc_prev.add_strokes_wash(scaled_p_strokes)
    wc_prev.simulate_flow(iterations=10)
    wc_prev.apply_drying_shift()
    wc_prev.save_image("test_output_facets_preview.png")

    # 2. TAM ÇIKTI (scale = 1.0)
    w_f, h_f = 1000, 800
    wc_full = WaterColorCanvas(width=w_f, height=h_f)
    wc_full.generate_paper_texture(scale=200, octaves=6, intensity=0.12)
    wc_full.add_strokes_wash(base_strokes)
    wc_full.simulate_flow(iterations=10)
    wc_full.apply_drying_shift()
    wc_full.save_image("test_output_facets_full.png")

    print("Test tamamlandı. Görseller oluşturuldu.")


if __name__ == "__main__":
    test_polygon_facets()
