import os
import sys
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.simulation import WatercolorEngine


def run_gravity_drip():
    print("\n--- TEST: DİNAMİK YERÇEKİMİ AKIŞI VE DAMLA SÜZÜLMESİ ---")
    engine = WatercolorEngine(width=400, height=500)
    cx, cy = 200, 100
    engine.add_droplet(cx, cy, radius=35, color_rgb=(15, 60, 200), water_amount=2.0, pigment_name="Phthalo Blue")
    
    y_coords = engine.grid_y
    lyr = engine.pigments.layers[0]
    p_init = lyr.c + lyr.d
    y_cm_initial = (p_init * y_coords).sum() / max(p_init.sum(), 1e-5)
    print(f"Başlangıç Pigment Ağırlık Merkezi Y: {y_cm_initial.item():.2f}")
    
    # Yerçekimi ile 30 adım koştur
    for step in range(30):
        engine.step(dt=0.04, gravity_strength=1.2, gravity_angle_deg=90.0)
    
    p_final = lyr.c + lyr.d
    y_cm_final = (p_final * y_coords).sum() / max(p_final.sum(), 1e-5)
    print(f"30 Adım Sonrası Pigment Ağırlık Merkezi Y: {y_cm_final.item():.2f}")
    delta_y = (y_cm_final - y_cm_initial).item()
    print(f"Aşağı Doğru Net Pigment İlerlemesi: {delta_y:.2f} piksel")
    
    out_img = engine.render()
    out_path = "test_output_gravity_drip.png"
    cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"Kaydedildi: {out_path}")
    assert delta_y > 1.0, "Yerçekimi pigmenti aşağı doğru sürüklemelidir!"


if __name__ == "__main__":
    run_gravity_drip()
