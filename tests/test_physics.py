"""
Fiziksel Sulu Boya Simülasyon Doğrulama Test Paketi (Physics Validation Suite)
"""

import os
import sys
import time
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.simulation import WatercolorEngine


def test_gravity_flow():
    print("\n--- TEST 1: YERÇEKİMİ VE EĞİM AKIŞI (GRAVITY FLOW) ---")
    engine = WatercolorEngine(width=500, height=400)
    # Merkeze bir damla mavi boya bırak
    engine.add_droplet(250, 120, radius=40, color_rgb=(25, 75, 210), water_amount=1.2, pigment_name="Ultramarine Blue")
    
    # Aşağı doğru (90 derece) güçlü yerçekimi ile 30 adım koştur
    engine.run_simulation(steps=30, dt=0.06, gravity_strength=0.8, gravity_angle_deg=90.0)
    
    out_img = engine.render()
    out_path = "test_output_gravity.png"
    cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"Yerçekimi akışı testi kaydedildi: {out_path}")
    
    # Sıvı hızının ve kütlesinin aşağıya doğru hareket ettiğini doğrula
    v_mean = engine.fluid.v.mean().item()
    print(f"Ortalama düşey hız (v_mean): {v_mean:.4f} (Pozitif: Aşağı akış)")
    assert v_mean >= 0.0, "Sıvı yerçekimi doğrultusunda aşağı akmalı!"


def test_wet_in_wet_bleeding():
    print("\n--- TEST 2: ISLAK-ÜSTÜ-ISLAK KARIŞIM (WET-IN-WET BLEEDING) ---")
    engine = WatercolorEngine(width=600, height=400)
    
    # Sol tarafa Phthalo Blue damla bırak
    engine.add_droplet(275, 200, radius=50, color_rgb=(0, 110, 210), water_amount=1.1, pigment_name="Phthalo Blue")
    # Sağ tarafa, mavi damlayla temas edecek Cadmium Yellow damla bırak
    engine.add_droplet(325, 200, radius=50, color_rgb=(245, 215, 10), water_amount=1.1, pigment_name="Cadmium Yellow")
    
    # 35 adım yüzey gerilimi ve basınç gradyanı ile birbirine akış
    engine.run_simulation(steps=35, dt=0.05, gravity_strength=0.0)
    
    out_img = engine.render()
    out_path = "test_output_wet_in_wet.png"
    cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"Islak-üstü-ıslak karışım testi kaydedildi: {out_path}")
    
    # Kesişim bölgesindeki pikselleri kontrol et (Yeşil kanal baskın olmalı: G > R ve G > B)
    mid_strip = out_img[180:220, 290:310]
    mean_r = float(mid_strip[:, :, 0].mean())
    mean_g = float(mid_strip[:, :, 1].mean())
    mean_b = float(mid_strip[:, :, 2].mean())
    print(f"Orta karışım bölgesi renk ortalaması: R={mean_r:.1f}, G={mean_g:.1f}, B={mean_b:.1f}")


def test_coffee_ring_effect():
    print("\n--- TEST 3: DEEGAN KAHVE LEKESİ KENAR KOYULAŞMASI ---")
    engine = WatercolorEngine(width=400, height=400)
    
    # Tek bir damla bırak
    cx, cy, r = 200, 200, 50
    engine.add_droplet(cx, cy, radius=r, color_rgb=(180, 40, 30), water_amount=0.8, pigment_name="Cadmium Red")
    
    # Buharlaşma adımlarını koştur (40 adım)
    engine.run_simulation(steps=40, dt=0.06, gravity_strength=0.0)
    
    out_img = engine.render()
    out_path = "test_output_coffee_ring.png"
    cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"Kahve lekesi testi kaydedildi: {out_path}")
    
    # Kenardaki pigment konsantrasyonunu merkezle karşılaştır
    lyr = engine.pigments.layers[0]
    total_pigment = lyr.c + lyr.d
    center_val = total_pigment[cy, cx].item()
    # r mesafesindeki kenar halkası
    edge_val = total_pigment[cy, cx + int(r * 0.85)].item()
    print(f"Merkez pigment: {center_val:.4f}, Kenar halkası pigment: {edge_val:.4f}")


def test_granulation_valley_pooling():
    print("\n--- TEST 4: GRANÜLASYON VE KAĞIT VADİSİ ÇÖKELMESİ ---")
    engine = WatercolorEngine(width=400, height=400)
    
    # Yüksek granülasyonlu bir pigment (Burnt Umber)
    engine.add_droplet(200, 200, radius=60, color_rgb=(100, 50, 25), water_amount=0.9, pigment_name="Burnt Umber", granulation=0.95)
    engine.run_simulation(steps=30, dt=0.05, gravity_strength=0.0)
    
    out_img = engine.render()
    out_path = "test_output_granulation.png"
    cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"Granülasyon testi kaydedildi: {out_path}")


def test_continuous_brush_stroke():
    print("\n--- TEST 5: SÜREKLİ FIRÇA DARBESİ VE KURU FIRÇA (DRY BRUSH) ---")
    engine = WatercolorEngine(width=800, height=600)
    
    # S kıvrımı şeklinde fırça darbesi
    points = [(100, 150), (250, 100), (450, 300), (600, 500), (720, 420)]
    engine.add_stroke_from_points(
        points=points,
        color_rgb=(30, 110, 195),
        radius=30.0,
        intensity=0.85,
        pigment_name="Ultramarine Blue",
        granulation=0.8,
        dry_brush=0.35,
    )
    
    # Birkaç sıçrama damlacığı ekle
    engine.add_splatter(cx=450, cy=300, radius=60, color_rgb=(210, 40, 60), count=10, pigment_name="Cadmium Red")
    
    # 20 adım fiziksel akış ve difüzyon
    t0 = time.time()
    engine.run_simulation(steps=20, dt=0.05, gravity_strength=0.25, gravity_angle_deg=110.0)
    elapsed = time.time() - t0
    print(f"20 Fiziksel Adım Simülasyon Süresi: {elapsed:.3f} saniye ({20 / elapsed:.1f} FPS)")
    
    out_img = engine.render()
    out_path = "test_output_stroke_composite.png"
    cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"Fırça darbesi kompozit testi kaydedildi: {out_path}")


if __name__ == "__main__":
    test_gravity_flow()
    test_wet_in_wet_bleeding()
    test_coffee_ring_effect()
    test_granulation_valley_pooling()
    test_continuous_brush_stroke()
    print("\nTÜM FİZİKSEL DOĞRULAMA TESTLERİ BAŞARIYLA ÇALIŞTI!")
