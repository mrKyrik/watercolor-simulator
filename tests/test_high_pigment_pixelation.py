import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import WaterColorCanvas


def test_high_pigment_pixelation():
    print("\n--- TEST: YOĞUN PİGMENT KENAR PİKSELLEŞME VE MERDİVENLENME GİDERME TESTİ ---")
    wc = WaterColorCanvas(width=600, height=300)
    wc.generate_paper_texture(scale=140.0, octaves=5, intensity=0.15)

    # Yoğun, derin mavi fırça darbesi (Resim 1'deki gibi)
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
    print(f"Test çıktısı kaydedildi: {out_path}")

    # Kenar kesiti spektrometrisi (Dikey kesit x=300)
    # Üst kenardan geçen dikey dilim (y=100..130)
    slice_r = arr[100:130, 300, 0].astype(float)
    diffs = np.abs(np.diff(slice_r))
    max_jump = np.max(diffs)
    
    print(f"Üst kenar dikey kesit (Kırmızı Kanal):\n{slice_r}")
    print(f"Maksimum ardışık tek piksel renk sıçraması: {max_jump:.1f} (Mevcut bozuk motorda >160 idi)")

    # Başarı kriteri:
    # 1. Tek bir pikselde 120'den büyük yapay uçurum olmamalıdır.
    # 2. Ara geçiş pikselleri (yarı tonlar) bulunmalıdır.
    intermediate_pixels = np.sum((slice_r > 30) & (slice_r < 210))
    print(f"Tespit edilen yarı-ton ara geçiş piksel sayısı: {intermediate_pixels}")

    assert max_jump < 120, f"Kenarda hala çok sert renk sıçraması var! Max jump: {max_jump}"
    assert intermediate_pixels >= 2, f"Yetersiz ara geçiş pikseli! Bulunan: {intermediate_pixels}"
    print(">>> BAŞARILI: Yoğun pigment kenarı kesintisiz ve organik olarak yumuşatıldı, piksel basamakları giderildi!\n")


if __name__ == "__main__":
    test_high_pigment_pixelation()
