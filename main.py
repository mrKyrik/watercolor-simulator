"""
main.py - Fiziksel Sulu Boya Simülasyonu Ana Giriş Kapısı
Akışkanlar Mekaniği (Navier-Stokes / Sığ Su), Gözenekli Kağıt Kılcallığı,
Deegan Buharlaşması ve Kubelka-Munk Optik Motoru Entegrasyonu
"""

import math
import cv2
import numpy as np
import torch

from engine.simulation import WatercolorEngine
from engine.paper import PaperSubstrate
from engine.fluid import ShallowWaterSolver
from engine.pigment import PigmentManager
from engine.optics import KubelkaMunkRenderer


class WaterColorCanvas:
    """
    Fiziksel Sulu Boya Tuvali (Physical Watercolor Canvas)
    
    Tüm simülasyon hesaplamaları GPU VRAM üzerinde gerçek korunumlu fizik
    denklemleriyle (Navier-Stokes Sığ Su + Deegan Buharlaşması + Kubelka-Munk) yürütülür.
    Eski arayüz ve betiklerle tam geriye dönük uyumluluk sağlar.
    """

    def __init__(
        self,
        width: int = 1000,
        height: int = 800,
        base_color: tuple[int, int, int] = (246, 245, 236),
    ):
        self.width = width
        self.height = height
        self.base_color = base_color
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Yeni Çekirdek Fizik Motoru
        self.engine = WatercolorEngine(
            width=width,
            height=height,
            base_color=base_color,
            device=self.device,
        )

        self._rendered_cache = None

    @property
    def canvas(self) -> np.ndarray:
        """Tuvalin mevcut halini Kubelka-Munk ile render edip (H, W, 3) float32 döndürür."""
        if self._rendered_cache is None:
            self._rendered_cache = self.engine.render().astype(np.float32)
        return self._rendered_cache

    @canvas.setter
    def canvas(self, val: np.ndarray):
        self._rendered_cache = val.astype(np.float32)

    @property
    def texture_map(self) -> np.ndarray:
        """Kağıt dokusu yükseklik matrisini NumPy olarak döndürür."""
        return self.engine.paper.z_paper.cpu().numpy()

    # ------------------------------------------------------------------
    # Temel İşlemler & Kağıt Dokusu
    # ------------------------------------------------------------------

    def reset_canvas(self):
        """Tuvali ve tüm akışkan/pigment durumlarını sıfırla."""
        self.engine.reset()
        self._rendered_cache = None

    def generate_paper_texture(
        self,
        scale: float = 140.0,
        octaves: int = 5,
        intensity: float = 0.15,
    ):
        """Çok ölçekli fraktal tensör enterpolasyonu ile kağıt dokusunu yeniler."""
        self.engine.paper.generate_paper(
            scale=scale,
            octaves=octaves,
            roughness=max(0.05, intensity * 1.5),
        )
        self._rendered_cache = None

    # ------------------------------------------------------------------
    # Fırça Darbesi (Strokes)
    # ------------------------------------------------------------------

    def add_stroke(
        self,
        points: list[tuple[float, float]],
        color: tuple[int, int, int],
        base_radius: float = 25.0,
        intensity: float = 0.85,
        edge_darkening: float = 0.60,
        dry_brush: float = 0.35,
        bristle_strength: float = 0.35,
        paper_interaction: float = 0.12,
    ):
        """Fiziksel sığ su ve pigment taşıyan fırça darbesi uygular."""
        if not points:
            return

        self.engine.add_stroke_from_points(
            points=points,
            color_rgb=color,
            radius=base_radius,
            intensity=intensity,
            edge_darkening=edge_darkening,
            dry_brush=dry_brush,
            bristle_strength=bristle_strength,
            paper_interaction=paper_interaction,
        )
        self._rendered_cache = None

    def add_strokes_wash(self, strokes: list[dict], color: tuple[int, int, int] = None):
        """Birden çok birbirine değen veya paralel fırça darbesini TEK BİR birleşik sıvı gölü olarak işler."""
        if not strokes:
            return
        self.engine.add_strokes_wash(strokes, color_rgb=color)
        self._rendered_cache = None

    # ------------------------------------------------------------------
    # Damlalar & Kümeler (Drops & Clusters)
    # ------------------------------------------------------------------

    def add_pigment_cluster(self, drops: list[dict]):
        """Birleşik damla alanını fiziksel sıvı olarak tuvale ekler."""
        if not drops:
            return

        for d in drops:
            cx = d.get("x", self.width // 2)
            cy = d.get("y", self.height // 2)
            r = d.get("radius", 50)
            col = d.get("color", (50, 100, 200))
            inten = d.get("intensity", 0.8)

            self.engine.add_droplet(
                cx=float(cx),
                cy=float(cy),
                radius=float(r),
                color_rgb=col,
                water_amount=0.90,
                pigment_concentration=inten * 1.3,
            )

        self._rendered_cache = None

    def add_pigment(
        self,
        center_x: int,
        center_y: int,
        color: tuple[int, int, int],
        radius: int = 50,
        intensity: float = 0.7,
        edge_darkening: float = 0.6,
        feathering: float = 0.4,
        paper_interaction: float = 0.12,
    ):
        """Tekil damla bırakma."""
        self.engine.add_droplet(
            cx=float(center_x),
            cy=float(center_y),
            radius=float(radius),
            color_rgb=color,
            water_amount=0.85,
            pigment_concentration=intensity * 1.3,
        )
        self._rendered_cache = None

    def add_splatter(
        self,
        center_x: int,
        center_y: int,
        color: tuple[int, int, int],
        radius: int = 50,
        count: int = 12,
        scatter_min: float = 1.2,
        scatter_max: float = 2.2,
        intensity: float = 0.75,
        wet_react: float = 0.85,
        dispersion: float = 1.70,
        seed: int = None,
    ):
        """Dinamik sıçrama parçacıkları."""
        self.engine.add_splatter(
            cx=float(center_x),
            cy=float(center_y),
            radius=float(radius),
            color_rgb=color,
            count=count,
            scatter_min=scatter_min,
            scatter_max=scatter_max,
            intensity=intensity,
            seed=seed,
        )
        self._rendered_cache = None

    def apply_wet_bloom(
        self,
        center_x: int,
        center_y: int,
        radius: int = 50,
        strength: float = 0.5,
        bloom_size: float = 0.20,
        octaves: int = 4,
    ):
        """Islak zemine temiz su dokunuşu (Backrun / Karnabahar etkisi)."""
        # Temiz su ekle (pigment konsantrasyonu 0, sadece su)
        dist = torch.sqrt((self.engine.grid_x - center_x) ** 2 + (self.engine.grid_y - center_y) ** 2)
        bloom_mask = torch.sigmoid((radius * 0.8 - dist) * 1.5)
        self.engine.fluid.add_fluid(bloom_mask, amount=strength * 1.5)
        self._rendered_cache = None

    # ------------------------------------------------------------------
    # Akışkan Dinamiği Simülasyonu
    # ------------------------------------------------------------------

    def simulate_flow(
        self,
        iterations: int = 15,
        diffusion_rate: float = 0.12,
        turbulence: float = 1.5,
        gravity_strength: float = 0.0,
        gravity_angle: float = 90.0,
    ):
        """2D Sığ Su Navier-Stokes fizik çözücüsünü koşturur."""
        steps = max(5, int(iterations * 1.5))
        self.engine.run_simulation(
            steps=steps,
            dt=0.045,
            gravity_strength=gravity_strength,
            gravity_angle_deg=gravity_angle,
        )
        self._rendered_cache = None

    def apply_drying_shift(
        self,
        value_shift: float = 0.05,
        saturation_shift: float = -0.08,
    ):
        """
        Fiziksel kuruma süreci Kubelka-Munk optik katmanında doğrudan modellenir.
        Ek buharlaşma adımı atarak serbest suyu kağıda sabitler.
        """
        # Sıvı filmini tamamen buharlaştırıp pigmenti liflere kilitler
        self.engine.fluid.h *= 0.15
        self._rendered_cache = None

    def save_image(self, filename: str = "output_paper.png"):
        self.engine.save_image(filename)
        print(f"Fiziksel simülasyon kaydedildi: {filename}")


if __name__ == "__main__":
    print("Fiziksel Sulu Boya Tuvali Başlatılıyor (GPU / Navier-Stokes + Kubelka-Munk)...")
    wc = WaterColorCanvas(width=1000, height=800)
    wc.generate_paper_texture(scale=160.0, octaves=5, intensity=0.15)

    s_curve_points = [(150, 200), (300, 150), (500, 350), (700, 550), (850, 450)]
    wc.add_stroke(s_curve_points, color=(35, 115, 210), base_radius=35.0, intensity=0.90)

    # Bir de damla ve sıçrama ekleyelim
    wc.add_pigment(500, 350, color=(235, 180, 20), radius=55, intensity=0.85)
    wc.add_splatter(700, 550, color=(210, 40, 60), radius=60, count=12)

    wc.simulate_flow(iterations=20, gravity_strength=0.2, gravity_angle=90.0)
    wc.apply_drying_shift()
    wc.save_image("output_paper.png")