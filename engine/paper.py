import math
import torch
import torch.nn.functional as F


class PaperSubstrate:
    """
    Gözenekli Selüloz Kağıt Katmanı (Porous Cellulose Paper Substrate)
    
    Fiziksel Özellikler:
    - 3D lif topoğrafyası z_paper(x, y)
    - Gözenek boşluk oranı (Porozite) phi(x, y)
    - Lif nem doygunluğu m(x, y) in [0, M_max]
    - Kılcal emiş potansiyeli ve Darcy difüzyonu
    - Mikroskobik temas açısı histerisisi ve kilitlenme (Pinning)
    """

    def __init__(
        self,
        width: int,
        height: int,
        scale: float = 120.0,
        octaves: int = 5,
        roughness: float = 0.25,
        porosity: float = 0.45,
        max_capacity: float = 1.0,
        absorb_rate: float = 0.08,
        fiber_diffusivity: float = 0.04,
        device: torch.device = None,
    ):
        self.width = width
        self.height = height
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.roughness = roughness
        self.base_porosity = porosity
        self.max_capacity = max_capacity
        self.absorb_rate = absorb_rate
        self.fiber_diffusivity = fiber_diffusivity

        # Durum Değişkenleri (GPU Tensörleri)
        self.z_paper = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        self.grad_zx = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        self.grad_zy = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        self.porosity = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        self.m = torch.zeros((height, width), dtype=torch.float32, device=self.device)  # Kağıt içindeki su
        self.pinning = torch.zeros((height, width), dtype=torch.float32, device=self.device)

        self.generate_paper(scale=scale, octaves=octaves, roughness=roughness)

    def generate_paper(self, scale: float = 120.0, octaves: int = 5, roughness: float = 0.25):
        """Çok ölçekli fraktal tensör enterpolasyonu ile gerçekçi kağıt lif dokusu üretir."""
        self.roughness = roughness
        h, w = self.height, self.width

        # Tensör tabanlı deterministik fraktal gürültü
        noise = torch.zeros((1, 1, h, w), dtype=torch.float32, device=self.device)
        cur_scale = max(scale, 1.0)
        amp = 1.0
        tot_amp = 0.0

        generator = torch.Generator(device=self.device).manual_seed(42)

        for _ in range(octaves):
            gw = max(2, int(math.ceil(w / cur_scale)) + 2)
            gh = max(2, int(math.ceil(h / cur_scale)) + 2)
            grid = torch.rand((1, 1, gh, gw), generator=generator, dtype=torch.float32, device=self.device)
            # Bicubic interpolasyon ile pürüzsüz lif dalgalanmaları
            upsampled = F.interpolate(grid, size=(h, w), mode="bicubic", align_corners=False)
            noise += upsampled * amp
            tot_amp += amp
            cur_scale /= 2.0
            amp *= 0.52

        noise = noise.squeeze() / max(tot_amp, 1e-5)
        # Normalizasyon [0, 1]
        n_min, n_max = noise.min(), noise.max()
        self.texture_norm = ((noise - n_min) / (n_max - n_min + 1e-7)).squeeze()
        self.z_paper = self.texture_norm * self.roughness

        # Topoğrafik gradyanlar (Eğim vektörü grad z)
        pad_z = F.pad(self.z_paper.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate").squeeze()
        self.grad_zx = 0.5 * (pad_z[1:-1, 2:] - pad_z[1:-1, :-2])
        self.grad_zy = 0.5 * (pad_z[2:, 1:-1] - pad_z[:-2, 1:-1])

        # Porozite (vadilerde ve lif aralarında gözeneklilik daha yüksektir)
        self.porosity = torch.clamp(self.base_porosity + (0.5 - self.z_paper) * 0.35, 0.1, 0.9)

        # Pinning matrisi: Eğim ne kadar yüksek ve mikroskobik pürüz ne kadar fazlaysa temas hattı o kadar kilitlenir
        grad_mag = torch.sqrt(self.grad_zx ** 2 + self.grad_zy ** 2)
        self.pinning = torch.clamp(grad_mag * 4.0 + self.z_paper * 0.5, 0.0, 1.0)

    def absorb_from_fluid(self, h_fluid: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Darcy & Washburn kılcal emilimi:
        Serbest su katmanından (h_fluid) kağıt lif doygunluğuna (m) su transferi.
        Dönen değer: Yüzeyden emilen sıvı debisi A(x, y).
        """
        # Kağıdın kalan boş kapasitesi
        capacity_deficit = torch.clamp(self.max_capacity - self.m, min=0.0)
        
        # Emme katsayısı: Kuru kağıt lifleri suyu kılcal emişle hızla çeker
        capillary_draw = self.absorb_rate * self.porosity * capacity_deficit
        
        # Gerçekte transfer edilebilecek su miktarı serbest su kalınlığıyla sınırlıdır
        absorbed = torch.min(h_fluid, capillary_draw * dt)
        
        self.m += absorbed
        return absorbed

    def diffuse_fiber_moisture(self, dt: float):
        """Kağıt lifleri boyunca suyun kılcal yayılımı (Richards gözenekli akışı)."""
        pad_m = F.pad(self.m.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
        
        # 5 noktalı Laplace stencil
        laplacian_kernel = torch.tensor(
            [[[[0.0, 1.0, 0.0],
               [1.0, -4.0, 1.0],
               [0.0, 1.0, 0.0]]]],
            dtype=torch.float32,
            device=self.device,
        )
        laplace_m = F.conv2d(pad_m, laplacian_kernel).squeeze()
        
        # Lif difüzyonu
        self.m += self.fiber_diffusivity * laplace_m * dt
        self.m = torch.clamp(self.m, 0.0, self.max_capacity)

    def reset_moisture(self):
        """Kağıdı tamamen kurutur."""
        self.m.zero_()
