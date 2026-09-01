import math
import torch
import torch.nn.functional as F


class PigmentLayer:
    """
    Tekil bir Pigment Türünün Fiziksel Durumu ve Spektral Özellikleri
    """

    def __init__(
        self,
        name: str,
        color_rgb: tuple[float, float, float],
        km_k: tuple[float, float, float],
        km_s: tuple[float, float, float],
        granulation: float = 0.5,
        staining: float = 0.5,
        diffusivity: float = 0.08,
        device: torch.device = None,
        height: int = 600,
        width: int = 800,
    ):
        self.name = name
        self.color_rgb = color_rgb
        self.km_k = torch.tensor(km_k, dtype=torch.float32, device=device)
        self.km_s = torch.tensor(km_s, dtype=torch.float32, device=device)
        self.granulation = granulation
        self.staining = staining
        self.diffusivity = diffusivity
        self.device = device

        # Durum Alanları (GPU)
        self.c = torch.zeros((height, width), dtype=torch.float32, device=device)  # Askıdaki serbest pigment
        self.d = torch.zeros((height, width), dtype=torch.float32, device=device)  # Kağıda kilitlenmiş pigment


class PigmentManager:
    """
    Koloidal Pigment Taşınım, Sedimantasyon, Granülasyon ve Deegan Kenar Yığılma Motoru
    """

    # Gerçek Sanatçı Pigmentlerinin Spektrofotometrik Referans Verileri
    ARTIST_PIGMENTS = {
        "Phthalo Blue": {
            "color": (0.00, 0.45, 0.85),
            "k": (0.95, 0.04, 0.01),      # Kırmızı tam emilir, yeşil ve mavi sıfıra yakın geçer!
            "s": (0.05, 0.70, 0.95),
            "granulation": 0.08,
            "staining": 0.85,
            "diffusivity": 0.12,
        },
        "Ultramarine Blue": {
            "color": (0.12, 0.24, 0.88),
            "k": (0.85, 0.38, 0.02),
            "s": (0.15, 0.45, 0.90),
            "granulation": 0.85,          # Yoğun vadi sedimantasyonu
            "staining": 0.35,
            "diffusivity": 0.06,
        },
        "Cadmium Yellow": {
            "color": (0.98, 0.85, 0.02),
            "k": (0.01, 0.04, 0.98),      # Mavi tam emilir, kırmızı ve yeşil serbest yansır
            "s": (0.95, 0.88, 0.05),
            "granulation": 0.25,
            "staining": 0.50,
            "diffusivity": 0.10,
        },
        "Quinacridone Rose": {
            "color": (0.88, 0.10, 0.42),
            "k": (0.04, 0.96, 0.20),      # Yeşil tam emilir, kırmızı ve mavi yansır (Macenta)
            "s": (0.85, 0.10, 0.50),
            "granulation": 0.12,
            "staining": 0.80,
            "diffusivity": 0.10,
        },
        "Cadmium Red": {
            "color": (0.88, 0.15, 0.12),
            "k": (0.02, 0.92, 0.95),      # Yeşil ve mavi emilir, kırmızı tam yansır
            "s": (0.90, 0.15, 0.10),
            "granulation": 0.35,
            "staining": 0.60,
            "diffusivity": 0.08,
        },
        "Burnt Umber": {
            "color": (0.42, 0.24, 0.12),
            "k": (0.55, 0.78, 0.88),
            "s": (0.40, 0.25, 0.15),
            "granulation": 0.92,          # Maksimum mineral granülasyonu
            "staining": 0.40,
            "diffusivity": 0.05,
        },
    }

    def __init__(self, width: int, height: int, device: torch.device = None):
        self.width = width
        self.height = height
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers: list[PigmentLayer] = []

        self._laplace_kernel = torch.tensor(
            [[[[0.0, 1.0, 0.0],
               [1.0, -4.0, 1.0],
               [0.0, 1.0, 0.0]]]],
            dtype=torch.float32,
            device=self.device,
        )

    def get_or_create_layer(
        self,
        name: str,
        color_rgb: tuple[float, float, float] = None,
        granulation: float = 0.5,
        staining: float = 0.5,
    ) -> PigmentLayer:
        for lyr in self.layers:
            if lyr.name == name:
                return lyr

        if name in self.ARTIST_PIGMENTS:
            spec = self.ARTIST_PIGMENTS[name]
            col = spec["color"]
            k = spec["k"]
            s = spec["s"]
            gran = spec["granulation"]
            stain = spec["staining"]
            diff = spec["diffusivity"]
        else:
            col = color_rgb or (0.2, 0.5, 0.8)
            # Otomatik K-S çıkarımı
            k = (max(0.01, 1.0 - col[0]), max(0.01, 1.0 - col[1]), max(0.01, 1.0 - col[2]))
            s = (max(0.05, col[0] * 0.8), max(0.05, col[1] * 0.8), max(0.05, col[2] * 0.8))
            gran = granulation
            stain = staining
            diff = 0.08

        layer = PigmentLayer(
            name=name,
            color_rgb=col,
            km_k=k,
            km_s=s,
            granulation=gran,
            staining=stain,
            diffusivity=diff,
            device=self.device,
            height=self.height,
            width=self.width,
        )
        self.layers.append(layer)
        return layer

    def deposit_stroke(
        self,
        layer: PigmentLayer,
        mask: torch.Tensor,
        concentration: float = 1.0,
        initial_stain_ratio: float = 0.12,
    ):
        """Fırça veya damla temasında pigmenti sıvı katmanına ve liflere bırakır."""
        added = mask * concentration
        # Sadece küçük bir kısım ilk anda liflere tutunur
        stain = added * initial_stain_ratio * layer.staining
        layer.d += stain
        # Asıl büyük kısım serbest sıvı içinde asılı kalır ve akışla kenara sürüklenir
        layer.c += (added - stain)

    def step(
        self,
        dt: float,
        u: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
        z_paper: torch.Tensor,
        pinning_mask: torch.Tensor,
        sample_coords: torch.Tensor,
    ):
        """
        Her pigment için adveksiyon, difüzyon, Deegan kenar kilitlenmesi ve granülasyon
        """
        speed = torch.sqrt(u ** 2 + v ** 2)
        valley_depth = torch.clamp(1.0 - z_paper / 0.25, 0.0, 1.0)

        # Deegan Kenar Çökelme Koşulu:
        # Sıvı filminin inceldiği (h < 0.20) temas hattında dengeli birikme
        edge_zone = torch.clamp((0.20 - h) / 0.18, 0.0, 1.0) * (pinning_mask * 0.5 + 0.5)

        for lyr in self.layers:
            # 1. SEMI-LAGRANGIAN ADVEKSİYON (u, v hız alanıyla sürüklenme)
            c_in = lyr.c.unsqueeze(0).unsqueeze(0)
            c_adv = F.grid_sample(
                c_in, sample_coords, mode="bilinear", padding_mode="border", align_corners=True
            ).squeeze()

            # 2. BROWNİAN DİFÜZYON (Islak göllerde dinamik boya kaynaşması)
            pad_c = F.pad(c_adv.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
            laplace_c = F.conv2d(pad_c, self._laplace_kernel).squeeze()
            wet_boost = 1.0 + 18.0 * torch.clamp(h / 0.25, 0.0, 1.0)
            c_diffused = torch.clamp(c_adv + lyr.diffusivity * wet_boost * laplace_c * dt * 4.0, min=0.0)

            # 3. DEEGAN KENAR KOYULAŞMASI & VADİ GRANÜLASYONU
            dryness = torch.clamp(1.0 - h / 0.10, 0.0, 1.0)
            quiescence = torch.clamp(1.0 - speed / 2.0, 0.0, 1.0)

            # Çökelme hızı:
            # a) Kenar buharlaşma zonunda dengeli menisküs kilitlenmesi
            # b) Vadilerde zengin mineral granülasyonu
            # c) Kuruyan bölgelerde lif tutunması
            deposit_rate = (
                0.04 +
                0.45 * edge_zone +
                0.35 * dryness +
                lyr.granulation * 0.65 * valley_depth * quiescence
            )
            deposit_amount = torch.min(c_diffused, c_diffused * deposit_rate * dt * 2.5)

            lyr.c = torch.clamp(c_diffused - deposit_amount, min=0.0)
            lyr.d += deposit_amount

            # 4. YENİDEN KALKMA (LIFTING)
            if lyr.staining < 0.75:
                lift_rate = 0.02 * (1.0 - lyr.staining) * torch.clamp(speed / 3.0, 0.0, 1.0) * (h > 0.15).float()
                lift_amount = torch.min(lyr.d, lyr.d * lift_rate * dt)
                lyr.d -= lift_amount
                lyr.c += lift_amount

            # Sınır koşulları
            lyr.c[0, :] = lyr.c[-1, :] = lyr.c[:, 0] = lyr.c[:, -1] = 0.0
            lyr.d[0, :] = lyr.d[-1, :] = lyr.d[:, 0] = lyr.d[:, -1] = 0.0

    def reset(self):
        for lyr in self.layers:
            lyr.c.zero_()
            lyr.d.zero_()
