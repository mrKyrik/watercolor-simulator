import math
import torch
import torch.nn.functional as F


class PigmentLayer:
    """
    Physical State and Spectral Properties of a Single Pigment Species
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

        # State Fields (GPU Tensors)
        self.c = torch.zeros((height, width), dtype=torch.float32, device=device)  # Suspended free pigment
        self.d = torch.zeros((height, width), dtype=torch.float32, device=device)  # Paper-pinned deposited pigment


class PigmentManager:
    """
    Colloidal Pigment Transport, Sedimentation, Granulation, and Deegan Edge Accumulation Engine
    """

    # Spectrophotometric Reference Data of Real Artist Pigments
    ARTIST_PIGMENTS = {
        "Phthalo Blue": {
            "color": (0.00, 0.45, 0.85),
            "k": (0.95, 0.04, 0.01),      # Red fully absorbed, green and blue transmitted near 100%
            "s": (0.05, 0.70, 0.95),
            "granulation": 0.08,
            "staining": 0.85,
            "diffusivity": 0.12,
        },
        "Ultramarine Blue": {
            "color": (0.12, 0.24, 0.88),
            "k": (0.85, 0.38, 0.02),
            "s": (0.15, 0.45, 0.90),
            "granulation": 0.85,          # Heavy valley sedimentation
            "staining": 0.35,
            "diffusivity": 0.06,
        },
        "Cadmium Yellow": {
            "color": (0.98, 0.85, 0.02),
            "k": (0.01, 0.04, 0.98),      # Blue fully absorbed, red and green freely reflected
            "s": (0.95, 0.88, 0.05),
            "granulation": 0.25,
            "staining": 0.50,
            "diffusivity": 0.10,
        },
        "Quinacridone Rose": {
            "color": (0.88, 0.10, 0.42),
            "k": (0.04, 0.96, 0.20),      # Green fully absorbed, red and blue reflected (Magenta)
            "s": (0.85, 0.10, 0.50),
            "granulation": 0.12,
            "staining": 0.80,
            "diffusivity": 0.10,
        },
        "Cadmium Red": {
            "color": (0.88, 0.15, 0.12),
            "k": (0.02, 0.92, 0.95),      # Green and blue absorbed, red fully reflected
            "s": (0.90, 0.15, 0.10),
            "granulation": 0.35,
            "staining": 0.60,
            "diffusivity": 0.08,
        },
        "Burnt Umber": {
            "color": (0.42, 0.24, 0.12),
            "k": (0.55, 0.78, 0.88),
            "s": (0.40, 0.25, 0.15),
            "granulation": 0.92,          # Maximum mineral granulation
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
            # Automatic K-S estimation
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
        """Deposits pigment into fluid layer and fibers upon brush stroke or drop contact."""
        added = mask * concentration
        # Only a small portion initial stains the fibers
        stain = added * initial_stain_ratio * layer.staining
        layer.d += stain
        # The main bulk remains suspended in free fluid and is advected towards the boundary
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
        Advection, diffusion, Deegan edge pinning, and granulation for each pigment layer.
        """
        speed = torch.sqrt(u ** 2 + v ** 2)
        valley_depth = torch.clamp(1.0 - z_paper / 0.25, 0.0, 1.0)

        # Deegan Edge Deposition Condition:
        # Balanced accumulation at the contact line where fluid thins (h < 0.20)
        edge_zone = torch.clamp((0.20 - h) / 0.18, 0.0, 1.0) * (pinning_mask * 0.5 + 0.5)

        for lyr in self.layers:
            # 1. SEMI-LAGRANGIAN ADVECTION (Advection via velocity field u, v)
            c_in = lyr.c.unsqueeze(0).unsqueeze(0)
            c_adv = F.grid_sample(
                c_in, sample_coords, mode="bilinear", padding_mode="border", align_corners=True
            ).squeeze()

            # 2. BROWNIAN DIFFUSION (Dynamic pigment blending in wet pools)
            pad_c = F.pad(c_adv.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
            laplace_c = F.conv2d(pad_c, self._laplace_kernel).squeeze()
            wet_boost = 1.0 + 18.0 * torch.clamp(h / 0.25, 0.0, 1.0)
            c_diffused = torch.clamp(c_adv + lyr.diffusivity * wet_boost * laplace_c * dt * 4.0, min=0.0)

            # 3. DEEGAN EDGE DARKENING & VALLEY GRANULATION
            # Real Deegan mechanism: At the evaporating contact line (thin h, pinned boundary),
            # particles jam and deposit heavily onto paper fibers.
            dryness = torch.clamp((0.15 - h) / 0.14, 0.0, 1.0)
            quiescence = torch.clamp(1.0 - speed / 1.8, 0.0, 1.0)

            # Capillary contact line pinning zone (high at thin rim, zero in deep center)
            contact_line_zone = torch.clamp((0.22 - h) / 0.18, 0.0, 1.0) * (pinning_mask * 0.7 + 0.3)

            # Deposition rate:
            # - Rim accumulation: Strong deposition at the pinned contact line (authentic coffee ring)
            # - Valley granulation: Heavy pigment settling into microscopic crevices
            # - Gentle background deposition in drying pools
            deposit_rate = (
                0.02 +
                0.85 * contact_line_zone +
                0.25 * dryness +
                lyr.granulation * 0.75 * valley_depth * quiescence
            )
            deposit_amount = torch.min(c_diffused, c_diffused * deposit_rate * dt * 3.2)

            lyr.c = torch.clamp(c_diffused - deposit_amount, min=0.0)
            lyr.d += deposit_amount

            # 4. RE-LIFTING (Rewetting and shear-induced particle mobilization)
            if lyr.staining < 0.75:
                lift_rate = 0.015 * (1.0 - lyr.staining) * torch.clamp(speed / 2.5, 0.0, 1.0) * (h > 0.12).float()
                lift_amount = torch.min(lyr.d, lyr.d * lift_rate * dt)
                lyr.d -= lift_amount
                lyr.c += lift_amount

            # Boundary conditions
            lyr.c[0, :] = lyr.c[-1, :] = lyr.c[:, 0] = lyr.c[:, -1] = 0.0
            lyr.d[0, :] = lyr.d[-1, :] = lyr.d[:, 0] = lyr.d[:, -1] = 0.0

    def reset(self):
        for lyr in self.layers:
            lyr.c.zero_()
            lyr.d.zero_()
