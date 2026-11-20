import math
import torch
import torch.nn.functional as F


class KubelkaMunkRenderer:
    """
    Kubelka-Munk Radiative Transfer Optical Engine
    
    Physical Features:
    - Multi-pigment blending (Duncan's Rule: K_mix = sum(c_i * K_i), S_mix = sum(c_i * S_i))
    - Authentic paint subtractive mixing via spectral absorption and scattering (Cyan + Yellow -> Clear Green)
    - Porous paper substrate reflection R_paper
    - Wet fluid film surface sheen (Fresnel water gloss)
    - Drying phase optical transformation (Drying shift: matting and refractive index matching)
    """

    def __init__(
        self,
        base_paper_color: tuple[float, float, float] = (0.96, 0.95, 0.90),
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Dry paper RGB reflectance coefficient R_g
        self.r_paper_base = torch.tensor(base_paper_color, dtype=torch.float32, device=self.device).view(3, 1, 1)

    @staticmethod
    def compute_layer_rt(
        k: torch.Tensor,
        s: torch.Tensor,
        thickness: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the reflectance R0 and transmittance T0 of a paint film of thickness x
        over an ideal black background (Rg = 0) using Kubelka-Munk equations.
        k, s: (3, H, W) or (3, 1, 1) float32
        thickness: (1, H, W) float32
        """
        s_safe = torch.clamp(s, min=1e-3)
        k_safe = torch.clamp(k, min=1e-4)

        a = 1.0 + (k_safe / s_safe)
        b = torch.sqrt(torch.clamp(a ** 2 - 1.0, min=1e-6))

        bsx = torch.clamp(b * s_safe * thickness, min=1e-5, max=18.0)

        sinh_bsx = torch.sinh(bsx)
        cosh_bsx = torch.cosh(bsx)

        denom = torch.clamp(a * sinh_bsx + b * cosh_bsx, min=1e-5)
        r_0 = sinh_bsx / denom
        t_0 = b / denom
        return r_0, t_0

    @staticmethod
    def composite_over_substrate(
        r_0: torch.Tensor,
        t_0: torch.Tensor,
        r_substrate: torch.Tensor,
        thickness: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculates composite reflectance R of a layer (r_0, t_0) over a substrate r_substrate:
        R = r_0 + (t_0^2 * r_substrate) / (1.0 - r_0 * r_substrate)
        """
        sub_reflect_denom = torch.clamp(1.0 - r_0 * r_substrate, min=1e-4)
        r_composite = r_0 + (t_0 ** 2 * r_substrate) / sub_reflect_denom

        if thickness is not None:
            # Numerical smoothing near zero thickness for sub-pixel boundary antialiasing
            blend_t = torch.clamp(thickness / 0.005, 0.0, 1.0)
            smooth_blend = blend_t * blend_t * (3.0 - 2.0 * blend_t)
            return (1.0 - smooth_blend) * r_substrate + smooth_blend * r_composite
        return r_composite

    def render(
        self,
        pigment_layers: list,
        h_fluid: torch.Tensor,
        m_paper: torch.Tensor,
        z_paper: torch.Tensor,
        dried_glazes: list = None,
        paper_tint_intensity: float = 0.12,
    ) -> torch.Tensor:
        """
        Converts all physical layers and dried glazes into the final RGB image using
        Multi-Layer Kubelka-Munk radiative transfer equations.
        Output Shape: (H, W, 3) in [0, 255] float32.
        """
        h_dim, w_dim = z_paper.shape

        # -------------------------------------------------------------
        # 1. PAPER SUBSTRATE OPTICS (R_paper)
        # Wet paper fibers (high m_paper) appear darker and deeper due to
        # cellulose-water refractive index matching (wet paper darkening).
        # -------------------------------------------------------------
        paper_topo = 1.0 - (z_paper.unsqueeze(0) / 0.3) * paper_tint_intensity
        moisture_darkening = 1.0 - torch.clamp(m_paper.unsqueeze(0), 0.0, 1.0) * 0.04
        r_substrate = torch.clamp(self.r_paper_base * paper_topo * moisture_darkening, 0.05, 1.0)

        # -------------------------------------------------------------
        # 2. MULTI-LAYER DRIED GLAZES (Wet-on-Dry Stratified Optics)
        # Each dried glaze acts as an optical filter over underlying glazes and paper:
        # R_(k) = R0_k + (T0_k^2 * R_(k-1)) / (1.0 - R0_k * R_(k-1))
        # -------------------------------------------------------------
        if dried_glazes:
            for gz in dried_glazes:
                gz_k = gz["k"]
                gz_s = gz["s"]
                gz_thick = gz["thickness"]
                r_0_gz, t_0_gz = self.compute_layer_rt(gz_k, gz_s, gz_thick)
                r_substrate = self.composite_over_substrate(r_0_gz, t_0_gz, r_substrate, thickness=gz_thick)

        # -------------------------------------------------------------
        # 3. ACTIVE WET LAYER (Duncan's Mixing Rule + Kubelka-Munk)
        # Suspended and newly deposited pigments in active water pool
        # -------------------------------------------------------------
        k_total = torch.zeros((3, h_dim, w_dim), dtype=torch.float32, device=self.device)
        s_total = torch.zeros((3, h_dim, w_dim), dtype=torch.float32, device=self.device)
        total_thickness = torch.zeros((1, h_dim, w_dim), dtype=torch.float32, device=self.device)

        has_active_pigment = False
        if pigment_layers:
            for lyr in pigment_layers:
                pigment_conc = (lyr.c * 0.90 + lyr.d * 1.15).unsqueeze(0)
                if pigment_conc.max() > 1e-4:
                    has_active_pigment = True
                total_thickness += pigment_conc

                k_i = lyr.km_k.view(3, 1, 1)
                s_i = lyr.km_s.view(3, 1, 1)

                k_total += pigment_conc * k_i
                s_total += pigment_conc * s_i

        if has_active_pigment:
            r_0_act, t_0_act = self.compute_layer_rt(k_total, s_total, total_thickness)
            r_final = self.composite_over_substrate(r_0_act, t_0_act, r_substrate, thickness=total_thickness)
        else:
            r_final = r_substrate

        # -------------------------------------------------------------
        # 4. SURFACE MATTNESS AND REALISTIC PAPER ABSORPTION
        # Watercolor absorbed into paper integrates with cellulose fibers,
        # becoming deep, vibrant, and matte.
        # -------------------------------------------------------------
        rgb_out = r_final.permute(1, 2, 0) * 255.0
        return torch.clamp(rgb_out, 0.0, 255.0)
