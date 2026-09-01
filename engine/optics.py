import math
import torch
import torch.nn.functional as F


class KubelkaMunkRenderer:
    """
    Kubelka-Munk Radyatif Transfer Optik Motoru
    
    Fiziksel Özellikler:
    - Çoklu pigment harmanlaması (Duncan Kuralı: K_mix = sum(c_i * K_i), S_mix = sum(c_i * S_i))
    - Spektral absorpsiyon ve saçılma yoluyla gerçek boya karışımı (Cyan + Sarı -> Berrak Yeşil)
    - Gözenekli kağıt altlığı yansıması R_paper
    - Islak sıvı filmi yüzey parıltısı (Fresnel su parlaklığı)
    - Kuruma fazı optik dönüşümü (Drying shift: matlaşma ve kırılma indisi eşleşmesi)
    """

    def __init__(
        self,
        base_paper_color: tuple[float, float, float] = (0.96, 0.95, 0.90),
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Kağıdın kuru haldeki RGB yansıtma katsayısı R_g
        self.r_paper_base = torch.tensor(base_paper_color, dtype=torch.float32, device=self.device).view(3, 1, 1)

    def render(
        self,
        pigment_layers: list,
        h_fluid: torch.Tensor,
        m_paper: torch.Tensor,
        z_paper: torch.Tensor,
        paper_tint_intensity: float = 0.12,
    ) -> torch.Tensor:
        """
        Tüm fiziksel katmanları Kubelka-Munk optik denklemleriyle nihai RGB görüntüsüne dönüştürür.
        Çıktı Shape: (H, W, 3) in [0, 255] float32.
        """
        h_dim, w_dim = z_paper.shape

        # -------------------------------------------------------------
        # 1. KAĞIT ALTLIĞI OPTİĞİ (R_paper)
        # Islak kağıt lifleri (m_paper yüksek) selüloz-su kırılma indisi eşleşmesi nedeniyle
        # daha koyu ve derin görünür (wet paper darkening).
        # -------------------------------------------------------------
        # Mikro lif topoğrafyası gölgelenmesi
        paper_topo = 1.0 - (z_paper.unsqueeze(0) / 0.3) * paper_tint_intensity
        # Lif ıslaklığı kararması (Sıvı ötesinde koyu halka oluşmaması için hafifletildi)
        moisture_darkening = 1.0 - torch.clamp(m_paper.unsqueeze(0), 0.0, 1.0) * 0.04
        r_substrate = torch.clamp(self.r_paper_base * paper_topo * moisture_darkening, 0.05, 1.0)

        # -------------------------------------------------------------
        # 2. TOPLAM KUBELKA-MUNK ABSORPSİYON (K) VE SAÇILMA (S) HESABI
        # Duncan Kuralı: K_total = sum(c_i * K_i), S_total = sum(c_i * S_i)
        # -------------------------------------------------------------
        k_total = torch.zeros((3, h_dim, w_dim), dtype=torch.float32, device=self.device)
        s_total = torch.zeros((3, h_dim, w_dim), dtype=torch.float32, device=self.device)
        total_thickness = torch.zeros((1, h_dim, w_dim), dtype=torch.float32, device=self.device)

        for lyr in pigment_layers:
            # Hem askıdaki pigment c hem de çökelen pigment d ışığı emer ve saçar
            pigment_conc = (lyr.c * 0.90 + lyr.d * 1.15).unsqueeze(0)  # Shape: (1, H, W)
            total_thickness += pigment_conc

            k_i = lyr.km_k.view(3, 1, 1)
            s_i = lyr.km_s.view(3, 1, 1)

            k_total += pigment_conc * k_i
            s_total += pigment_conc * s_i

        # Minimum saçılma sınırlayıcısı (Sıfıra bölmeyi engeller)
        s_safe = torch.clamp(s_total, min=1e-3)
        k_safe = torch.clamp(k_total, min=1e-4)

        # -------------------------------------------------------------
        # 3. KUBELKA-MUNK YANSITMA DENKLEMLERİ (R ve T)
        # -------------------------------------------------------------
        a = 1.0 + (k_safe / s_safe)
        b = torch.sqrt(torch.clamp(a ** 2 - 1.0, min=1e-6))

        # Optik kalınlık: b * S * x
        bsx = torch.clamp(b * s_safe * total_thickness, min=1e-4, max=18.0)

        sinh_bsx = torch.sinh(bsx)
        cosh_bsx = torch.cosh(bsx)

        denom = a * sinh_bsx + b * cosh_bsx
        r_0 = sinh_bsx / torch.clamp(denom, min=1e-5)
        t_0 = b / torch.clamp(denom, min=1e-5)

        # Kağıt altlığı (r_substrate) üzerindeki net yansıma
        # R = r_0 + (t_0^2 * r_substrate) / (1.0 - r_0 * r_substrate)
        sub_reflect_denom = torch.clamp(1.0 - r_0 * r_substrate, min=1e-4)
        r_composite = r_0 + (t_0 ** 2 * r_substrate) / sub_reflect_denom

        # C1 Kesintisiz Sub-Pixel Kağıt Altlığı Geçişi
        # Kubelka-Munk formülü kalınlık sıfırken (total_thickness = 0) analitik olarak
        # r_substrate'e eşittir. Bu geçiş, sıfıra yakın noktalardaki sayısal dalgalanmaları
        # pürüzsüzleştirerek radyatif transfer eğrisinin tüm ara yarı-tonlarını korur.
        blend_t = torch.clamp(total_thickness / 0.006, 0.0, 1.0)
        smooth_blend = blend_t * blend_t * (3.0 - 2.0 * blend_t)
        r_final = (1.0 - smooth_blend) * r_substrate + smooth_blend * r_composite

        # -------------------------------------------------------------
        # 4. YÜZEY MATLIĞI VE GERÇEKÇİ KAĞIT EMİLİMİ
        # Sulu boya kağıda emildiğinde 3D plastik parlama yapmaz;
        # renkler selüloz lifleriyle bütünleşerek derin ve mat bir hal alır.
        # -------------------------------------------------------------

        # (3, H, W) -> (H, W, 3) ve [0, 255] aralığına ölçekleme
        rgb_out = r_final.permute(1, 2, 0) * 255.0
        return torch.clamp(rgb_out, 0.0, 255.0)
