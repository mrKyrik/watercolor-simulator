import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .paper import PaperSubstrate
from .fluid import ShallowWaterSolver
from .pigment import PigmentManager
from .optics import KubelkaMunkRenderer
from .noise import fast_fractal_noise_gpu


class WatercolorEngine:
    """
    Multiscale Hybrid Watercolor Simulation Engine
    
    1. Macro Scale: Navier-Stokes Shallow Water, gravity, surface tension, capillary suction, and Kubelka-Munk
    2. Micro Scale (Phenomenological Sub-Grid):
       - Centripetal Catmull-Rom brush path and velocity dynamics
       - Multi-octave fractal edge deformation (torn and fibrous boundaries)
       - Bristle tracks and dry brush gaps
       - Sharp Gaussian edge ring (meniscus coffee ring)
       - Multi-scale body mottling / puddling and paper granulation
    """

    def __init__(
        self,
        width: int = 1000,
        height: int = 800,
        base_color: tuple[int, int, int] = (246, 245, 236),
        device: torch.device = None,
    ):
        self.width = width
        self.height = height
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_color = base_color

        # Physical Components
        self.paper = PaperSubstrate(
            width=width,
            height=height,
            scale=140.0,
            octaves=5,
            roughness=0.22,
            porosity=0.45,
            device=self.device,
        )
        self.fluid = ShallowWaterSolver(
            width=width,
            height=height,
            gravity=320.0,
            viscosity=0.35,
            surface_tension=20.0,
            friction_coeff=0.25,
            device=self.device,
        )
        self.pigments = PigmentManager(width=width, height=height, device=self.device)
        self.optics = KubelkaMunkRenderer(
            base_paper_color=(base_color[0] / 255.0, base_color[1] / 255.0, base_color[2] / 255.0),
            device=self.device,
        )

        # Multi-Layer Glaze Stack (Wet-on-Dry Stratified Glazes)
        self.dried_glazes: list[dict] = []

        y_ind, x_ind = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=torch.float32) + 0.5,
            torch.arange(width, device=self.device, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        self.grid_x = x_ind
        self.grid_y = y_ind

    def reset(self):
        """Resets the canvas and all physical states, including dried glazes."""
        self.fluid.h.zero_()
        self.fluid.u.zero_()
        self.fluid.v.zero_()
        self.paper.reset_moisture()
        self.pigments.reset()
        self.dried_glazes.clear()

    def dry_current_wash(self):
        """
        Commits all active fluid and pigment into a permanent, dried glaze layer.
        Enables multi-layer wet-on-dry watercolor glazing:
        - Active deposited pigments are locked into a permanent optical layer.
        - Fluid and paper moisture evaporate to dry state (h=0, m=0).
        - Subsequent brush strokes sit transparently over this layer without muddying.
        """
        if not self.pigments.layers:
            return

        h_dim, w_dim = self.height, self.width
        k_total = torch.zeros((3, h_dim, w_dim), dtype=torch.float32, device=self.device)
        s_total = torch.zeros((3, h_dim, w_dim), dtype=torch.float32, device=self.device)
        total_thickness = torch.zeros((1, h_dim, w_dim), dtype=torch.float32, device=self.device)

        has_pigment = False
        for lyr in self.pigments.layers:
            # When dried, all remaining suspended pigment (c) deposits onto fibers (d)
            conc = (lyr.c + lyr.d).unsqueeze(0)
            if float(conc.max().item()) > 1e-4:
                has_pigment = True
            total_thickness += conc
            k_total += conc * lyr.km_k.view(3, 1, 1)
            s_total += conc * lyr.km_s.view(3, 1, 1)

        if has_pigment:
            self.dried_glazes.append({
                "k": k_total,
                "s": s_total,
                "thickness": total_thickness,
            })

        # Evaporate active water and clear active layers for next wash
        self.fluid.h.zero_()
        self.fluid.u.zero_()
        self.fluid.v.zero_()
        self.paper.reset_moisture()
        self.pigments.layers.clear()

    # ------------------------------------------------------------------
    # Brush Stroke Dynamics (Multiscale Hybrid ROI)
    # ------------------------------------------------------------------

    @staticmethod
    def _catmull_rom_centripetal(points: list[tuple[float, float]], samples_per_seg: int = 8) -> list[tuple[float, float, float]]:
        """Centripetal Catmull-Rom spline interpolation and velocity profiling."""
        if len(points) < 2:
            return [(p[0], p[1], 1.0) for p in points]
        if len(points) == 2:
            p0, p1 = points[0], points[1]
            pts = []
            for t in np.linspace(0, 1, samples_per_seg):
                pts.append((p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]), 1.0))
            return pts

        extended = [points[0]] + list(points) + [points[-1]]
        dense_points = []
        alpha = 0.5

        def get_t(t_prev, p_a, p_b):
            dist = math.hypot(p_b[0] - p_a[0], p_b[1] - p_a[1])
            return t_prev + max(dist, 1e-4) ** alpha

        for i in range(1, len(extended) - 2):
            p0, p1, p2, p3 = extended[i - 1], extended[i], extended[i + 1], extended[i + 2]
            t0 = 0.0
            t1 = get_t(t0, p0, p1)
            t2 = get_t(t1, p1, p2)
            t3 = get_t(t2, p2, p3)

            seg_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            speed = max(0.5, min(seg_dist / 15.0, 3.0))

            for t in np.linspace(t1, t2, samples_per_seg, endpoint=False):
                a1 = ((t1 - t) * p0[0] + (t - t0) * p1[0]) / (t1 - t0), ((t1 - t) * p0[1] + (t - t0) * p1[1]) / (t1 - t0)
                a2 = ((t2 - t) * p1[0] + (t - t1) * p2[0]) / (t2 - t1), ((t2 - t) * p1[1] + (t - t1) * p2[1]) / (t2 - t1)
                a3 = ((t3 - t) * p2[0] + (t - t2) * p3[0]) / (t3 - t2), ((t3 - t) * p2[1] + (t - t2) * p3[1]) / (t3 - t2)

                b1 = ((t2 - t) * a1[0] + (t - t0) * a2[0]) / (t2 - t0), ((t2 - t) * a1[1] + (t - t0) * a2[1]) / (t2 - t0)
                b2 = ((t3 - t) * a2[0] + (t - t1) * a3[0]) / (t3 - t1), ((t3 - t) * a2[1] + (t - t1) * a3[1]) / (t3 - t1)

                c = ((t2 - t) * b1[0] + (t - t1) * b2[0]) / (t2 - t1), ((t2 - t) * b1[1] + (t - t1) * b2[1]) / (t2 - t1)
                dense_points.append((c[0], c[1], speed))

        dense_points.append((points[-1][0], points[-1][1], 1.0))
        return dense_points

    def add_stroke_from_points(
        self,
        points: list[tuple[float, float]],
        color_rgb: tuple[int, int, int],
        radius: float = 25.0,
        intensity: float = 0.85,
        edge_darkening: float = 0.65,
        dry_brush: float = 0.40,
        bristle_strength: float = 0.35,
        paper_interaction: float = 0.12,
        water_amount: float = 0.80,
        pigment_name: str = "Custom",
        granulation: float = 0.5,
        staining: float = 0.5,
    ):
        """
        Organic and rich brush stroke deposition:
        - Fractal distance field deformation
        - Sharp meniscus edge ring (Gaussian)
        - Multi-scale body mottling
        - Bristle tracks and dry brush scumbling
        - Shallow water and pigment deposit
        """
        if not points:
            return
        if len(points) == 1:
            self.add_droplet(points[0][0], points[0][1], radius, color_rgb, water_amount, intensity, pigment_name, granulation, staining)
            return

        curve = self._catmull_rom_centripetal(points, samples_per_seg=8)
        if len(curve) < 2:
            return

        xs = [p[0] for p in curve]
        ys = [p[1] for p in curve]
        max_r = radius * 2.2
        min_x = max(0, int(min(xs) - max_r - 12))
        max_x = min(self.width, int(max(xs) + max_r + 12))
        min_y = max(0, int(min(ys) - max_r - 12))
        max_y = min(self.height, int(max(ys) + max_r + 12))
        roi_w = max_x - min_x
        roi_h = max_y - min_y

        if roi_w <= 0 or roi_h <= 0:
            return

        # ROI Grid (+0.5 sub-pixel centering)
        roi_y = torch.arange(min_y, max_y, device=self.device, dtype=torch.float32).view(-1, 1) + 0.5
        roi_x = torch.arange(min_x, max_x, device=self.device, dtype=torch.float32).view(1, -1) + 0.5

        sub_phi = torch.full((roi_h, roi_w), 1e6, dtype=torch.float32, device=self.device)
        sub_speed = torch.zeros((roi_h, roi_w), dtype=torch.float32, device=self.device)

        # Line segment distance projection (GPU tensors)
        for i in range(len(curve) - 1):
            ax, ay, sp_a = curve[i]
            bx, by, sp_b = curve[i + 1]

            dx, dy = bx - ax, by - ay
            seg_len_sq = max(dx * dx + dy * dy, 1e-4)

            r_a = radius * max(0.85, min(1.15, 1.05 - 0.10 * (sp_a - 1.0)))
            r_b = radius * max(0.85, min(1.15, 1.05 - 0.10 * (sp_b - 1.0)))

            s = ((roi_x - ax) * dx + (roi_y - ay) * dy) / seg_len_sq
            s_clamped = torch.clamp(s, 0.0, 1.0)

            proj_x = ax + s_clamped * dx
            proj_y = ay + s_clamped * dy
            r_local = (1.0 - s_clamped) * r_a + s_clamped * r_b
            sp_local = (1.0 - s_clamped) * sp_a + s_clamped * sp_b

            dist = torch.sqrt((roi_x - proj_x) ** 2 + (roi_y - proj_y) ** 2)
            u_k = dist / torch.clamp(r_local, min=1.0)

            mask_update = (u_k < sub_phi)
            sub_phi = torch.where(mask_update, u_k, sub_phi)
            sub_speed = torch.where(mask_update, sp_local, sub_speed)

        # GPU Multi-Scale Fractal Noise Layers (Band limited - minimum wavelength >= 6px)
        f_shape = fast_fractal_noise_gpu(roi_w, roi_h, max(radius * 0.40, 8.0), 3, device=self.device)
        f_edge  = fast_fractal_noise_gpu(roi_w, roi_h, 14.0, 3, device=self.device)
        f_gran  = fast_fractal_noise_gpu(roi_w, roi_h, 12.0, 3, device=self.device)

        # 1. Edge fractal deformation and Analytical Sub-Pixel Antialiasing
        sub_phi = sub_phi * (1.0 + (f_shape - 0.5) * 0.25)
        # Capillary-sensitive and concentration-aware dynamic transition width:
        d_px = 7.0 * (1.0 + 0.25 * intensity)
        delta_phi = max(d_px / float(max(radius, 2.0)), 0.050)
        edge_falloff = torch.clamp((1.0 - sub_phi) / delta_phi, 0.0, 1.0)
        smooth_inside = edge_falloff * edge_falloff * (3.0 - 2.0 * edge_falloff)

        # Body Wash Kernel: Full interior coverage (homogeneous between 0.0 and 0.60)
        interior_ramp = torch.clamp((sub_phi - 0.60) / 0.38, 0.0, 1.0)
        wash_kernel = 1.0 - 0.25 * (interior_ramp ** 2)

        # 2. Paper fiber dry brush thresholding (Natural fiber scumbling)
        sub_texture = self.paper.texture_norm[min_y:max_y, min_x:max_x]
        dry_threshold = dry_brush * torch.clamp(sub_speed / 2.0, 0.0, 1.0) * 0.55
        dry_factor = torch.clamp((sub_texture - dry_threshold + 0.20) / 0.40, 0.0, 1.0)
        dry_mask = 0.20 + 0.80 * dry_factor

        # 3. Brush bristle factor
        bristle_factor = 1.0 - bristle_strength * 0.35 * torch.clamp(sub_speed / 1.5, 0.3, 1.0)

        # Ground Wetness Check (Wet-Awareness)
        # If ground is already wet, inner edge ring is suppressed and paint does not pin to fibers!
        existing_h = self.fluid.h[min_y:max_y, min_x:max_x]
        dry_boundary_mask = torch.clamp((0.10 - existing_h) / 0.08, 0.0, 1.0)
        is_dry_surface = torch.clamp((0.12 - existing_h) / 0.10, 0.0, 1.0)

        # 4. Single Monotonic Meniscus Edge Ring (with Colloidal Jamming Saturation)
        # Note: Only formed at the outer contact line touching dry paper!
        ring_pos = 0.94 + (f_edge - 0.5) * 0.04
        edge_ring = torch.exp(-((sub_phi - ring_pos) ** 2) / (2.0 * 0.070 ** 2))
        jammed_edge_darkening = edge_darkening / (1.0 + 0.30 * intensity)
        edge_alpha = edge_ring * jammed_edge_darkening * 0.75 * dry_boundary_mask

        # 5. Paper Fiber Topography Mottling
        conc = torch.clamp((0.5 - sub_texture) * 0.40 + f_gran * 0.45 + 0.35, 0.0, 1.0) ** 1.15
        core_alpha = wash_kernel * torch.clamp(conc * 1.05 + 0.15, 0.0, 1.0) * intensity
        # Balanced density: Interior remains smooth and even, while meniscus edge builds naturally
        alpha = torch.clamp(core_alpha + edge_alpha, 0.0, 1.5) * smooth_inside * dry_mask * bristle_factor

        # 6. Micro Fiber Granulation and Texture Bias (Only attaches to dry fibers)
        grain = (torch.rand((roi_h, roi_w), device=self.device) * 2.0 - 1.0)
        granulation_bias = grain * sub_texture * 0.05 * alpha * is_dry_surface
        texture_bias = (0.5 - sub_texture) * paper_interaction * alpha * is_dry_surface

        # Deposit into Fluid Layer (water height h for shallow water solver)
        self.fluid.h[min_y:max_y, min_x:max_x] += torch.clamp(alpha, 0.0, 1.0) * water_amount

        # Deposit into Pigment Layer
        col_norm = (color_rgb[0] / 255.0, color_rgb[1] / 255.0, color_rgb[2] / 255.0)
        layer_name = f"Pigment_{color_rgb[0]}_{color_rgb[1]}_{color_rgb[2]}" if pigment_name == "Custom" else pigment_name
        layer = self.pigments.get_or_create_layer(
            name=layer_name,
            color_rgb=col_norm,
            granulation=granulation,
            staining=staining,
        )
        
        # 7. Langmuir Surface Adsorption Saturation (Prevents artificial over-darkening)
        max_d = 2.6
        current_d = layer.d[min_y:max_y, min_x:max_x]
        capacity_factor = torch.clamp((max_d - current_d) / max_d, 0.15, 1.0)
        effective_deposit = alpha * intensity * 2.2 * capacity_factor

        # Fresh paint remains predominantly in fluid suspension (layer.c), allowing Deegan
        # advection and valley granulation during drying, with balanced initial fiber stain.
        initial_stain = max(0.08, 0.18 * layer.staining) * is_dry_surface
        layer.d[min_y:max_y, min_x:max_x] += effective_deposit * initial_stain + (granulation_bias + texture_bias)
        layer.c[min_y:max_y, min_x:max_x] += effective_deposit * (1.0 - initial_stain)

    def add_strokes_wash(
        self,
        strokes: list[dict],
        color_rgb: tuple[int, int, int] = None,
        water_amount: float = 0.85,
        pigment_name: str = "Custom",
        granulation: float = 0.5,
        staining: float = 0.5,
    ):
        """Processes multiple touching or parallel brush strokes as a SINGLE Unified Manifold fluid domain.
        All strokes' distance fields are merged into one potential: phi_unified = min_i(phi_i).
        Eliminates inner white seams between strokes by 100%, flowing the entire region like a single stroke."""
        if not strokes:
            return

        all_pts = []
        max_r = 10.0
        for st in strokes:
            pts = st.get("points", [])
            if pts:
                all_pts.extend(pts)
                max_r = max(max_r, float(st.get("radius", 25.0)))

        if not all_pts:
            return

        all_xs = [p[0] for p in all_pts]
        all_ys = [p[1] for p in all_pts]
        pad = int(max_r * 2.5) + 15
        min_x = max(0, int(min(all_xs)) - pad)
        max_x = min(self.width, int(max(all_xs)) + pad)
        min_y = max(0, int(min(all_ys)) - pad)
        max_y = min(self.height, int(max(all_ys)) + pad)

        roi_w = max_x - min_x
        roi_h = max_y - min_y
        if roi_w <= 0 or roi_h <= 0:
            return

        roi_x = self.grid_x[min_y:max_y, min_x:max_x]
        roi_y = self.grid_y[min_y:max_y, min_x:max_x]

        # Single unified distance field (phi_unified)
        phi_unified = torch.full((roi_h, roi_w), 1e6, dtype=torch.float32, device=self.device)
        speed_unified = torch.zeros((roi_h, roi_w), dtype=torch.float32, device=self.device)

        avg_intensity = 0.0
        avg_edge_darkening = 0.0
        avg_dry_brush = 0.0
        avg_bristle = 0.0
        avg_pap_ix = 0.0

        stroke_u_fields = []
        for st in strokes:
            u_stroke_min = torch.full((roi_h, roi_w), 1e6, dtype=torch.float32, device=self.device)
            pts = st.get("points", [])
            r_base = float(st.get("radius", 25.0))
            avg_intensity += float(st.get("intensity", 0.85))
            avg_edge_darkening += float(st.get("edge_darkening", 0.60))
            avg_dry_brush += float(st.get("dry_brush", 0.35))
            avg_bristle += float(st.get("bristle_strength", 0.35))
            avg_pap_ix += float(st.get("paper_interaction", 0.12))

            if len(pts) < 2:
                if len(pts) == 1:
                    cx, cy = pts[0]
                    dist = torch.sqrt((roi_x - cx) ** 2 + (roi_y - cy) ** 2)
                    u_k = dist / max(r_base, 1.0)
                    u_stroke_min = torch.minimum(u_stroke_min, u_k)
                    phi_unified = torch.minimum(phi_unified, u_k)
                stroke_u_fields.append(u_stroke_min)
                continue

            curve = self._catmull_rom_centripetal(pts, samples_per_seg=8)
            for i in range(len(curve) - 1):
                ax, ay, sp_a = curve[i]
                bx, by, sp_b = curve[i + 1]

                dx, dy = bx - ax, by - ay
                seg_len_sq = max(dx * dx + dy * dy, 1e-4)

                s = ((roi_x - ax) * dx + (roi_y - ay) * dy) / seg_len_sq
                s_clamped = torch.clamp(s, 0.0, 1.0)

                proj_x = ax + s_clamped * dx
                proj_y = ay + s_clamped * dy

                sp_local = sp_a + s_clamped * (sp_b - sp_a)
                r_local = r_base * max(0.85, min(1.15, 1.05 - 0.10 * (float(sp_a) - 1.0)))

                dist = torch.sqrt((roi_x - proj_x) ** 2 + (roi_y - proj_y) ** 2)
                u_k = dist / max(r_local, 1.0)

                u_stroke_min = torch.minimum(u_stroke_min, u_k)
                mask_up = (u_k < phi_unified)
                phi_unified = torch.where(mask_up, u_k, phi_unified)
                speed_unified = torch.where(mask_up, sp_local, speed_unified)

            stroke_u_fields.append(u_stroke_min)

        n_s = max(len(strokes), 1)
        avg_intensity /= n_s
        avg_edge_darkening /= n_s
        avg_dry_brush /= n_s
        avg_bristle /= n_s
        avg_pap_ix /= n_s

        # APPLY FRACTAL NOISE AND CONTINUOUS MENISCUS ON UNIFIED FIELD
        f_shape = fast_fractal_noise_gpu(roi_w, roi_h, max(max_r * 0.40, 8.0), 3, device=self.device)
        f_edge = fast_fractal_noise_gpu(roi_w, roi_h, 14.0, 3, device=self.device)
        f_gran = fast_fractal_noise_gpu(roi_w, roi_h, 12.0, 3, device=self.device)

        phi_deformed = phi_unified * (1.0 + (f_shape - 0.5) * 0.25)
        # Capillary-sensitive dynamic transition width:
        d_px = 7.0 * (1.0 + 0.25 * avg_intensity)
        delta_phi = max(d_px / float(max(max_r, 2.0)), 0.050)
        edge_falloff = torch.clamp((1.0 - phi_deformed) / delta_phi, 0.0, 1.0)
        smooth_inside = edge_falloff * edge_falloff * (3.0 - 2.0 * edge_falloff)

        # Body Wash Kernel (Eliminates interior seams)
        interior_ramp = torch.clamp((phi_deformed - 0.60) / 0.38, 0.0, 1.0)
        wash_kernel = 1.0 - 0.25 * (interior_ramp ** 2)

        # Paper fiber dry brush
        sub_texture = self.paper.texture_norm[min_y:max_y, min_x:max_x]
        dry_threshold = avg_dry_brush * torch.clamp(speed_unified / 2.0, 0.0, 1.0) * 0.55
        dry_factor = torch.clamp((sub_texture - dry_threshold + 0.20) / 0.40, 0.0, 1.0)
        dry_mask = 0.20 + 0.80 * dry_factor

        bristle_factor = 1.0 - avg_bristle * 0.35 * torch.clamp(speed_unified / 1.5, 0.3, 1.0)

        # SINGLE OUTER PERIMETER RING (with Colloidal Jamming Saturation)
        ring_pos = 0.94 + (f_edge - 0.5) * 0.04
        edge_ring = torch.exp(-((phi_deformed - ring_pos) ** 2) / (2.0 * 0.070 ** 2))
        jammed_edge_darkening = avg_edge_darkening / (1.0 + 0.50 * avg_intensity)
        edge_alpha = edge_ring * jammed_edge_darkening * 0.60

        conc = torch.clamp((0.5 - sub_texture) * 0.40 + f_gran * 0.45 + 0.35, 0.0, 1.0) ** 1.15
        core_alpha = wash_kernel * torch.clamp(conc * 1.30 + 0.20, 0.0, 1.0) * avg_intensity
        alpha = torch.clamp(core_alpha + edge_alpha, 0.0, 1.0) * smooth_inside * dry_mask * bristle_factor

        grain = (torch.rand((roi_h, roi_w), device=self.device) * 2.0 - 1.0)
        granulation_bias = grain * sub_texture * 0.05 * alpha
        texture_bias = (0.5 - sub_texture) * avg_pap_ix * alpha

        # LOAD AS A SINGLE FLUID POOL
        self.fluid.h[min_y:max_y, min_x:max_x] += alpha * water_amount

        # MULTI-COLOR INTEGRATION (Shepard / Smooth Partition of Unity for Graded Wash)
        # Each stroke carries its color, forming a seamless, step-free gradient!
        sigma_c = 0.85
        weights = []
        for u_s in stroke_u_fields:
            w_s = torch.exp(- (u_s ** 2) / (2.0 * (sigma_c ** 2))) + 1e-4
            weights.append(w_s)

        sum_w = torch.clamp(sum(weights), min=1e-5)
        normalized_weights = [w / sum_w for w in weights]

        # Smooth pigment loading for each color:
        for k, st in enumerate(strokes):
            col = st.get("color", color_rgb or (40, 100, 200))
            col_norm = (col[0] / 255.0, col[1] / 255.0, col[2] / 255.0)
            p_name = st.get("name", f"Pigment_{col[0]}_{col[1]}_{col[2]}")
            layer = self.pigments.get_or_create_layer(
                name=p_name,
                color_rgb=col_norm,
                granulation=st.get("granulation", granulation),
                staining=st.get("staining", staining),
            )
            w_share = normalized_weights[k]

            max_d = 2.6
            current_d = layer.d[min_y:max_y, min_x:max_x]
            capacity_factor = torch.clamp((max_d - current_d) / max_d, 0.15, 1.0)

            # Smooth share assigned to this color layer:
            dep_k = alpha * avg_intensity * 2.5 * capacity_factor * w_share

            initial_stain = max(0.40, 0.50 * layer.staining)
            layer.d[min_y:max_y, min_x:max_x] += dep_k * initial_stain + (granulation_bias + texture_bias) * w_share
            layer.c[min_y:max_y, min_x:max_x] += dep_k * (1.0 - initial_stain)

    # ------------------------------------------------------------------
    # Unified Drops & Clusters
    # ------------------------------------------------------------------

    def add_pigment_cluster(self, drops: list[dict]):
        """Deposits a unified fluid drop cluster onto canvas with organic fractal profile."""
        if not drops:
            return

        phi_unified = torch.full((self.height, self.width), 1e6, dtype=torch.float32, device=self.device)
        weight_sum = torch.zeros((self.height, self.width), dtype=torch.float32, device=self.device)

        avg_intensity = 0.0
        avg_edge_darkening = 0.0
        avg_feathering = 0.0
        avg_paper_interaction = 0.0

        for d in drops:
            cx, cy = d["x"], d["y"]
            r = max(d.get("radius", 50), 1.0)
            col = d.get("color", (50, 100, 200))
            inten = d.get("intensity", 0.8)
            edge_dk = d.get("edge_darkening", 0.6)
            feath = d.get("feathering", 0.4)
            pap_ix = d.get("paper_interaction", 0.12)

            avg_intensity += inten
            avg_edge_darkening += edge_dk
            avg_feathering += feath
            avg_paper_interaction += pap_ix

            dist = torch.sqrt((self.grid_x - cx) ** 2 + (self.grid_y - cy) ** 2)
            noise = fast_fractal_noise_gpu(self.width, self.height, max(r * 0.45, 1.0), octaves=3, device=self.device)
            u_k = (dist * (1.0 + (noise - 0.5) * 0.45)) / r

            phi_unified = torch.minimum(phi_unified, u_k)
            w_k = torch.exp(-0.5 * (torch.clamp(u_k / 0.65, 0.0, 3.0) ** 2)) * (u_k < 1.05).float()
            weight_sum += w_k

        n_drops = len(drops)
        avg_intensity /= n_drops
        avg_edge_darkening /= n_drops
        avg_feathering /= n_drops
        avg_paper_interaction /= n_drops
        avg_radius = sum(float(d.get("radius", 40.0)) for d in drops) / max(n_drops, 1)

        # Sub-Pixel C1 Antialiasing (Eliminates pixelation)
        d_px = 7.0 * (1.0 + 0.25 * avg_intensity)
        delta_phi = max(d_px / float(max(avg_radius, 2.0)), 0.050)
        edge_falloff = torch.clamp((1.0 - phi_unified) / delta_phi, 0.0, 1.0)
        smooth_inside = edge_falloff * edge_falloff * (3.0 - 2.0 * edge_falloff)

        # Body Wash Kernel (Continuous, balanced interior)
        wash_kernel = torch.clamp(0.85 - 0.20 * (phi_unified ** 2), 0.0, 1.0)

        # Paper Fiber Topography and Granulation Body Texture
        c_gran = fast_fractal_noise_gpu(self.width, self.height, 12.0, 3, device=self.device)
        pigment_conc = torch.clamp((0.5 - self.paper.texture_norm) * 0.40 + c_gran * 0.45 + 0.35, 0.0, 1.0) ** 1.15

        core_alpha = wash_kernel * torch.clamp(pigment_conc * 1.05 + 0.15, 0.0, 1.0) * avg_intensity

        # Single Monotonic Meniscus Edge Ring (with Colloidal Jamming Saturation)
        edge_noise = fast_fractal_noise_gpu(self.width, self.height, 14.0, 3, device=self.device)
        ring_pos = 0.94 + (edge_noise - 0.5) * 0.04
        edge_ring = torch.exp(-((phi_unified - ring_pos) ** 2) / (2.0 * 0.070 ** 2))
        jammed_edge_darkening = avg_edge_darkening / (1.0 + 0.30 * avg_intensity)
        edge_alpha = edge_ring * jammed_edge_darkening * 0.75

        vein_noise = (fast_fractal_noise_gpu(self.width, self.height, 14.0, 3, device=self.device) - 0.5) * avg_feathering * 0.15 * wash_kernel
        # Balanced density: allows natural meniscus concentration above the body level
        alpha = torch.clamp(core_alpha + edge_alpha + vein_noise, 0.0, 1.5) * smooth_inside

        grain = (torch.rand((self.height, self.width), device=self.device) * 2.0 - 1.0)
        granulation_bias = grain * self.paper.texture_norm * 0.05
        texture_bias = (0.5 - self.paper.texture_norm) * avg_paper_interaction

        # Load fluid and pigment
        self.fluid.h += torch.clamp(alpha, 0.0, 1.0) * 0.85
        safe_weights = torch.clamp(weight_sum, min=1e-6)

        for d in drops:
            col = d.get("color", (50, 100, 200))
            col_norm = (col[0] / 255.0, col[1] / 255.0, col[2] / 255.0)
            p_name = d.get("name", f"Drop_{col[0]}_{col[1]}_{col[2]}")
            layer = self.pigments.get_or_create_layer(
                name=p_name,
                color_rgb=col_norm,
                granulation=d.get("granulation", 0.5),
                staining=d.get("staining", 0.5),
            )
            # Proportional distribution across multiple drops
            dist_d = torch.sqrt((self.grid_x - d["x"]) ** 2 + (self.grid_y - d["y"]) ** 2)
            u_d = dist_d / max(d.get("radius", 50), 1.0)
            w_d = torch.exp(-0.5 * (torch.clamp(u_d / 0.65, 0.0, 3.0) ** 2)) * (u_d < 1.05).float()
            drop_share = (w_d / safe_weights) if len(drops) > 1 else torch.ones_like(alpha)

            # Langmuir Saturation Controlled Pigment Deposition
            max_d = 2.6
            capacity_factor = torch.clamp((max_d - layer.d) / max_d, 0.15, 1.0)
            deposit_weight = alpha * float(d.get("intensity", 1.0)) * drop_share * 2.2 * capacity_factor

            # In wet drops, initial stain is controlled so advection & Deegan drying can concentrate rim
            initial_stain = max(0.08, 0.18 * layer.staining)
            layer.d += deposit_weight * initial_stain + (granulation_bias + texture_bias) * alpha
            layer.c += deposit_weight * (1.0 - initial_stain)

    def add_droplet(
        self,
        cx: float,
        cy: float,
        radius: float,
        color_rgb: tuple[int, int, int],
        water_amount: float = 0.85,
        pigment_concentration: float = 1.30,
        pigment_name: str = "Custom",
        granulation: float = 0.5,
        staining: float = 0.5,
    ):
        """Deposits a single fluid droplet."""
        self.add_pigment_cluster([{
            "x": cx, "y": cy,
            "radius": radius,
            "color": color_rgb,
            "intensity": pigment_concentration,
            "edge_darkening": 0.65,
            "feathering": 0.35,
            "paper_interaction": 0.12,
            "granulation": granulation,
            "staining": staining,
            "name": pigment_name,
        }])

    def add_splatter(
        self,
        cx: float,
        cy: float,
        radius: float,
        color_rgb: tuple[int, int, int],
        count: int = 14,
        scatter_min: float = 1.2,
        scatter_max: float = 2.4,
        intensity: float = 0.80,
        pigment_name: str = "Custom",
        seed: int = None,
    ):
        """Organic splatter droplets."""
        rng = np.random.default_rng(seed)
        drops = []
        for _ in range(count):
            angle = rng.uniform(0.0, 2.0 * math.pi)
            dist = rng.uniform(scatter_min, scatter_max) * radius
            dx = cx + dist * math.cos(angle)
            dy = cy + dist * math.sin(angle)
            dr = rng.uniform(3.0, 9.5)
            if 0 <= dx < self.width and 0 <= dy < self.height:
                drops.append({
                    "x": dx, "y": dy,
                    "radius": dr,
                    "color": color_rgb,
                    "intensity": intensity * rng.uniform(0.65, 1.0),
                    "edge_darkening": 0.70,
                    "feathering": 0.20,
                    "paper_interaction": 0.12,
                })
        if drops:
            self.add_pigment_cluster(drops)

    # ------------------------------------------------------------------
    # Physical Simulation Loop (Macro Solver)
    # ------------------------------------------------------------------

    def step(
        self,
        dt: float = 0.05,
        gravity_strength: float = 0.0,
        gravity_angle_deg: float = 90.0,
    ):
        """
        Macro physical simulation step (GPU):
        - Shallow water pressure gradient and flow
        - Capillary coalescence
        - Evaporation and fiber saturation absorption
        """
        rad = math.radians(gravity_angle_deg)
        f_ext_x = gravity_strength * math.cos(rad) * 35.0
        f_ext_y = gravity_strength * math.sin(rad) * 35.0

        # Capillary Absorption
        absorbed_flux = self.paper.absorb_from_fluid(self.fluid.h, dt)
        self.paper.diffuse_fiber_moisture(dt)

        # Shallow Water Solution
        self.fluid.step(
            dt=dt,
            z_paper=self.paper.z_paper,
            absorbed_flux=absorbed_flux,
            pinning_mask=self.paper.pinning,
            external_force_x=f_ext_x,
            external_force_y=f_ext_y,
        )

        # Pigment Advection
        scale_x = 2.0 / max(self.width, 1)
        scale_y = 2.0 / max(self.height, 1)
        disp_x = self.fluid.u * dt * scale_x
        disp_y = self.fluid.v * dt * scale_y
        disp = torch.stack([disp_x, disp_y], dim=-1).unsqueeze(0)
        sample_coords = torch.clamp(self.fluid.base_grid - disp, -1.0, 1.0)

        self.pigments.step(
            dt=dt,
            u=self.fluid.u,
            v=self.fluid.v,
            h=self.fluid.h,
            z_paper=self.paper.z_paper,
            pinning_mask=self.paper.pinning,
            sample_coords=sample_coords,
        )

    def run_simulation(
        self,
        steps: int = 20,
        dt: float = 0.05,
        gravity_strength: float = 0.0,
        gravity_angle_deg: float = 90.0,
    ):
        for _ in range(steps):
            self.step(
                dt=dt,
                gravity_strength=gravity_strength,
                gravity_angle_deg=gravity_angle_deg,
            )

    # ------------------------------------------------------------------
    # Render & Export
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray:
        """Renders GPU canvas using Kubelka-Munk radiative transfer equations."""
        with torch.no_grad():
            rgb_tensor = self.optics.render(
                pigment_layers=self.pigments.layers,
                h_fluid=self.fluid.h,
                m_paper=self.paper.m,
                z_paper=self.paper.z_paper,
                dried_glazes=self.dried_glazes,
            )
            arr = rgb_tensor.cpu().numpy().astype(np.uint8)
            return arr

    def save_image(self, filename: str):
        arr = self.render()
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, bgr)
