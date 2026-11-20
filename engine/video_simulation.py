"""
engine/video_simulation.py - Flowing Watercolor Video Simulation Engine.

Specialized extension of WatercolorEngine for temporal video sequences:
1. Dense Optical Flow (OpenCV cv2.DISOpticalFlow, FAST preset) on normalized silhouette masks.
2. PyTorch Semi-Lagrangian GPU state advection (F.grid_sample) for persistent fluid height
   and dual-phase pigment concentrations (c, d) without temporal boiling.
3. Strictly screen-space static paper substrate topography (z_paper and pinning invariance).
4. Scene-cut & transition detection (mask IoU < 0.15 or polarity flip) with soft state reset / wash drying.
5. EDT-weighted incremental droplet infilling on newly uncovered pixels (Delta M_new = M_t \\ M_{t-1}).
6. Continuous 3D harmonic Perlin noise color selection over historical fine-art pigment triads
   (Prussian Blue, Burnt Sienna, Quinacridone Rose, Gamboge Yellow, Viridian Green)
   mixed subtractively via Kubelka-Munk Duncan optics.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from engine.simulation import WatercolorEngine
from engine.pigment import PigmentManager, PigmentLayer
from engine.optics import KubelkaMunkRenderer
from engine.noise import fast_fractal_noise_gpu

logger = logging.getLogger("engine.video_simulation")


class TransitionMode(str, Enum):
    """Behavior applied when a scene cut or polarity flip is detected."""
    SOFT_RESET = "soft_reset"      # Zero velocities, clear active suspended & deposited pigments
    WASH_DRYING = "wash_drying"    # Commit departing state to permanent glaze via dry_current_wash()
    FULL_RESET = "full_reset"      # Complete canvas reset including past dried glazes
    NONE = "none"                  # Force optical flow advection regardless of IoU


# Calibrated spectrophotometric data for classical fine-art watercolor pigments
FINE_ART_PIGMENTS: Dict[str, Dict[str, Any]] = {
    "Prussian Blue": {
        "color_rgb": (0, 49, 83),
        "color": (0.00, 0.19, 0.33),
        "k": (0.95, 0.04, 0.01),      # Strong red absorption, deep cyan/blue transmission
        "s": (0.05, 0.70, 0.95),
        "granulation": 0.10,          # Smooth wash, low mineral settling
        "staining": 0.88,             # High staining (bonds strongly to cellulose fibers)
        "diffusivity": 0.12,          # Rapid capillary spread in wet pools
        "role": "Luminous core shadow, cool depth, structural foundation",
    },
    "Burnt Sienna": {
        "color_rgb": (160, 82, 45),
        "color": (0.63, 0.32, 0.18),
        "k": (0.45, 0.75, 0.90),      # Earthy warm iron oxide profile
        "s": (0.50, 0.25, 0.10),
        "granulation": 0.85,          # Heavy mineral valley sedimentation
        "staining": 0.40,             # Low staining (easily re-lifted by fluid motion)
        "diffusivity": 0.05,          # Slow particulate diffusion
        "role": "Mottled organic warmth, paper texture articulation, shadow neutralizer",
    },
    "Quinacridone Rose": {
        "color_rgb": (213, 48, 100),
        "color": (0.84, 0.19, 0.39),
        "k": (0.04, 0.96, 0.20),      # Deep green absorption (brilliant magenta)
        "s": (0.85, 0.10, 0.50),
        "granulation": 0.12,          # Highly transparent dye-like wash
        "staining": 0.80,             # Strong staining
        "diffusivity": 0.10,          # High mobility
        "role": "Vibrant accent highlights, dress folds, floral warm-cool bridge",
    },
    "Gamboge Yellow": {
        "color_rgb": (228, 155, 15),
        "color": (0.89, 0.61, 0.06),
        "k": (0.01, 0.04, 0.98),      # Deep blue absorption (radiant golden yellow)
        "s": (0.95, 0.88, 0.05),
        "granulation": 0.25,          # Moderate fiber interaction
        "staining": 0.55,             # Medium staining
        "diffusivity": 0.10,          # Active capillary flow
        "role": "Radiant luminous glow, apple core highlights, emerald mixer",
    },
    "Viridian Green": {
        "color_rgb": (18, 130, 88),
        "color": (0.07, 0.51, 0.35),
        "k": (0.85, 0.08, 0.75),      # Red and blue absorption (clear emerald)
        "s": (0.10, 0.85, 0.20),
        "granulation": 0.75,          # Mineral granulation in paper hollows
        "staining": 0.45,             # Moderate staining
        "diffusivity": 0.06,          # Sluggish mineral transport
        "role": "Botanical atmosphere, misty foliage, deep teal transitions",
    },
}

# Register fine-art pigments into PigmentManager reference dictionary
PigmentManager.ARTIST_PIGMENTS.update(FINE_ART_PIGMENTS)


@dataclass
class DropletSpec:
    """Specification of an individual procedural watercolor droplet."""
    x: float
    y: float
    radius: float
    color_rgb: Tuple[int, int, int]
    pigment_name: str
    intensity: float = 0.85
    water_amount: float = 0.80
    edge_darkening: float = 0.65
    granulation: float = 0.50
    staining: float = 0.50


class PerlinNoise3D:
    """
    Self-contained, deterministic 3D Classic Perlin Noise generator.
    Implements Ken Perlin's quintic polynomial easing for smooth spatial-temporal fields.
    """

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.Generator(np.random.PCG64(seed))
        p = np.arange(256, dtype=int)
        rng.shuffle(p)
        self.perm = np.tile(p, 2)
        # 12 canonical 3D gradient vectors
        self.gradients = np.array([
            [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
            [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
            [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        ], dtype=float)

    @staticmethod
    def _fade(t: float) -> float:
        # Quintic polynomial: 6t^5 - 15t^4 + 10t^3
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    def _grad(self, hash_val: int, x: float, y: float, z: float) -> float:
        g = self.gradients[hash_val % 12]
        return g[0] * x + g[1] * y + g[2] * z

    def sample(self, x: float, y: float, z: float) -> float:
        """Samples scalar noise in [-1.0, 1.0] at continuous coordinate (x, y, z)."""
        xi = int(math.floor(x)) & 255
        yi = int(math.floor(y)) & 255
        zi = int(math.floor(z)) & 255

        xf = x - math.floor(x)
        yf = y - math.floor(y)
        zf = z - math.floor(z)

        u = self._fade(xf)
        v = self._fade(yf)
        w = self._fade(zf)

        p = self.perm
        aaa = p[p[p[xi] + yi] + zi]
        aba = p[p[p[xi] + yi + 1] + zi]
        aab = p[p[p[xi] + yi] + zi + 1]
        abb = p[p[p[xi] + yi + 1] + zi + 1]
        baa = p[p[p[xi + 1] + yi] + zi]
        bba = p[p[p[xi + 1] + yi + 1] + zi]
        bab = p[p[p[xi + 1] + yi] + zi + 1]
        bbb = p[p[p[xi + 1] + yi + 1] + zi + 1]

        x1 = self._lerp(self._grad(aaa, xf, yf, zf), self._grad(baa, xf - 1, yf, zf), u)
        x2 = self._lerp(self._grad(aba, xf, yf - 1, zf), self._grad(bba, xf - 1, yf - 1, zf), u)
        y1 = self._lerp(x1, x2, v)

        x3 = self._lerp(self._grad(aab, xf, yf, zf - 1), self._grad(bab, xf - 1, yf, zf - 1), u)
        x4 = self._lerp(self._grad(abb, xf, yf - 1, zf - 1), self._grad(bbb, xf - 1, yf - 1, zf - 1), u)
        y2 = self._lerp(x3, x4, v)

        return float(self._lerp(y1, y2, w))


class HarmonicPalette3D:
    """
    3D Continuous Harmonic Perlin Color Field Selector.
    
    Traverses fine-art pigment triads smoothly based on spatial position (x, y)
    and frame index t, ensuring natural, fluid color transitions across animated silhouettes.
    """

    def __init__(
        self,
        scale_x: float = 400.0,
        scale_y: float = 350.0,
        scale_t: float = 45.0,
        seed: int = 42,
    ) -> None:
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_t = scale_t
        self.noise1 = PerlinNoise3D(seed=seed)
        self.noise2 = PerlinNoise3D(seed=seed + 1013)

        # 5 canonical pigments along circular sequence
        self.pigment_names = [
            "Gamboge Yellow",
            "Quinacridone Rose",
            "Prussian Blue",
            "Viridian Green",
            "Burnt Sienna",
        ]

    def sample_pigment_at(
        self,
        x: float,
        y: float,
        frame_idx: int,
        distance_norm: float = 0.5,
    ) -> Tuple[Tuple[int, int, int], str, float, float]:
        """
        Samples a fine-art pigment at continuous coordinates (x, y, frame_idx).
        
        Args:
            x: X-coordinate on canvas.
            y: Y-coordinate on canvas.
            frame_idx: Frame number (controls temporal evolution).
            distance_norm: Normalized distance transform depth in [0.0, 1.0].
            
        Returns:
            Tuple of (color_rgb, pigment_name, granulation, staining).
        """
        nx = x / self.scale_x
        ny = y / self.scale_y
        nt = frame_idx / self.scale_t

        val1 = self.noise1.sample(nx, ny, nt)
        val2 = self.noise2.sample(nx + 17.31, ny + 31.79, nt + 43.15)

        # Compute continuous harmonic phase angle phi in [0.0, 1.0)
        phi_angle = math.atan2(val2, val1)  # [-pi, pi]
        phi = (phi_angle + math.pi) / (2.0 * math.pi)
        phi = phi % 1.0

        # Radial depth modulation:
        # Near contour boundary (distance_norm < 0.20): bias toward granulating mineral pigments
        # Deep interior (distance_norm > 0.60): bias toward staining transparent dyes
        if distance_norm < 0.20 and (val1 > 0.0):
            # Mineral valley settling near borders
            chosen_name = "Burnt Sienna" if val2 >= 0 else "Viridian Green"
        elif distance_norm > 0.65 and (val2 > 0.0):
            # Luminous high-transparency core
            chosen_name = "Quinacridone Rose" if val1 >= 0 else "Prussian Blue"
        else:
            # Standard harmonic circular progression across the 5 pigments
            seg = phi * 5.0
            idx = int(math.floor(seg)) % 5
            chosen_name = self.pigment_names[idx]

        spec = FINE_ART_PIGMENTS[chosen_name]
        return (
            spec["color_rgb"],
            chosen_name,
            float(spec["granulation"]),
            float(spec["staining"]),
        )


@dataclass
class VideoSimulationConfig:
    """Configuration hyperparameters for video watercolor simulation."""
    # Optical Flow Configuration
    flow_preset: int = cv2.DISOPTICAL_FLOW_PRESET_FAST  # FAST=1, ULTRAFAST=0, MEDIUM=2
    flow_max_magnitude: float = 120.0                   # Max displacement clamp (px)
    
    # State Advection (F.grid_sample)
    advection_mode: str = "bilinear"                    # 'bilinear' or 'nearest'
    advection_padding: str = "zeros"                    # 'zeros' eliminates canvas border streaking
    align_corners: bool = True                          # Matches fluid.py coordinate standard
    
    # Motion Momentum Coupling
    momentum_coupling_beta: float = 0.25                # Motion coupling factor [0.0, 0.5]
    momentum_fps_scale: float = 15.0                    # px/frame to px/sec scale multiplier
    
    # Boundary & Wake Dynamics
    wake_decay_rate: float = 0.40                       # Deposited pigment wake retention [0.0, 0.5]
    evaporate_wake_fluid: bool = True                   # Zero fluid outside current silhouette
    
    # Transition & Scene Cut Detection
    iou_cut_threshold: float = 0.15                     # IoU below this triggers cut handling
    transition_mode: TransitionMode = TransitionMode.SOFT_RESET
    
    # Infilling & Simulation Steps
    sim_substeps: int = 6                               # Physical Navier-Stokes relaxation substeps
    sim_dt: float = 0.040                               # Time delta per physical relaxation step
    edt_scale_cr: float = 0.75                          # Droplet radius scale: R = clamp(cr * D, min_r, max_r)
    droplet_r_min: float = 4.0                          # Min droplet radius (px)
    droplet_r_max: float = 50.0                         # Max droplet radius (px)
    infill_starvation_threshold: float = 0.15           # Refill thinned interior areas


class VideoWatercolorEngine(WatercolorEngine):
    """
    Temporal Watercolor Video Simulation Engine extending WatercolorEngine.
    
    Maintains persistent fluid and pigment state across video frames using
    DIS Optical Flow and Semi-Lagrangian GPU advection, while anchoring the
    paper substrate topography strictly static in screen space.
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 384,
        config: Optional[VideoSimulationConfig] = None,
        base_color: Tuple[int, int, int] = (246, 245, 236),
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(width=width, height=height, base_color=base_color, device=device)
        self.config = config or VideoSimulationConfig()

        # Update hyperparameters from optional kwargs for pipeline compatibility
        if "edt_scale_cr" in kwargs:
            self.config.edt_scale_cr = float(kwargs["edt_scale_cr"])
        if "droplet_r_min" in kwargs:
            self.config.droplet_r_min = float(kwargs["droplet_r_min"])
        if "droplet_r_max" in kwargs:
            self.config.droplet_r_max = float(kwargs["droplet_r_max"])
        if "wake_decay_rate" in kwargs:
            self.config.wake_decay_rate = float(kwargs["wake_decay_rate"])
        if "sim_substeps" in kwargs:
            self.config.sim_substeps = int(kwargs["sim_substeps"])
        if "sim_dt" in kwargs:
            self.config.sim_dt = float(kwargs["sim_dt"])

        # 3D Harmonic Perlin color selector
        palette_scale_t = float(kwargs.get("palette_scale_t", 45.0))
        self.palette = HarmonicPalette3D(scale_t=palette_scale_t)

        # Register fine-art pigments
        self.pigments.ARTIST_PIGMENTS.update(FINE_ART_PIGMENTS)

        # Instantiate OpenCV DIS Optical Flow
        self.dis_flow = cv2.DISOpticalFlow.create(self.config.flow_preset)

        # Precompute normalized base grid for F.grid_sample: [-1.0, 1.0] x [-1.0, 1.0]
        y_coords = torch.linspace(-1.0, 1.0, height, device=self.device, dtype=torch.float32)
        x_coords = torch.linspace(-1.0, 1.0, width, device=self.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        self.base_norm_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # Shape: (1, H, W, 2)

        # Coordinate scale factors (align_corners=True)
        self.scale_x = 2.0 / max(width - 1.0, 1.0)
        self.scale_y = 2.0 / max(height - 1.0, 1.0)

        # Temporal sequence caching
        self.prev_mask: Optional[np.ndarray] = None
        self.frame_idx: int = 0
        self.rng = np.random.Generator(np.random.PCG64(12345))

    # ------------------------------------------------------------------
    # 1. Optical Flow & Motion Tracking
    # ------------------------------------------------------------------

    def compute_optical_flow(self, prev_mask: np.ndarray, curr_mask: np.ndarray) -> np.ndarray:
        """
        Calculates dense forward optical flow from prev_mask to curr_mask via cv2.DISOpticalFlow.
        
        Args:
            prev_mask: (H, W) uint8 binary mask from frame t-1.
            curr_mask: (H, W) uint8 binary mask from frame t.
            
        Returns:
            (H, W, 2) float32 displacement field array [u, v].
        """
        if prev_mask.shape != curr_mask.shape:
            raise ValueError(f"Mask shape mismatch: {prev_mask.shape} vs {curr_mask.shape}")

        p_img = prev_mask if prev_mask.dtype == np.uint8 else (prev_mask > 0).astype(np.uint8) * 255
        c_img = curr_mask if curr_mask.dtype == np.uint8 else (curr_mask > 0).astype(np.uint8) * 255

        if len(p_img.shape) == 3:
            p_img = cv2.cvtColor(p_img, cv2.COLOR_BGR2GRAY)
        if len(c_img.shape) == 3:
            c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow
        flow = self.dis_flow.calc(p_img, c_img, None)  # Shape (H, W, 2), float32

        # Clamp extreme magnitude spikes to prevent runaway dislocations
        if self.config.flow_max_magnitude > 0:
            mag = np.linalg.norm(flow, axis=-1, keepdims=True)
            scale = np.minimum(1.0, self.config.flow_max_magnitude / (mag + 1e-6))
            flow = flow * scale

        return flow

    def calculate_mask_iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """Calculates Intersection-over-Union between two binary silhouette masks."""
        a_bin = (mask_a > 127)
        b_bin = (mask_b > 127)
        intersection = np.logical_and(a_bin, b_bin).sum()
        union = np.logical_or(a_bin, b_bin).sum()
        if union == 0:
            return 1.0  # Both empty is continuous
        return float(intersection) / float(union)

    # ------------------------------------------------------------------
    # 2. Semi-Lagrangian GPU Tensor Advection
    # ------------------------------------------------------------------

    def advect_state(self, flow: np.ndarray) -> None:
        """
        Warps persistent fluid (h, u, v) and pigment layers (c, d) along optical flow displacement
        using PyTorch backward Semi-Lagrangian F.grid_sample.
        
        CRITICAL INVARIANCE:
        Paper substrate topography self.paper.z_paper and pinning energy self.paper.pinning
        remain strictly STATIC in screen space, preventing sliding wallpaper artifacts.
        """
        # Convert numpy flow to GPU tensor
        flow_t = torch.from_numpy(flow).to(self.device, dtype=torch.float32, non_blocking=True)

        # Scale pixel displacement to normalized [-1, 1] grid offsets
        disp_x = flow_t[..., 0] * self.scale_x
        disp_y = flow_t[..., 1] * self.scale_y
        disp = torch.stack([disp_x, disp_y], dim=-1).unsqueeze(0)  # Shape: (1, H, W, 2)

        # Construct backward sampling coordinates
        sample_coords = torch.clamp(self.base_norm_grid - disp, -1.0, 1.0)

        mode = self.config.advection_mode
        padding = self.config.advection_padding
        align = self.config.align_corners

        # 1. Advect fluid height h
        h_in = self.fluid.h.unsqueeze(0).unsqueeze(0)
        self.fluid.h.copy_(
            F.grid_sample(h_in, sample_coords, mode=mode, padding_mode=padding, align_corners=align).squeeze()
        )

        # 2. Advect fluid velocities u, v
        u_in = self.fluid.u.unsqueeze(0).unsqueeze(0)
        v_in = self.fluid.v.unsqueeze(0).unsqueeze(0)
        self.fluid.u.copy_(
            F.grid_sample(u_in, sample_coords, mode=mode, padding_mode=padding, align_corners=align).squeeze()
        )
        self.fluid.v.copy_(
            F.grid_sample(v_in, sample_coords, mode=mode, padding_mode=padding, align_corners=align).squeeze()
        )

        # 3. Advect pigment layers: suspended (c) and deposited (d)
        for lyr in self.pigments.layers:
            c_in = lyr.c.unsqueeze(0).unsqueeze(0)
            d_in = lyr.d.unsqueeze(0).unsqueeze(0)
            lyr.c.copy_(
                F.grid_sample(c_in, sample_coords, mode=mode, padding_mode=padding, align_corners=align).squeeze()
            )
            lyr.d.copy_(
                F.grid_sample(d_in, sample_coords, mode=mode, padding_mode=padding, align_corners=align).squeeze()
            )

    def couple_motion_momentum(self, flow: np.ndarray, mask_gpu: torch.Tensor) -> None:
        """
        Couples external character motion momentum into the internal Navier-Stokes fluid velocity.
        """
        beta = self.config.momentum_coupling_beta
        if beta <= 0.0:
            return

        flow_t = torch.from_numpy(flow).to(self.device, dtype=torch.float32, non_blocking=True)
        # Scale px/frame to px/sec
        scale = self.config.momentum_fps_scale
        u_ext = flow_t[..., 0] * scale
        v_ext = flow_t[..., 1] * scale

        # Inject momentum exclusively within active silhouette
        self.fluid.u.copy_(self.fluid.u * (1.0 - beta) + u_ext * beta * mask_gpu)
        self.fluid.v.copy_(self.fluid.v * (1.0 - beta) + v_ext * beta * mask_gpu)

    def apply_boundary_wake(self, curr_mask_gpu: torch.Tensor) -> None:
        """
        Handles retreating pixels vacated by silhouette motion:
        - Active water evaporates outside the silhouette (h = 0).
        - Deposited pigment decays exponentially, leaving delicate drying trails.
        """
        if self.config.evaporate_wake_fluid:
            self.fluid.h *= curr_mask_gpu

        decay = self.config.wake_decay_rate
        # Inside silhouette: weight = 1.0; Outside: weight = decay
        wake_factor = curr_mask_gpu + (1.0 - curr_mask_gpu) * decay

        for lyr in self.pigments.layers:
            lyr.c *= curr_mask_gpu
            lyr.d *= wake_factor

    # ------------------------------------------------------------------
    # 3. Transition & Scene-Cut Handling
    # ------------------------------------------------------------------

    def handle_transition(self, mode: Optional[TransitionMode] = None) -> None:
        """
        Executes transition strategy when an IoU drop or polarity flip occurs.
        """
        t_mode = mode or self.config.transition_mode

        if t_mode == TransitionMode.SOFT_RESET:
            self.fluid.h.zero_()
            self.fluid.u.zero_()
            self.fluid.v.zero_()
            self.paper.reset_moisture()
            for lyr in self.pigments.layers:
                lyr.c.zero_()
                lyr.d.zero_()
        elif t_mode == TransitionMode.WASH_DRYING:
            self.dry_current_wash()
        elif t_mode == TransitionMode.FULL_RESET:
            self.reset()
        elif t_mode == TransitionMode.NONE:
            pass

    def soft_scene_reset(self) -> None:
        """Convenience method for soft transition reset."""
        self.handle_transition(self.config.transition_mode)

    # ------------------------------------------------------------------
    # 4. EDT-Weighted Droplet Infilling
    # ------------------------------------------------------------------

    def generate_edt_droplets(
        self,
        target_mask: np.ndarray,
        dist_map: np.ndarray,
        frame_idx: int,
        max_droplets: int = 50,
        coverage_target: float = 0.95,
    ) -> List[DropletSpec]:
        """
        Allocates organically spaced droplets weighted by Euclidean Distance Transform clearance.
        Radii are dynamically scaled: R ~ clamp(cr * D(x, y), r_min, r_max).
        Pigments are selected from the 3D harmonic Perlin field.
        """
        y_indices, x_indices = np.where(target_mask > 0)
        num_pixels = len(y_indices)
        if num_pixels == 0:
            return []

        cr = self.config.edt_scale_cr
        r_min = self.config.droplet_r_min
        r_max = self.config.droplet_r_max

        # Distance values on target pixels
        pixel_dists = dist_map[y_indices, x_indices].astype(float)
        d_max = max(float(pixel_dists.max()), 1.0)

        # Sampling probability distribution: P(x, y) ~ (D/D_max)^0.65 + 0.15
        weights = (pixel_dists / d_max) ** 0.65 + 0.15
        total_weight = weights.sum()
        if total_weight <= 0:
            probabilities = np.ones(num_pixels) / num_pixels
        else:
            probabilities = weights / total_weight

        droplets: List[DropletSpec] = []
        covered_mask = np.zeros_like(target_mask, dtype=bool)
        target_bool = (target_mask > 0)
        target_total_count = np.count_nonzero(target_bool)

        # Sample candidate points with Poisson-like spatial rejection
        candidates_pool = self.rng.choice(num_pixels, size=min(num_pixels, max_droplets * 4), p=probabilities, replace=True)

        for p_idx in candidates_pool:
            if len(droplets) >= max_droplets:
                break

            cy = float(y_indices[p_idx])
            cx = float(x_indices[p_idx])
            local_d = float(pixel_dists[p_idx])
            norm_d = local_d / d_max

            radius = float(np.clip(cr * local_d, r_min, r_max))

            # Check overlap against already placed droplets
            is_too_close = False
            for prev in droplets:
                dist_sq = (cx - prev.x) ** 2 + (cy - prev.y) ** 2
                min_sep = 0.60 * min(radius, prev.radius)
                if dist_sq < (min_sep * min_sep):
                    is_too_close = True
                    break

            if is_too_close:
                continue

            # Sample fine-art pigment from 3D harmonic palette
            col_rgb, p_name, gran, stain = self.palette.sample_pigment_at(
                x=cx, y=cy, frame_idx=frame_idx, distance_norm=norm_d
            )

            # Intensity scales slightly with radius to keep broad washes vibrant
            intensity = float(np.clip(0.80 + 0.15 * (radius / r_max), 0.70, 1.10))

            drop = DropletSpec(
                x=cx,
                y=cy,
                radius=radius,
                color_rgb=col_rgb,
                pigment_name=p_name,
                intensity=intensity,
                water_amount=0.85,
                edge_darkening=0.65,
                granulation=gran,
                staining=stain,
            )
            droplets.append(drop)

            # Rasterize coverage approximation
            cv2.circle(covered_mask, (int(round(cx)), int(round(cy))), int(round(radius)), True, -1)
            cur_cov = np.count_nonzero(covered_mask & target_bool) / max(target_total_count, 1)
            if cur_cov >= coverage_target:
                break

        # Fallback: if coverage is below 70% and we still have budget, seed residual component centroids
        if len(droplets) < max_droplets and target_total_count > 0:
            uncovered = target_bool & (~covered_mask)
            if np.count_nonzero(uncovered) > 30:
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(uncovered.astype(np.uint8))
                for lab in range(1, num_labels):
                    if len(droplets) >= max_droplets:
                        break
                    area = stats[lab, cv2.CC_STAT_AREA]
                    if area < 10:
                        continue
                    cen_x, cen_y = centroids[lab]
                    loc_d = float(dist_map[int(round(cen_y)), int(round(cen_x))]) if (0 <= int(round(cen_y)) < self.height and 0 <= int(round(cen_x)) < self.width) else 4.0
                    r = float(np.clip(cr * loc_d, r_min, r_max))
                    col_rgb, p_name, gran, stain = self.palette.sample_pigment_at(
                        x=cen_x, y=cen_y, frame_idx=frame_idx, distance_norm=loc_d / d_max
                    )
                    droplets.append(DropletSpec(
                        x=float(cen_x),
                        y=float(cen_y),
                        radius=r,
                        color_rgb=col_rgb,
                        pigment_name=p_name,
                        intensity=0.85,
                        water_amount=0.80,
                        edge_darkening=0.60,
                        granulation=gran,
                        staining=stain,
                    ))

        return droplets

    def deposit_droplets(self, droplets: List[DropletSpec]) -> None:
        """
        Deposits a list of DropletSpec onto the canvas using WatercolorEngine.add_pigment_cluster
        with unified fluid manifold potential field phi_unified.
        """
        if not droplets:
            return

        cluster_drops = []
        for d in droplets:
            cluster_drops.append({
                "x": d.x,
                "y": d.y,
                "radius": d.radius,
                "color": d.color_rgb,
                "intensity": d.intensity,
                "edge_darkening": d.edge_darkening,
                "feathering": 0.35,
                "paper_interaction": 0.12,
                "granulation": d.granulation,
                "staining": d.staining,
                "name": d.pigment_name,
            })

        self.add_pigment_cluster(cluster_drops)

    def initialize_full_silhouette(
        self,
        curr_mask: np.ndarray,
        dist_map: np.ndarray,
        frame_idx: int = 0,
    ) -> int:
        """
        Seeds initial watercolor drops across the entire silhouette for Frame 0 or after a scene cut.
        """
        self.frame_idx = frame_idx
        self.prev_mask = curr_mask.copy()

        droplets = self.generate_edt_droplets(
            target_mask=curr_mask,
            dist_map=dist_map,
            frame_idx=frame_idx,
            max_droplets=45,
            coverage_target=0.92,
        )

        self.deposit_droplets(droplets)

        # Run physical relaxation substeps
        for _ in range(self.config.sim_substeps):
            self.step(dt=self.config.sim_dt)

        return len(droplets)

    # ------------------------------------------------------------------
    # 5. Master Video Frame Simulation Step
    # ------------------------------------------------------------------

    def step_video_frame(
        self,
        prev_mask: np.ndarray,
        curr_mask: np.ndarray,
        dist_map: np.ndarray,
        frame_idx: int,
        substeps: Optional[int] = None,
        dt: Optional[float] = None,
        is_scene_cut: bool = False,
    ) -> int:
        """
        Executes a single video frame simulation step:
        1. If is_scene_cut or IoU < cut_threshold: executes transition reset & full infill.
        2. Else:
           a. Computes DIS optical flow (prev_mask -> curr_mask).
           b. Advects GPU state (h, u, v, c, d) via F.grid_sample while holding paper STATIC.
           c. Couples motion momentum into fluid velocity.
           d. Applies wake dissipation to retreating pixels.
           e. Identifies newly uncovered pixels (Delta M_new = curr_mask \\ prev_mask).
           f. Injects EDT-scaled droplets into newly uncovered pixels.
           g. Runs physical Navier-Stokes relaxation substeps.
        3. Updates state cache and returns number of droplets placed.
        """
        self.frame_idx = frame_idx
        n_substeps = substeps or self.config.sim_substeps
        step_dt = dt or self.config.sim_dt

        # Check transition condition
        iou = self.calculate_mask_iou(prev_mask, curr_mask)
        cut_triggered = is_scene_cut or (iou < self.config.iou_cut_threshold)

        if cut_triggered:
            self.handle_transition(self.config.transition_mode)
            placed = self.initialize_full_silhouette(curr_mask, dist_map, frame_idx=frame_idx)
            return placed

        # 1. Compute DIS optical flow
        flow = self.compute_optical_flow(prev_mask, curr_mask)

        # 2. Advect GPU state (fluid height, velocities, pigments)
        self.advect_state(flow)

        # 3. Create GPU mask tensor for current frame
        curr_mask_t = torch.from_numpy(curr_mask > 127).to(self.device, dtype=torch.float32, non_blocking=True)

        # 4. Momentum coupling
        self.couple_motion_momentum(flow, curr_mask_t)

        # 5. Boundary wake dissipation for retreating pixels
        self.apply_boundary_wake(curr_mask_t)

        # 6. Incremental infilling on newly uncovered silhouette pixels: Delta M_new = curr_mask \ prev_mask
        curr_bin = (curr_mask > 127)
        prev_bin = (prev_mask > 127)
        uncovered = curr_bin & (~prev_bin)
        infill_mask = uncovered.astype(np.uint8) * 255

        # Check for starved interior pixels (where persistent pigment thinned significantly)
        with torch.no_grad():
            total_pigment = torch.zeros((self.height, self.width), dtype=torch.float32, device=self.device)
            for lyr in self.pigments.layers:
                total_pigment += (lyr.c + lyr.d)
            starved_gpu = (total_pigment < self.config.infill_starvation_threshold) * curr_mask_t
            starved_np = starved_gpu.cpu().numpy() > 0.5
            # Combine newly uncovered + starved interior
            combined_infill = (uncovered | starved_np).astype(np.uint8) * 255

        infill_pixels = np.count_nonzero(combined_infill)
        droplets_placed = 0

        if infill_pixels > 15:
            # Scale droplet count to size of uncovered area
            max_drops = int(np.clip(infill_pixels // 400 + 4, 3, 30))
            droplets = self.generate_edt_droplets(
                target_mask=combined_infill,
                dist_map=dist_map,
                frame_idx=frame_idx,
                max_droplets=max_drops,
                coverage_target=0.90,
            )
            self.deposit_droplets(droplets)
            droplets_placed = len(droplets)

        # 7. Physical relaxation substeps (wet-in-wet bleed & coffee-ring accumulation)
        for _ in range(n_substeps):
            self.step(dt=step_dt)

        self.prev_mask = curr_mask.copy()
        return droplets_placed

    def render_rgb(self) -> np.ndarray:
        """
        Renders the active composite frame via Kubelka-Munk optics over static paper.
        Returns uint8 RGB numpy array of shape (H, W, 3).
        """
        return self.render()
