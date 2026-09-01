<div align="center">

# 💧 Watercolor Engine
### A Physically Based, GPU-Accelerated Watercolor Simulation & Digital Painting Engine in PyTorch

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![GUI PyQt6](https://img.shields.io/badge/GUI-PyQt6-41CD52?style=for-the-badge&logo=qt&logoColor=white)](https://www.qt.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <b>Features 2D Shallow Water Navier-Stokes fluids, porous paper hydrology, and Kubelka-Munk spectral optics for photorealistic paint bleeding, mineral granulation, and organic wet-in-wet mixing.</b>
</p>

[Key Features](#-key-features) •
[Architecture](#-architecture--pipeline) •
[Mathematical Formulation](#-mathematical-formulation) •
[Quick Start](#-quick-start) •
[Python API](#-python-api-example) •
[Interactive GUI](#-interactive-gui)

</div>

---

## 🌟 Key Features

### 🌊 1. 2D Shallow Water Navier-Stokes Dynamics
* **Semi-Lagrangian Advection:** High-order bilinear particle back-tracing on GPU grids ensures unconditionally stable fluid advection.
* **Surface Tension & Meniscus Ringing:** Laplace capillary pressure ($\Delta P = \gamma \kappa$) drives natural edge accumulation (coffee-ring effect) and fluid leveling.
* **Gravity & Inclination:** Simulates paper tilt and organic gravity dripping with directional advection forces.

### 📜 2. Porous Cellulose Substrate & Hydrology
* **Multi-Scale Fractal Topography:** Microscopic paper fiber roughness generated via fractal Perlin noise cascades across customizable octaves.
* **Deegan Contact-Line Pinning:** Pigments dynamically pin to paper fiber peaks when the liquid film thins below critical capillary thresholds.
* **Mineral Granulation:** Heavy pigment particles precipitate into microscopic paper valleys, producing authentic watercolor texture (mottling).

### 🌈 3. Kubelka-Munk Spectral Radiative Transfer Optics
* **Duncan Mixing Rule:** True subtractive spectral color blending via per-layer absorption ($K$) and scattering ($S$) coefficients.
* **Non-Linear Optical Thickness:** Preserves physical light extinction across glazes without artificial clipping or muddy black overlaps.
* **Wet Substrate Darkening:** Simulates refractive index matching between water and cellulose fibers for authentic wet paper translucency.

### 🖌️ 4. Unified Spatial Wash Manifold
* **Continuous Graded Washes:** Groups contiguous and parallel brush strokes into a unified distance field ($\phi_{unified} = \min \phi_i$), eliminating internal seams and hairline fissures.
* **Shepard Spectral Partition of Unity:** Seamless $C^\infty$ color interpolation across multi-colored wash bands.
* **Analytic Hermite Anti-Aliasing:** $C^1$ cubic smoothstep boundary transition over sub-pixel bands for razor-sharp, alias-free edges.

---

## 🏗️ Architecture & Pipeline

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                           User Interaction                             │
  │            (Continuous Catmull-Rom Splines & Pigment Drops)            │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                 1. Unified Spatial Manifold (GPU)                      │
  │     Distance Field:  φ_unified(x,y) = min_i (u_i(x,y))                 │
  │     Multi-Color:     Shepard Partition of Unity & Sub-Pixel Hermite    │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                 2. Shallow Water Fluid Dynamics                        │
  │     Height (h):      ∂h/∂t + ∇·(hu) = -e_evap                          │
  │     Velocity (u,v):  Semi-Lagrangian Advection + Viscosity + Gravity   │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                 3. Pigment Transport & Deposition                      │
  │     Suspended (c):   Advection + Capillary-Boosted Diffusion D(h)∇²c   │
  │     Deposited (d):   Deegan Contact-Line Pinning + Valley Granulation  │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                 4. Kubelka-Munk Radiative Transfer                     │
  │     Reflectance:     R = R_0 + (T_0²·R_paper) / (1 - R_0·R_paper)      │
  │     Output:          100% Photorealistic Physical Watercolor Painting  │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Mathematical Formulation

### 1. Fluid & Pigment Transport
$$\frac{\partial h}{\partial t} + \nabla \cdot (h \mathbf{u}) = -e_{\text{evap}}$$

$$\frac{\partial c_k}{\partial t} + \mathbf{u} \cdot \nabla c_k = D_{\text{eff}}(h) \nabla^2 c_k - \mathcal{R}_{\text{deposit}}(h, \nabla z, c_k)$$

where the effective diffusivity in liquid pools scales dynamically:
$$D_{\text{eff}}(h) = D_0 \left(1.0 + 18.0 \cdot \text{clamp}\left(\frac{h}{0.25}, 0, 1\right)\right)$$

### 2. Deegan Contact-Line Pinning
$$\mathcal{R}_{\text{deposit}} = \left(0.04 + 0.45 \cdot \text{clamp}\left(\frac{0.20 - h}{0.18}, 0, 1\right) + \gamma_{\text{gran}} \cdot (1 - z_{\text{paper}})\right) c_k$$

### 3. Kubelka-Munk Radiative Transfer
$$K_{\text{total}} = \sum_k c_k K_k, \qquad S_{\text{total}} = \sum_k c_k S_k$$

$$a = 1.0 + \frac{K_{\text{total}}}{S_{\text{total}}}, \qquad b = \sqrt{a^2 - 1.0}$$

$$R = R_0 + \frac{T_0^2 \cdot R_{\text{substrate}}}{1.0 - R_0 \cdot R_{\text{substrate}}}, \qquad R_0 = \frac{\sinh(b S x)}{a \sinh(b S x) + b \cosh(b S x)}$$

---

## 🚀 Quick Start

### Prerequisites
* Python 3.10 or higher
* NVIDIA GPU with CUDA support (CPU fallback is supported automatically)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/watercolor-engine.git
   cd watercolor-engine
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision numpy opencv-python PyQt6
   ```

4. **Launch the Interactive Studio:**
   ```bash
   python ui.py
   ```

---

## 💻 Python API Example

You can run headlessly and integrate the simulation into your own generative art or game development pipeline:

```python
from main import WaterColorCanvas

# 1. Initialize GPU Canvas
canvas = WaterColorCanvas(width=1000, height=800)

# 2. Generate multi-scale cold-press paper substrate
canvas.generate_paper_texture(scale=150.0, octaves=5, intensity=0.15)

# 3. Apply a graded multi-color brush wash
stroke_bundle = [
    {"points": [(100, 200), (500, 210), (900, 200)], "color": (170, 255, 255), "radius": 50.0},
    {"points": [(100, 270), (500, 280), (900, 270)], "color": (0, 255, 255),   "radius": 50.0},
    {"points": [(100, 340), (500, 350), (900, 340)], "color": (0, 174, 255),   "radius": 50.0},
    {"points": [(100, 410), (500, 420), (900, 410)], "color": (0, 0, 127),     "radius": 50.0},
]
canvas.add_strokes_wash(stroke_bundle)

# 4. Add concentrated pigment drop with wet bloom
canvas.add_droplet(x=500, y=300, radius=45.0, color_rgb=(220, 40, 60), water_amount=0.9)
canvas.apply_wet_bloom(center_x=500, center_y=300, radius=45.0, strength=0.6)

# 5. Run Navier-Stokes fluid & drying simulation
canvas.simulate_flow(iterations=15, gravity_strength=0.06, gravity_angle=90.0)
canvas.apply_drying_shift()

# 6. Save photorealistic high-res render
canvas.save_image("masterpiece.png")
```

---

## 🎨 Interactive GUI

The included **PyQt6 Studio (`ui.py`)** provides a full-featured digital painting environment:

| Feature | Description |
| :--- | :--- |
| **🖌️ Brush Mode** | Smooth drag-and-draw with speed-sensitive tapering, dry-brush scumbling, and fiber split simulation. |
| **💧 Drop & Bloom** | Interactive pigment dropper with wet-in-wet capillary explosion and splatter dispersion. |
| **🌊 Live Fluid Parameters** | Real-time sliders for fluid viscosity, diffusion rate, numerical turbulence, and gravity angle. |
| **⚡ Instant GPU Preview** | Ultra-fast interactive preview with 1-click full-resolution 4K export. |

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

<div align="center">
  <sub>Built with ❤️ for artists, game developers, and computer graphics researchers.</sub>
</div>
