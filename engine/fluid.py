import math
import torch
import torch.nn.functional as F


class ShallowWaterSolver:
    """
    2D Shallow Water (Saint-Venant) and Navier-Stokes Fluid Solver
    
    Physical Conservations:
    - Free surface fluid thickness h(x, y) >= 0
    - Velocity field u(x, y), v(x, y) (pixels/second)
    - Mass-conserving Semi-Lagrangian Advection (grid_sample)
    - Hydrostatic pressure and topographic slope acceleration: -g * grad(h + z_paper)
    - Surface tension (Young-Laplace): gamma * grad(Laplace(h))
    - Deegan outward capillary evaporative flux
    - Capillary contact line pinning
    """

    def __init__(
        self,
        width: int,
        height: int,
        gravity: float = 380.0,
        viscosity: float = 0.35,
        surface_tension: float = 18.0,
        friction_coeff: float = 0.25,
        evap_base: float = 0.002,
        evap_edge_boost: float = 0.05,
        device: torch.device = None,
    ):
        self.width = width
        self.height = height
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.g = gravity
        self.viscosity = viscosity
        self.gamma = surface_tension
        self.friction_coeff = friction_coeff
        self.evap_base = evap_base
        self.evap_edge_boost = evap_edge_boost

        self.h = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        self.u = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        self.v = torch.zeros((height, width), dtype=torch.float32, device=self.device)

        y_coords = torch.linspace(-1.0, 1.0, height, device=self.device)
        x_coords = torch.linspace(-1.0, 1.0, width, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        self.base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        self._laplace_kernel = torch.tensor(
            [[[[0.0, 1.0, 0.0],
               [1.0, -4.0, 1.0],
               [0.0, 1.0, 0.0]]]],
            dtype=torch.float32,
            device=self.device,
        )

    def add_fluid(self, mask: torch.Tensor, amount: float = 1.0):
        self.h += mask * amount

    def step(
        self,
        dt: float,
        z_paper: torch.Tensor,
        absorbed_flux: torch.Tensor,
        pinning_mask: torch.Tensor,
        external_force_x: float = 0.0,
        external_force_y: float = 0.0,
    ):
        wet_mask = (self.h > 1e-4)

        # -------------------------------------------------------------
        # 1. SEMI-LAGRANGIAN ADVECTION (Transport of u, v, and h fields)
        # -------------------------------------------------------------
        scale_x = 2.0 / max(self.width, 1)
        scale_y = 2.0 / max(self.height, 1)

        disp_x = self.u * dt * scale_x
        disp_y = self.v * dt * scale_y
        disp = torch.stack([disp_x, disp_y], dim=-1).unsqueeze(0)

        sample_coords = torch.clamp(self.base_grid - disp, -1.0, 1.0)

        h_in = self.h.unsqueeze(0).unsqueeze(0)
        u_in = self.u.unsqueeze(0).unsqueeze(0)
        v_in = self.v.unsqueeze(0).unsqueeze(0)

        h_adv = F.grid_sample(h_in, sample_coords, mode="bilinear", padding_mode="border", align_corners=True).squeeze()
        u_adv = F.grid_sample(u_in, sample_coords, mode="bilinear", padding_mode="border", align_corners=True).squeeze()
        v_adv = F.grid_sample(v_in, sample_coords, mode="bilinear", padding_mode="border", align_corners=True).squeeze()

        self.h = h_adv
        self.u = u_adv
        self.v = v_adv

        # -------------------------------------------------------------
        # 2. HYDROSTATIC PRESSURE & TOPOGRAPHIC SLOPE FORCE
        # Total Height: eta = h + z_paper * 3.5
        # -------------------------------------------------------------
        eta = self.h + z_paper * 3.5
        pad_eta = F.pad(eta.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate").squeeze()
        grad_eta_x = 0.5 * (pad_eta[1:-1, 2:] - pad_eta[1:-1, :-2])
        grad_eta_y = 0.5 * (pad_eta[2:, 1:-1] - pad_eta[:-2, 1:-1])

        f_press_x = -self.g * grad_eta_x
        f_press_y = -self.g * grad_eta_y

        # -------------------------------------------------------------
        # 3. SURFACE TENSION (Young-Laplace Curvature Pressure)
        # Meniscus and capillary coalescence
        # -------------------------------------------------------------
        pad_h = F.pad(self.h.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
        curv = F.conv2d(pad_h, self._laplace_kernel).squeeze()

        pad_curv = F.pad(curv.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate").squeeze()
        grad_curv_x = 0.5 * (pad_curv[1:-1, 2:] - pad_curv[1:-1, :-2])
        grad_curv_y = 0.5 * (pad_curv[2:, 1:-1] - pad_curv[:-2, 1:-1])

        f_cap_x = self.gamma * grad_curv_x
        f_cap_y = self.gamma * grad_curv_y

        # -------------------------------------------------------------
        # 4. DEEGAN OUTWARD CAPILLARY RADIAL FLUX
        # -------------------------------------------------------------
        pad_h_raw = F.pad(self.h.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate").squeeze()
        grad_h_x = 0.5 * (pad_h_raw[1:-1, 2:] - pad_h_raw[1:-1, :-2])
        grad_h_y = 0.5 * (pad_h_raw[2:, 1:-1] - pad_h_raw[:-2, 1:-1])

        edge_proximity = torch.clamp((0.35 - self.h) / 0.30, 0.0, 1.0)
        deegan_flux_x = -grad_h_x * edge_proximity * 40.0
        deegan_flux_y = -grad_h_y * edge_proximity * 40.0

        # -------------------------------------------------------------
        # 5. VELOCITY UPDATE, EXTERNAL GRAVITY & DAMPING
        # -------------------------------------------------------------
        # Shallow Water acceleration in the wet domain
        acc_x = f_press_x + f_cap_x + deegan_flux_x + external_force_x
        acc_y = f_press_y + f_cap_y + deegan_flux_y + external_force_y

        self.u += acc_x * dt
        self.v += acc_y * dt

        # Realistic bottom friction & contact-line pinning
        # Deep pools have low friction; thin films near pinned contact line experience pinning
        edge_pin = pinning_mask * 1.2 * torch.clamp((0.15 - self.h) / 0.12, 0.0, 1.0)
        eff_depth = torch.clamp(self.h, min=0.03)
        friction = self.viscosity + (self.friction_coeff * 0.15 + edge_pin * 0.4) / eff_depth
        damping = torch.clamp(1.0 - friction * dt, min=0.05, max=1.0)
        self.u *= damping
        self.v *= damping

        # Capillary braking strictly on dry canvas
        dry_mask = (self.h <= 1e-4)
        self.u[dry_mask] = 0.0
        self.v[dry_mask] = 0.0
        edge_brake = torch.clamp(self.h / 0.01, 0.0, 1.0)
        self.u *= edge_brake
        self.v *= edge_brake

        # -------------------------------------------------------------
        # 6. EVAPORATION AND CAPILLARY ABSORPTION LOSS (Mass Conservation)
        # -------------------------------------------------------------
        edge_singularity = self.evap_edge_boost / torch.sqrt(torch.clamp(self.h, min=0.02))
        evaporation_flux = (self.evap_base + edge_singularity) * wet_mask.float()

        dh = -evaporation_flux * dt - absorbed_flux
        self.h = torch.clamp(self.h + dh, min=0.0)

        # Canvas boundary conditions
        self.h[0, :] = self.h[-1, :] = self.h[:, 0] = self.h[:, -1] = 0.0
        self.u[0, :] = self.u[-1, :] = self.u[:, 0] = self.u[:, -1] = 0.0
        self.v[0, :] = self.v[-1, :] = self.v[:, 0] = self.v[:, -1] = 0.0

    def reset(self):
        self.h.zero_()
        self.u.zero_()
        self.v.zero_()

    def get_mass_and_center_of_mass(self) -> tuple[float, tuple[float, float]]:
        """Computes total fluid mass and (x, y) center of mass in the wet domain."""
        total_mass = float(self.h.sum().item())
        if total_mass < 1e-6:
            return 0.0, (0.0, 0.0)
        
        y_coords = torch.arange(self.height, device=self.device, dtype=torch.float32).view(-1, 1)
        x_coords = torch.arange(self.width, device=self.device, dtype=torch.float32).view(1, -1)
        
        cm_x = float((self.h * x_coords).sum().item() / total_mass)
        cm_y = float((self.h * y_coords).sum().item() / total_mass)
        return total_mass, (cm_x, cm_y)
