import math
import torch
import torch.nn.functional as F


def fast_fractal_noise_gpu(
    width: int,
    height: int,
    scale: float,
    octaves: int = 3,
    device: torch.device = None,
    seed: int = None,
) -> torch.Tensor:
    """
    GPU Üzerinde Ultra Hızlı Bikübik Fraktal Gürültü Motoru (Sub-millisecond Tensor Noise)
    Kağıt lifleri, fırça kenar kırılmaları ve gövde pıhtılaşması için kullanılır.
    """
    if scale < 0.5:
        scale = 0.5
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=dev)
    if seed is not None:
        generator.manual_seed(seed)

    noise = torch.zeros((1, 1, height, width), dtype=torch.float32, device=dev)
    cur_scale = float(scale)
    cur_amp = 1.0
    tot_amp = 0.0

    for _ in range(octaves):
        gw = max(2, int(math.ceil(width / cur_scale)) + 2)
        gh = max(2, int(math.ceil(height / cur_scale)) + 2)
        grid = torch.rand((1, 1, gh, gw), generator=generator, dtype=torch.float32, device=dev)
        upsampled = F.interpolate(grid, size=(height, width), mode="bicubic", align_corners=False)
        noise += upsampled * cur_amp
        tot_amp += cur_amp
        cur_scale /= 2.0
        cur_amp *= 0.5

    noise = noise.squeeze()
    t_min, t_max = noise.min(), noise.max()
    if t_max - t_min > 1e-5:
        return (noise - t_min) / (t_max - t_min)
    return noise / max(tot_amp, 1e-5)
