import torch
import torch.nn.functional as F
import numpy as np

def test_morphological_closing():
    # 200x200 alanda, ortasında 20 piksellik delik olan bir döngü fırça darbesi
    grid_y, grid_x = torch.meshgrid(torch.arange(200), torch.arange(200), indexing="ij")
    # Halka şeklinde darbe (merkezi 100, 100; yarıçapı 40, fırça yarıçapı 25)
    r_center = torch.sqrt((grid_x - 100.0)**2 + (grid_y - 100.0)**2)
    dist_to_ring = torch.abs(r_center - 40.0)
    phi = dist_to_ring / 25.0  # Ortasında dist = 40 > 25 olduğu için delik kalır!

    print(f"Orijinal delik merkezi phi: {phi[100, 100].item():.3f} (1.0'dan buyuk oldugu icin beyaz delik!)")

    # Morfolojik Kapanma (Capillary Hole Closing)
    # Kılcal yüzey gerilimi iç delikleri kapatır
    k_size = 31
    pad = k_size // 2
    # Negatif phi üzerinde max_pool (genleşme)
    neg_phi = -phi.unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(neg_phi, kernel_size=k_size, stride=1, padding=pad)
    # Ardından -max_pool(-dilated) (erozyon)
    closed_neg = -F.max_pool2d(-dilated, kernel_size=k_size, stride=1, padding=pad)
    phi_closed = -closed_neg.squeeze()

    print(f"Kapanma sonrasi delik merkezi phi: {phi_closed[100, 100].item():.3f} (1.0'in altina indi, delik kapandi!)")
    print(f"Dis sinir (x=175): Orijinal phi = {phi[100, 175].item():.3f}, Kapanma sonrasi = {phi_closed[100, 175].item():.3f} (Dis sinir korundu!)")

if __name__ == "__main__":
    test_morphological_closing()
