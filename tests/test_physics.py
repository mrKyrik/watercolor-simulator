"""
Physics-Based Watercolor Simulation Validation Suite
"""

import os
import sys
import time
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.simulation import WatercolorEngine


def test_gravity_flow():
    print("\n--- TEST 1: GRAVITY AND INCLINE FLOW ---")
    engine = WatercolorEngine(width=500, height=400)
    # Deposit a droplet of blue paint in the center
    engine.add_droplet(250, 120, radius=40, color_rgb=(25, 75, 210), water_amount=1.2, pigment_name="Ultramarine Blue")
    
    # Run 30 steps with strong downward gravity (90 degrees)
    engine.run_simulation(steps=30, dt=0.06, gravity_strength=0.8, gravity_angle_deg=90.0)
    
    out_img = engine.render()
    out_path = "test_output_gravity.png"
    cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"Gravity flow test saved: {out_path}")
    
    # Verify fluid velocity and mass moves downward
    wet_mask = (engine.fluid.h > 0.01)
    v_wet = engine.fluid.v[wet_mask].mean().item() if wet_mask.any() else 0.0
    v_mean = engine.fluid.v.mean().item()
    print(f"Wet fluid vertical velocity (v_wet): {v_wet:.4f} (Positive: Downward flow)")
    print(f"Canvas-wide vertical velocity (v_mean): {v_mean:.4f}")
    assert v_wet > 0.0, "Fluid in the wet droplet must flow downward in the direction of gravity!"


def test_wet_in_wet_bleeding():
    print("\n--- TEST 2: WET-IN-WET BLEEDING ---")
    engine = WatercolorEngine(width=600, height=400)
    
    # Deposit Phthalo Blue drop on the left
    engine.add_droplet(275, 200, radius=50, color_rgb=(0, 110, 210), water_amount=1.1, pigment_name="Phthalo Blue")
    # Deposit Cadmium Yellow drop touching the blue drop on the right
    engine.add_droplet(325, 200, radius=50, color_rgb=(245, 215, 10), water_amount=1.1, pigment_name="Cadmium Yellow")
    
    # Run 35 steps of surface tension and pressure gradient flow into each other
    engine.run_simulation(steps=35, dt=0.05, gravity_strength=0.0)
    
    out_img = engine.render()
    out_path = "test_output_wet_in_wet.png"
    cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"Wet-in-wet bleeding test saved: {out_path}")
    
    # Check pixels in the intersection zone
    mid_strip = out_img[180:220, 290:310]
    mean_r = float(mid_strip[:, :, 0].mean())
    mean_g = float(mid_strip[:, :, 1].mean())
    mean_b = float(mid_strip[:, :, 2].mean())
    print(f"Middle blend zone color average: R={mean_r:.1f}, G={mean_g:.1f}, B={mean_b:.1f}")


def test_coffee_ring_effect():
    print("\n--- TEST 3: DEEGAN COFFEE RING EDGE DARKENING ---")
    engine = WatercolorEngine(width=400, height=400)
    
    # Deposit a single droplet
    cx, cy, r = 200, 200, 50
    engine.add_droplet(cx, cy, radius=r, color_rgb=(180, 40, 30), water_amount=0.8, pigment_name="Cadmium Red")
    
    # Run evaporation steps (40 steps)
    engine.run_simulation(steps=40, dt=0.06, gravity_strength=0.0)
    
    out_img = engine.render()
    out_path = "test_output_coffee_ring.png"
    cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"Coffee ring test saved: {out_path}")
    
    # Compare edge pigment concentration with center
    lyr = engine.pigments.layers[0]
    total_pigment = lyr.c + lyr.d
    center_val = total_pigment[cy, cx].item()
    # Edge ring at radius r
    edge_val = total_pigment[cy, cx + int(r * 0.85)].item()
    print(f"Center pigment: {center_val:.4f}, Edge ring pigment: {edge_val:.4f}")


def test_granulation_valley_pooling():
    print("\n--- TEST 4: GRANULATION AND PAPER VALLEY SEDIMENTATION ---")
    engine = WatercolorEngine(width=400, height=400)
    
    # High granulation pigment (Burnt Umber)
    engine.add_droplet(200, 200, radius=60, color_rgb=(100, 50, 25), water_amount=0.9, pigment_name="Burnt Umber", granulation=0.95)
    engine.run_simulation(steps=30, dt=0.05, gravity_strength=0.0)
    
    out_img = engine.render()
    out_path = "test_output_granulation.png"
    cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"Granulation test saved: {out_path}")


def test_continuous_brush_stroke():
    print("\n--- TEST 5: CONTINUOUS BRUSH STROKE AND DRY BRUSH ---")
    engine = WatercolorEngine(width=800, height=600)
    
    # S-curve brush stroke
    points = [(100, 150), (250, 100), (450, 300), (600, 500), (720, 420)]
    engine.add_stroke_from_points(
        points=points,
        color_rgb=(30, 110, 195),
        radius=30.0,
        intensity=0.85,
        pigment_name="Ultramarine Blue",
        granulation=0.8,
        dry_brush=0.35,
    )
    
    # Add a few splatter droplets
    engine.add_splatter(cx=450, cy=300, radius=60, color_rgb=(210, 40, 60), count=10, pigment_name="Cadmium Red")
    
    # 20 steps of physical flow and diffusion
    t0 = time.time()
    engine.run_simulation(steps=20, dt=0.05, gravity_strength=0.25, gravity_angle_deg=110.0)
    elapsed = time.time() - t0
    print(f"20 Physical Steps Simulation Time: {elapsed:.3f} seconds ({20 / elapsed:.1f} FPS)")
    
    out_img = engine.render()
    out_path = "test_output_stroke_composite.png"
    cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"Brush stroke composite test saved: {out_path}")


if __name__ == "__main__":
    test_gravity_flow()
    test_wet_in_wet_bleeding()
    test_coffee_ring_effect()
    test_granulation_valley_pooling()
    test_continuous_brush_stroke()
    print("\nALL PHYSICAL VALIDATION TESTS EXECUTED SUCCESSFULLY!")
