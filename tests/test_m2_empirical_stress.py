"""
tests/test_m2_empirical_stress.py - Empirical Stress Test Suite for Milestone 2 Watercolor Video Engine.

Adversarially and empirically stress-tests the visual and physical watercolor dynamics
of data/bad_apple_watercolor.mp4 across 4 dimensions:
1. Anti-Boiling & Motion Consistency: Backward motion-compensated MAE, temporal strobe acceleration, and absence of high-frequency flicker.
2. Fluid Droplet Fill Dynamics: Organic pigment RGB variance, multi-channel gradients, absence of flat monochrome fills.
3. Fluid Bleeding & Coffee-Ring Effect: Meniscus edge darkening, boundary soft gradient transition width, capillary bleed.
4. Temporal Stability & Container Integrity: Zero NaN/Inf, Shannon entropy (> 2.5 bits) on all 225 frames, smooth continuity, audio-video muxing.
"""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.video_simulation import (
    VideoWatercolorEngine,
    VideoSimulationConfig,
    DropletSpec,
    FINE_ART_PIGMENTS,
)

VIDEO_PATH = Path("data/bad_apple_watercolor.mp4")
PREPROCESSED_DIR = Path("data/preprocessed")
MASKS_DIR = PREPROCESSED_DIR / "masks"
METADATA_PATH = PREPROCESSED_DIR / "metadata.json"


@pytest.fixture(scope="module")
def loaded_video_frames():
    """Loads all frames from the deliverable video data/bad_apple_watercolor.mp4."""
    if not VIDEO_PATH.exists():
        pytest.skip(f"Video deliverable {VIDEO_PATH} does not exist")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    assert cap.isOpened(), f"Failed to open video file {VIDEO_PATH}"
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) == 225, f"Expected 225 frames, got {len(frames)}"
    return frames


@pytest.fixture(scope="module")
def loaded_masks():
    """Loads all binary preprocessed masks."""
    if not MASKS_DIR.exists():
        pytest.skip(f"Masks directory {MASKS_DIR} does not exist")
    masks = []
    for i in range(225):
        m_path = MASKS_DIR / f"mask_{i:04d}.png"
        assert m_path.exists(), f"Missing mask {m_path}"
        m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
        masks.append(m)
    return masks


# =====================================================================
# 1. Empirical Anti-Boiling & Motion Compensation Tests
# =====================================================================

class TestEmpiricalAntiBoiling:
    """
    Empirical anti-boiling tests verifying that Semi-Lagrangian GPU state advection
    couples fluid and pigment to silhouette motion, eliminating temporal boiling.
    """

    def test_interframe_motion_compensated_mae(self, loaded_video_frames, loaded_masks):
        """
        Computes backward motion-compensated pixel difference across consecutive frames.
        Warping frame t to frame t+1 via DIS optical flow must yield low residual MAE (< 15.0).
        """
        dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        mc_maes = []
        raw_maes = []

        for i in range(len(loaded_video_frames) - 1):
            m0 = loaded_masks[i]
            m1 = loaded_masks[i + 1]

            overlap = (m0 > 127) & (m1 > 127)
            union = (m0 > 127) | (m1 > 127)
            iou = np.count_nonzero(overlap) / max(np.count_nonzero(union), 1)

            # Skip scene cuts (abrupt transitions) and near-empty frames
            if iou < 0.15 or np.count_nonzero(overlap) < 500:
                continue

            f0 = loaded_video_frames[i].astype(np.float32)
            f1 = loaded_video_frames[i + 1].astype(np.float32)

            # Calculate dense optical flow from m0 to m1
            flow = dis.calc(m0, m1, None)

            h, w = m0.shape
            grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
            # Backward warp: where in f0 does each pixel in f1 come from?
            map_x = grid_x - flow[..., 0]
            map_y = grid_y - flow[..., 1]
            warped_f0 = cv2.remap(
                f0, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )

            diff_mc = np.abs(warped_f0 - f1)
            diff_raw = np.abs(f0 - f1)

            mc_maes.append(float(np.mean(diff_mc[overlap])))
            raw_maes.append(float(np.mean(diff_raw[overlap])))

        assert len(mc_maes) >= 200, f"Expected >= 200 continuous frame pairs, tested {len(mc_maes)}"

        mean_mc_mae = float(np.mean(mc_maes))
        max_mc_mae = float(np.max(mc_maes))
        min_mc_mae = float(np.min(mc_maes))

        # Assert backward motion-compensated residual is bounded and smooth
        assert mean_mc_mae < 15.0, f"Mean motion-compensated MAE too high: {mean_mc_mae:.2f}"
        assert max_mc_mae < 75.0, f"Max motion-compensated MAE spike too high: {max_mc_mae:.2f}"
        assert min_mc_mae < 5.0, f"Min motion-compensated MAE should be low in static regions: {min_mc_mae:.2f}"

    def test_motion_compensation_error_reduction(self, loaded_video_frames, loaded_masks):
        """
        Verifies that motion compensation substantially reduces inter-frame error
        by at least 50% relative to uncompensated frame difference, proving state advection.
        """
        dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        reductions = []

        for i in range(len(loaded_video_frames) - 1):
            m0 = loaded_masks[i]
            m1 = loaded_masks[i + 1]
            overlap = (m0 > 127) & (m1 > 127)
            union = (m0 > 127) | (m1 > 127)
            iou = np.count_nonzero(overlap) / max(np.count_nonzero(union), 1)

            if iou < 0.15 or np.count_nonzero(overlap) < 1000:
                continue

            f0 = loaded_video_frames[i].astype(np.float32)
            f1 = loaded_video_frames[i + 1].astype(np.float32)

            flow = dis.calc(m0, m1, None)
            h, w = m0.shape
            grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
            map_x = grid_x - flow[..., 0]
            map_y = grid_y - flow[..., 1]
            warped_f0 = cv2.remap(f0, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            raw_err = float(np.mean(np.abs(f0 - f1)[overlap]))
            mc_err = float(np.mean(np.abs(warped_f0 - f1)[overlap]))

            if raw_err > 5.0:
                rel_reduction = (raw_err - mc_err) / raw_err
                reductions.append(rel_reduction)

        mean_reduction = float(np.mean(reductions))
        # Motion compensation should reduce error by over 50% on average (empirically > 70%)
        assert mean_reduction > 0.50, f"Expected > 50% error reduction via motion compensation, got {mean_reduction * 100:.1f}%"

    def test_second_order_temporal_strobe_metric(self, loaded_video_frames, loaded_masks):
        """
        Computes 3-frame motion-compensated second temporal difference:
        S = 0.5 * |W(f0) - 2*f1 + W(f2)|.
        Absence of high-frequency strobe oscillation requires mean S < 20.0.
        """
        dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        mc_strobe_metrics = []

        for i in range(len(loaded_video_frames) - 2):
            m0 = loaded_masks[i]
            m1 = loaded_masks[i + 1]
            m2 = loaded_masks[i + 2]

            overlap3 = (m0 > 127) & (m1 > 127) & (m2 > 127)
            iou01 = np.count_nonzero((m0 > 127) & (m1 > 127)) / max(np.count_nonzero((m0 > 127) | (m1 > 127)), 1)
            iou12 = np.count_nonzero((m1 > 127) & (m2 > 127)) / max(np.count_nonzero((m1 > 127) | (m2 > 127)), 1)
            if iou01 < 0.15 or iou12 < 0.15 or np.count_nonzero(overlap3) < 500:
                continue

            flow01 = dis.calc(m0, m1, None)
            flow21 = dis.calc(m2, m1, None)

            h, w = m0.shape
            gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)

            map0_x = gx - flow01[..., 0]
            map0_y = gy - flow01[..., 1]
            w_f0 = cv2.remap(loaded_video_frames[i].astype(np.float32), map0_x, map0_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            map2_x = gx - flow21[..., 0]
            map2_y = gy - flow21[..., 1]
            w_f2 = cv2.remap(loaded_video_frames[i + 2].astype(np.float32), map2_x, map2_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            f1 = loaded_video_frames[i + 1].astype(np.float32)

            mc_strobe = 0.5 * np.abs(w_f0 - 2.0 * f1 + w_f2)
            mc_strobe_metrics.append(float(np.mean(mc_strobe[overlap3])))

        mean_strobe = float(np.mean(mc_strobe_metrics))
        assert mean_strobe < 20.0, f"Second-order temporal strobe metric too high: {mean_strobe:.2f}"


# =====================================================================
# 2. Fluid Droplet Fill Dynamics Tests
# =====================================================================

class TestFluidDropletFillDynamics:
    """
    Verifies that character silhouettes are filled with organic watercolor pigment
    exhibiting non-zero RGB variance, multi-channel color gradients, and absence of flat monochrome fills.
    """

    def test_silhouette_interior_rgb_variance(self, loaded_video_frames, loaded_masks):
        """
        Asserts non-zero, high RGB variance across silhouette interiors in all non-empty frames.
        Eliminates flat monochrome fills (e.g. solid thresholding or uniform fills).
        """
        variances = []
        for i, (f, m) in enumerate(zip(loaded_video_frames, loaded_masks)):
            fg = m > 127
            if np.count_nonzero(fg) > 300:
                r = f[fg, 2].astype(float)
                g = f[fg, 1].astype(float)
                b = f[fg, 0].astype(float)
                var_rgb = float(np.var(r) + np.var(g) + np.var(b))
                variances.append(var_rgb)

        assert len(variances) >= 200
        min_var = float(np.min(variances))
        mean_var = float(np.mean(variances))

        # Flat fills would have variance ~ 0. Organic watercolor must have rich tonal spread.
        assert min_var > 500.0, f"Interior RGB variance too low on frame (flat fill detected): {min_var:.2f}"
        assert mean_var > 5000.0, f"Mean interior RGB variance too low: {mean_var:.2f}"

    def test_multi_channel_color_gradients(self, loaded_video_frames, loaded_masks):
        """
        Verifies multi-channel chromatic diversity across R, G, and B inside silhouettes:
        std(R-G) + std(G-B) + std(B-R) > 15.0.
        """
        color_diversities = []
        for f, m in zip(loaded_video_frames, loaded_masks):
            fg = m > 127
            if np.count_nonzero(fg) > 500:
                r = f[fg, 2].astype(float)
                g = f[fg, 1].astype(float)
                b = f[fg, 0].astype(float)
                div = float(np.std(r - g) + np.std(g - b) + np.std(b - r))
                color_diversities.append(div)

        assert len(color_diversities) >= 200
        min_div = float(np.min(color_diversities))
        mean_div = float(np.mean(color_diversities))

        assert min_div > 15.0, f"Color diversity minimum too low: {min_div:.2f}"
        assert mean_div > 80.0, f"Mean color diversity too low: {mean_div:.2f}"

    def test_absence_of_flat_monochrome_fills(self, loaded_video_frames, loaded_masks):
        """
        Checks that histogram distributions inside the silhouette are continuous and multi-tonal,
        not concentrated in a single peak (e.g. flat binary fill).
        """
        for i in range(0, len(loaded_video_frames), 15):
            f = loaded_video_frames[i]
            m = loaded_masks[i]
            fg = m > 127
            if np.count_nonzero(fg) < 500:
                continue

            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[fg]
            hist, _ = np.histogram(gray, bins=32, range=(0, 256))
            # Number of non-empty bins in a 32-bin histogram
            active_bins = np.count_nonzero(hist > 0)
            assert active_bins >= 8, f"Frame {i} has only {active_bins} active tonal bins (flat fill detected)"


# =====================================================================
# 3. Fluid Bleeding & Coffee-Ring Effect Tests
# =====================================================================

class TestFluidBleedingAndCoffeeRing:
    """
    Verifies presence of soft gradient borders and edge darkening (contact-line pinning)
    at silhouette boundaries and individual droplet clusters.
    """

    def test_droplet_radial_coffee_ring_meniscus_peak(self):
        """
        Empirically verifies in the simulation engine that a watercolor droplet exhibits
        a distinct coffee-ring edge peak (meniscus accumulation) at r ~ 0.8-0.9 R before edge falloff.
        """
        engine = VideoWatercolorEngine(width=250, height=250)
        drop = DropletSpec(
            x=125.0,
            y=125.0,
            radius=75.0,
            color_rgb=(20, 70, 170),
            pigment_name="Prussian Blue",
            intensity=0.90,
            edge_darkening=0.85,
        )
        engine.deposit_droplets([drop])

        lyr = engine.pigments.layers[0]
        tot_pigment = (lyr.c + lyr.d).cpu().numpy()

        y, x = np.ogrid[:250, :250]
        r_map = np.sqrt((x - 125.0) ** 2 + (y - 125.0) ** 2)

        # Compare interior body (r in [40, 50]) vs meniscus ring (r in [60, 68])
        mask_interior = (r_map >= 40.0) & (r_map <= 50.0)
        mask_ring = (r_map >= 60.0) & (r_map <= 68.0)
        mask_outside = (r_map >= 90.0) & (r_map <= 100.0)

        p_interior = float(np.mean(tot_pigment[mask_interior]))
        p_ring = float(np.mean(tot_pigment[mask_ring]))
        p_outside = float(np.mean(tot_pigment[mask_outside]))

        # Coffee ring effect: pigment at meniscus ring is significantly higher than interior body
        assert p_ring > p_interior * 1.05, f"Meniscus ring ({p_ring:.4f}) should exceed interior ({p_interior:.4f}) by >= 5%"
        # Soft falloff: outside is near zero
        assert p_outside < 0.15 * p_ring, f"Outside boundary should feather down to paper: {p_outside:.4f}"

    def test_boundary_soft_gradient_transition_width(self, loaded_video_frames, loaded_masks):
        """
        Verifies that silhouette boundaries in the video exhibit soft, feathered gradient borders
        (transition span across boundary >= 25.0 darkness units) rather than sharp binary mask step edges.
        """
        transition_spans = []

        for i in range(0, len(loaded_video_frames), 10):
            f = loaded_video_frames[i]
            m = loaded_masks[i]
            if np.count_nonzero(m > 127) < 1000:
                continue

            fg = (m > 127).astype(np.uint8)
            bg = (m <= 127).astype(np.uint8)
            d_in = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
            d_out = cv2.distanceTransform(bg, cv2.DIST_L2, 5)

            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float)
            darkness = 255.0 - gray

            # Outer boundary band in paper: d_out in [1.5, 3.0]
            mask_out = (d_out >= 1.5) & (d_out <= 3.0)
            # Inner boundary band in silhouette: d_in in [3.0, 5.0]
            mask_in = (d_in >= 3.0) & (d_in <= 5.0)

            if np.count_nonzero(mask_out) > 50 and np.count_nonzero(mask_in) > 50:
                dark_out = float(np.mean(darkness[mask_out]))
                dark_in = float(np.mean(darkness[mask_in]))
                span = dark_in - dark_out
                transition_spans.append(span)

        assert len(transition_spans) >= 15
        mean_span = float(np.mean(transition_spans))
        # A soft feathered edge must show significant darkness increase across the boundary
        assert mean_span > 25.0, f"Boundary transition too steep or flat (mean span: {mean_span:.2f})"

    def test_boundary_capillary_pigment_bleed(self, loaded_video_frames, loaded_masks):
        """
        Verifies that pigment bleeds past the binary mask contact line into the paper substrate:
        Darkness in the 1-2px band immediately outside the mask (d_out in [1, 2])
        is elevated compared to deep bare paper (d_out >= 15).
        """
        bleed_elevations = []

        for i in range(0, len(loaded_video_frames), 5):
            f = loaded_video_frames[i]
            m = loaded_masks[i]
            if np.count_nonzero(m > 127) < 1000:
                continue

            bg = (m <= 127).astype(np.uint8)
            d_out = cv2.distanceTransform(bg, cv2.DIST_L2, 5)

            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float)
            darkness = 255.0 - gray

            bleed_band = (d_out >= 1.0) & (d_out <= 2.0)
            far_paper = (d_out >= 15.0)

            if np.count_nonzero(bleed_band) > 100 and np.count_nonzero(far_paper) > 500:
                dark_bleed = float(np.mean(darkness[bleed_band]))
                dark_paper = float(np.mean(darkness[far_paper]))
                bleed_elevations.append(dark_bleed - dark_paper)

        assert len(bleed_elevations) >= 30
        mean_bleed = float(np.mean(bleed_elevations))
        # Capillary bleeding into paper must elevate darkness near the boundary
        assert mean_bleed > 1.0, f"Expected capillary bleed past contact line, got delta: {mean_bleed:.2f}"


# =====================================================================
# 4. Temporal Stability & Container Integrity Tests
# =====================================================================

class TestTemporalStabilityAndContinuity:
    """
    Verifies zero NaN/Inf values, non-zero Shannon entropy (> 2.5 bits),
    smooth temporal continuity across all 225 frames, and valid container muxing.
    """

    def test_zero_nan_inf_all_225_frames(self, loaded_video_frames):
        """Asserts zero NaN or Inf values across all 225 frames."""
        for i, frame in enumerate(loaded_video_frames):
            assert not np.isnan(frame).any(), f"NaN detected at frame {i}"
            assert not np.isinf(frame).any(), f"Inf detected at frame {i}"

    def test_shannon_entropy_all_225_frames(self, loaded_video_frames):
        """
        Computes Shannon spatial entropy for every single frame across the 225-frame timeline.
        Asserts H > 2.5 bits for 100% of frames (verifies rich paper & pigment texture).
        """
        entropies = []
        for i, frame in enumerate(loaded_video_frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
            hist = hist[hist > 0]
            ent = -float(np.sum(hist * np.log2(hist)))
            entropies.append(ent)
            assert ent > 2.5, f"Frame {i} entropy too low: {ent:.3f} bits"

        min_ent = float(np.min(entropies))
        mean_ent = float(np.mean(entropies))
        max_ent = float(np.max(entropies))

        assert min_ent >= 3.5, f"Expected min entropy >= 3.5, got {min_ent:.3f}"
        assert mean_ent >= 4.5, f"Expected mean entropy >= 4.5, got {mean_ent:.3f}"

    def test_smooth_temporal_continuity_and_scene_cuts(self, loaded_video_frames, loaded_masks):
        """
        Verifies smooth inter-frame transition across continuous scenes
        and confirms scene cut detection aligns with mask IoU drops (< 0.15).
        """
        scene_cut_indices = []
        continuous_maes = []

        for i in range(len(loaded_video_frames) - 1):
            m0 = loaded_masks[i]
            m1 = loaded_masks[i + 1]

            overlap = (m0 > 127) & (m1 > 127)
            union = (m0 > 127) | (m1 > 127)
            iou = np.count_nonzero(overlap) / max(np.count_nonzero(union), 1)

            diff = np.abs(loaded_video_frames[i + 1].astype(float) - loaded_video_frames[i].astype(float))
            mean_diff = float(np.mean(diff))

            if iou < 0.15:
                scene_cut_indices.append(i)
            else:
                continuous_maes.append(mean_diff)
                # In continuous scenes, frame-to-frame full frame MAE must be bounded (< 45.0)
                assert mean_diff < 45.0, f"Abrupt frame flicker at frame {i}->{i+1}: MAE={mean_diff:.2f}"

        # Verified scene cut frames in bad apple segment
        assert len(scene_cut_indices) > 0, "Should detect at least 1 scene cut in 15s sequence"
        assert 195 <= scene_cut_indices[0] <= 205, f"Expected scene cut around frame 200, found {scene_cut_indices}"

    def test_video_container_and_audio_muxing_integrity(self):
        """
        Verifies video container attributes via OpenCV and FFprobe:
        H.264 video stream, AAC stereo audio stream, duration in [5, 15] seconds, FPS >= 12.
        """
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        assert cap.isOpened()
        fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = fcount / max(fps, 1.0)
        cap.release()

        assert fcount == 225, f"Expected 225 frames, got {fcount}"
        assert abs(fps - 15.0) < 0.1, f"Expected 15.0 FPS, got {fps}"
        assert w == 512 and h == 384, f"Expected 512x384 resolution, got {w}x{h}"
        assert 14.8 <= duration <= 15.2, f"Expected ~15.0s duration, got {duration:.2f}s"

        if shutil.which("ffprobe"):
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=codec_type,codec_name,channels",
                "-of", "json", str(VIDEO_PATH)
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            info = json.loads(res.stdout)
            streams = info.get("streams", [])
            types = {s["codec_type"]: s for s in streams}

            assert "video" in types, "H.264 Video stream missing"
            assert types["video"]["codec_name"] == "h264", f"Unexpected video codec {types['video']['codec_name']}"

            assert "audio" in types, "AAC Audio stream missing"
            assert types["audio"]["codec_name"] == "aac", f"Unexpected audio codec {types['audio']['codec_name']}"
            assert int(types["audio"].get("channels", 1)) == 2, "Expected stereo audio"
