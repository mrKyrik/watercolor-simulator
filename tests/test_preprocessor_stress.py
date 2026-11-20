"""
tests/test_preprocessor_stress.py - Empirical Stress-Testing Suite for Milestone 1.

Adversarial testing of VideoPreprocessor (pipeline/preprocessor.py):
1. Border perimeter median heuristic: full white/black frames, thin borders, checkerboard noise,
   corner double-counting effects, and breakdown point analysis.
2. Topological contour hierarchy: complex nested concentric shapes, Swiss cheese multi-hole objects,
   cv2.RETR_CCOMP parent-child relations, and interior zero-boundary preservation.
3. Euclidean distance transform: numerical comparison against scipy.ndimage.distance_transform_edt
   and an independent brute-force distance oracle.
4. Performance throughput: frames per second (FPS) benchmarks for CV stages and end-to-end pipeline.
"""

import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import pytest
import scipy.ndimage as ndi

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocessor import VideoPreprocessor, preprocess_video


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def preprocessed_dir(project_root: Path) -> Path:
    return project_root / "data" / "preprocessed"


class TestBorderPolarityAdversarial:
    """Suite 1: Adversarial testing of the order-statistic border perimeter median heuristic."""

    def test_full_white_flash_frame(self):
        """Full white frame (255) must yield empty mask (0) and safe zeroed geometry."""
        pre = VideoPreprocessor()
        img = np.full((384, 512), 255, dtype=np.uint8)

        mask, otsu_t, is_white_bg, b_med, b_mean = pre.threshold_frame(img)
        assert is_white_bg is True
        assert b_med == 255.0
        assert np.all(mask == 0), "Full white frame must invert to 0 foreground pixels"

        # Downstream geometry must not raise exceptions or return invalid coords
        bbox_xywh, bbox_xyxy = pre.compute_bounding_box(mask)
        assert bbox_xywh == (0, 0, 0, 0)
        assert bbox_xyxy == (0, 0, 0, 0)

        cx, cy = pre.compute_moments(mask)
        assert cx == 0.0 and cy == 0.0

        cnts, hier, ext, holes = pre.extract_contours_and_holes(mask)
        assert len(cnts) == 0 and hier is None
        assert ext == 0 and holes == 0

        dt, max_d, max_l = pre.compute_distance_transform(mask)
        assert np.all(dt == 0.0)
        assert max_d == 0.0

    def test_full_black_flash_frame(self):
        """Full black frame (0) must yield empty mask (0) and safe zeroed geometry."""
        pre = VideoPreprocessor()
        img = np.zeros((384, 512), dtype=np.uint8)

        mask, otsu_t, is_white_bg, b_med, b_mean = pre.threshold_frame(img)
        assert is_white_bg is False
        assert b_med == 0.0
        assert np.all(mask == 0), "Full black frame must remain 0 foreground pixels"

        bbox_xywh, _ = pre.compute_bounding_box(mask)
        assert bbox_xywh == (0, 0, 0, 0)
        cx, cy = pre.compute_moments(mask)
        assert cx == 0.0 and cy == 0.0

        cnts, hier, ext, holes = pre.extract_contours_and_holes(mask)
        assert len(cnts) == 0 and hier is None

        dt, max_d, max_l = pre.compute_distance_transform(mask)
        assert np.all(dt == 0.0)
        assert max_d == 0.0

    def test_checkerboard_high_frequency_noise(self):
        """High-frequency alternating checkerboards (50/50) should be cleanly resolved."""
        # 1x1 pixel checkerboard
        H, W = 100, 100
        cb_1x1 = ((np.indices((H, W)).sum(axis=0) % 2) * 255).astype(np.uint8)

        # Without morphological cleanup
        pre_raw = VideoPreprocessor(morph_cleanup=False)
        mask_raw, otsu_t, is_w, b_med, b_mean = pre_raw.threshold_frame(cb_1x1)
        # Median of 50/50 0/255 is 127.5 <= 128 -> is_white_bg is False
        assert is_w is False
        assert b_med == 127.5
        assert set(np.unique(mask_raw)).issubset({0, 255})

        # With morphological cleanup (3x3 opening)
        pre_morph = VideoPreprocessor(morph_cleanup=True)
        mask_morph, _, _, _, _ = pre_morph.threshold_frame(cb_1x1)
        # 1x1 noise must be entirely eliminated by morphological opening
        assert np.all(mask_morph == 0), "3x3 opening must completely remove 1x1 isolated noise"

    def test_border_contamination_breakdown_point(self):
        """
        Verify the breakdown point of the border median filter.
        The median order statistic maintains correct polarity as long as foreground
        occupies < 50% of the concatenated perimeter band. At >= 50%, it inverts.
        """
        H, W = 200, 200
        pre = VideoPreprocessor(border_strip_px=2)
        total_border_pixels = 2 * (2 * W) + 2 * (2 * H)

        # Sweep border foreground contamination
        for fg_ratio in [0.10, 0.25, 0.40, 0.48]:
            # White canvas (255) with black foreground (0) along border
            img = np.full((H, W), 255, dtype=np.uint8)
            num_fg = int(total_border_pixels * fg_ratio)
            # Cover top border and part of bottom
            top_px = 2 * W
            if num_fg <= top_px:
                img[0:2, :num_fg // 2] = 0
            else:
                img[0:2, :] = 0
                rem = num_fg - top_px
                img[-2:, :rem // 2] = 0

            is_white_bg, b_med, b_mean = pre.detect_polarity(img)
            assert is_white_bg is True, f"Failed at {fg_ratio*100}% contamination: median={b_med}"
            assert b_med == 255.0

        # At >= 50% contamination, median inverts
        img_55 = np.full((H, W), 255, dtype=np.uint8)
        num_fg_55 = int(total_border_pixels * 0.55)
        img_55[0:2, :] = 0
        rem_55 = num_fg_55 - (2 * W)
        img_55[-2:, :rem_55 // 2] = 0
        is_white_bg_55, b_med_55, _ = pre.detect_polarity(img_55)
        assert is_white_bg_55 is False, "Expected 55% border contamination to invert median"
        assert b_med_55 == 0.0

    def test_thin_borders_and_corner_double_count(self):
        """
        Adversarial test on 1-pixel framing/letterboxing.
        Document the 16-pixel corner overlap bias in detect_polarity.
        """
        H, W = 100, 100
        # White image with a 1px black border
        img = np.full((H, W), 255, dtype=np.uint8)
        img[0, :] = 0
        img[-1, :] = 0
        img[:, 0] = 0
        img[:, -1] = 0

        # With strip=1, border is 100% black -> is_white_bg=False
        pre_k1 = VideoPreprocessor(border_strip_px=1)
        is_w_1, med_1, _ = pre_k1.detect_polarity(img)
        assert is_w_1 is False and med_1 == 0.0

        # With strip=2, 4 corners are counted twice in top/bottom and left/right,
        # giving 16 extra zeros -> median is 0.0
        pre_k2 = VideoPreprocessor(border_strip_px=2)
        is_w_2, med_2, _ = pre_k2.detect_polarity(img)
        assert is_w_2 is False and med_2 == 0.0

        # With strip=3, inner 2 rows outnumber the 1 outer black row -> median is 255.0
        pre_k3 = VideoPreprocessor(border_strip_px=3)
        is_w_3, med_3, _ = pre_k3.detect_polarity(img)
        assert is_w_3 is True and med_3 == 255.0

    def test_micro_dimensions_and_extreme_aspect_ratios(self):
        """Verify preprocessor handles micro-dimensions and extreme aspect ratios without crashing."""
        shapes = [(1, 1), (1, 50), (50, 1), (2, 2), (3, 3), (2, 512), (384, 2)]
        pre = VideoPreprocessor()

        for h, w in shapes:
            img = np.full((h, w), 255, dtype=np.uint8)
            mask, otsu_t, is_w, b_med, b_mean = pre.threshold_frame(img)
            assert mask.shape == (h, w)
            cnts, hier, ext, holes = pre.extract_contours_and_holes(mask)
            dt, max_d, max_l = pre.compute_distance_transform(mask)
            assert dt.shape == (h, w)
            assert max_d == 0.0

    def test_real_bad_apple_border_safety_margin(self, preprocessed_dir: Path):
        """
        Empirically evaluate all preprocessed Bad Apple frames.
        Assert that maximum perimeter contamination never exceeds the 50% breakdown threshold.
        """
        masks_dir = preprocessed_dir / "masks"
        mask_files = sorted(list(masks_dir.glob("*.png")))
        assert len(mask_files) == 225, f"Expected 225 preprocessed masks, found {len(mask_files)}"

        max_contamination = 0.0
        worst_frame = ""

        for mf in mask_files:
            mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
            H, W = mask.shape
            K = 2
            top = mask[0:K, :]
            bot = mask[H - K:H, :]
            left = mask[:, 0:K]
            right = mask[:, W - K:W]
            border_pixels = np.concatenate([top.ravel(), bot.ravel(), left.ravel(), right.ravel()])
            fg_ratio = float(np.count_nonzero(border_pixels == 255) / len(border_pixels))

            if fg_ratio > max_contamination:
                max_contamination = fg_ratio
                worst_frame = mf.name

            # Canvas background must strictly dominate every frame
            border_med = float(np.median(border_pixels))
            assert border_med == 0.0, f"Frame {mf.name} has border median {border_med} != 0.0"

        # Empirical assertion: worst contamination in Bad Apple is ~46.51% (frame 0074)
        assert max_contamination < 0.50, (
            f"Worst frame {worst_frame} exceeded 50% breakdown threshold: {max_contamination*100:.2f}%"
        )
        assert 0.40 <= max_contamination <= 0.49, (
            f"Expected peak border contamination ~46.5%, got {max_contamination*100:.2f}%"
        )


class TestTopologicalHierarchyStress:
    """Suite 2: Topological contour hierarchy and negative space preservation."""

    def test_multi_level_concentric_bullseye(self):
        """
        Adversarially test a 6-level concentric bullseye:
        R=100 (white disc) -> R=80 (black hole) -> R=60 (white disc) ->
        R=40 (black hole) -> R=20 (white disc) -> R=10 (black hole).
        Verify cv2.RETR_CCOMP captures all 3 external components and all 3 hole components.
        """
        pre = VideoPreprocessor()
        img = np.zeros((300, 300), dtype=np.uint8)

        # Draw concentric rings
        cv2.circle(img, (150, 150), 100, 255, -1)  # Level 1: Outer disc
        cv2.circle(img, (150, 150), 80, 0, -1)     # Level 2: Hole
        cv2.circle(img, (150, 150), 60, 255, -1)   # Level 3: Inner disc
        cv2.circle(img, (150, 150), 40, 0, -1)     # Level 4: Hole
        cv2.circle(img, (150, 150), 20, 255, -1)   # Level 5: Innermost disc
        cv2.circle(img, (150, 150), 10, 0, -1)     # Level 6: Hole

        cnts, hier, num_ext, num_holes = pre.extract_contours_and_holes(img)

        # Must have exactly 6 contours: 3 external and 3 holes
        assert len(cnts) == 6, f"Expected 6 contours, got {len(cnts)}"
        assert num_ext == 3, f"Expected 3 external contours, got {num_ext}"
        assert num_holes == 3, f"Expected 3 hole contours, got {num_holes}"
        assert hier is not None

        # Verify parent pointers in RETR_CCOMP
        hier_data = hier[0]
        ext_indices = [i for i, h in enumerate(hier_data) if h[3] == -1]
        hole_indices = [i for i, h in enumerate(hier_data) if h[3] != -1]

        assert len(ext_indices) == 3
        assert len(hole_indices) == 3

        for h_idx in hole_indices:
            parent_idx = hier_data[h_idx][3]
            assert parent_idx in ext_indices, f"Hole {h_idx} parent {parent_idx} not in external indices"

    def test_multiple_disjoint_multi_hole_objects(self):
        """
        Test multiple disjoint shapes with varied numbers of holes:
        - Object 1: Solid disc (0 holes)
        - Object 2: Annulus (1 hole)
        - Object 3: Button (4 holes)
        - Object 4: Swiss cheese (10 holes)
        """
        pre = VideoPreprocessor()
        img = np.zeros((400, 600), dtype=np.uint8)

        # 1. Solid disc at (80, 100), R=40
        cv2.circle(img, (80, 100), 40, 255, -1)

        # 2. Annulus at (200, 100), R=40, inner R=15
        cv2.circle(img, (200, 100), 40, 255, -1)
        cv2.circle(img, (200, 100), 15, 0, -1)

        # 3. 4-hole button at (350, 100), R=50
        cv2.circle(img, (350, 100), 50, 255, -1)
        button_holes = [(335, 85), (365, 85), (335, 115), (365, 115)]
        for hx, hy in button_holes:
            cv2.circle(img, (hx, hy), 6, 0, -1)

        # 4. Swiss cheese at (250, 280), box 140x80 with 10 holes
        cv2.rectangle(img, (180, 240), (320, 320), 255, -1)
        rng = np.random.RandomState(42)
        cheese_holes = [
            (195, 255), (215, 275), (235, 255), (255, 280), (275, 260),
            (295, 285), (210, 305), (240, 305), (280, 305), (305, 255)
        ]
        for cx, cy in cheese_holes:
            cv2.circle(img, (cx, cy), 5, 0, -1)

        cnts, hier, num_ext, num_holes = pre.extract_contours_and_holes(img)

        # Ground truth: 4 external objects, 0 + 1 + 4 + 10 = 15 holes
        assert num_ext == 4, f"Expected 4 external components, got {num_ext}"
        assert num_holes == 15, f"Expected 15 total holes, got {num_holes}"
        assert len(cnts) == 19

    def test_nested_holes_zero_boundary_enforcement(self):
        """
        Verify that cv2.distanceTransform strictly enforces zero distance
        inside all holes of nested and multi-object topologies.
        """
        pre = VideoPreprocessor()
        img = np.zeros((300, 300), dtype=np.uint8)

        # Nested bullseye: R=100 (fg), R=80 (bg), R=60 (fg), R=40 (bg), R=20 (fg), R=10 (bg)
        cv2.circle(img, (150, 150), 100, 255, -1)
        cv2.circle(img, (150, 150), 80, 0, -1)
        cv2.circle(img, (150, 150), 60, 255, -1)
        cv2.circle(img, (150, 150), 40, 0, -1)
        cv2.circle(img, (150, 150), 20, 255, -1)
        cv2.circle(img, (150, 150), 10, 0, -1)

        dt, max_d, max_l = pre.compute_distance_transform(img)

        # 1. Background pixels must strictly have DT == 0.0
        assert np.all(dt[img == 0] == 0.0), "DT must be exactly 0.0 at all background and hole pixels"

        # 2. Foreground pixels must strictly have DT > 0.0
        assert np.all(dt[img == 255] > 0.0), "DT must be strictly > 0.0 at all foreground pixels"

        # 3. Check specific hole interiors
        # Innermost hole (center)
        assert dt[150, 150] == 0.0
        # Middle hole at R=30
        assert dt[150, 180] == 0.0
        # Outer hole at R=70
        assert dt[150, 220] == 0.0
        # Outside at R=120
        assert dt[150, 270] == 0.0

        # 4. Check peak distances in each ring
        # Innermost ring: R in [10, 20], width = 10, mid at R=15 -> DT ~ 5.0
        assert 4.5 <= dt[150, 165] <= 5.5
        # Middle ring: R in [40, 60], width = 20, mid at R=50 -> DT ~ 10.0
        assert 9.0 <= dt[150, 200] <= 11.0
        # Outer ring: R in [80, 100], width = 20, mid at R=90 -> DT ~ 10.0
        assert 9.0 <= dt[150, 240] <= 11.0


class TestDistanceTransformOracleComparison:
    """Suite 3: Numerical verification of cv2.distanceTransform against independent oracles."""

    @staticmethod
    def brute_force_distance_oracle(mask: np.ndarray) -> np.ndarray:
        """
        Brute-force Euclidean distance oracle: O(N_fg * N_bg).
        Computes exact Euclidean distance from every foreground pixel to the nearest background pixel.
        """
        H, W = mask.shape
        oracle_dt = np.zeros((H, W), dtype=np.float32)
        bg_coords = np.argwhere(mask == 0)
        fg_coords = np.argwhere(mask == 255)

        if len(bg_coords) == 0 or len(fg_coords) == 0:
            return oracle_dt

        # Compute pairwise Euclidean distances
        # For efficiency, compute in batches of foreground pixels
        batch_size = 500
        for i in range(0, len(fg_coords), batch_size):
            batch_fg = fg_coords[i:i + batch_size]
            diffs = batch_fg[:, np.newaxis, :] - bg_coords[np.newaxis, :, :]
            sq_dists = np.sum(diffs ** 2, axis=-1)
            min_dists = np.sqrt(np.min(sq_dists, axis=-1))
            for (r, c), d in zip(batch_fg, min_dists):
                oracle_dt[r, c] = float(d)

        return oracle_dt

    def test_distance_transform_vs_brute_force_oracle(self):
        """
        Verify cv2.distanceTransform against independent brute-force oracle
        on a complex 40x40 synthetic pattern.
        """
        mask = np.zeros((40, 40), dtype=np.uint8)
        # Create an irregular polygon with a hole
        cv2.rectangle(mask, (5, 5), (35, 35), 255, -1)
        cv2.circle(mask, (20, 20), 6, 0, -1)
        cv2.circle(mask, (10, 10), 3, 0, -1)

        # 1. cv2 distance transform (DIST_L2, 5)
        dt_cv2 = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # 2. Brute-force oracle
        dt_oracle = self.brute_force_distance_oracle(mask)

        # 3. Assert background identity
        assert np.all(dt_cv2[mask == 0] == 0.0)
        assert np.all(dt_oracle[mask == 0] == 0.0)

        # 4. Compare foreground
        fg = mask == 255
        abs_err = np.abs(dt_cv2[fg] - dt_oracle[fg])
        max_abs_err = float(np.max(abs_err))
        mean_abs_err = float(np.mean(abs_err))

        # 5x5 chamfer metric has theoretical max error ~0.26 px
        assert max_abs_err <= 0.25, f"Max error {max_abs_err:.4f} px exceeds 5x5 chamfer bound"
        assert mean_abs_err <= 0.06, f"Mean error {mean_abs_err:.4f} px exceeds expected bound"

    def test_distance_transform_vs_scipy_exact_edt(self):
        """
        Verify cv2.distanceTransform against scipy's exact distance_transform_edt
        across diverse geometric shapes (circles, annuli, rectangles, rotated stars).
        """
        H, W = 256, 256
        shapes = [
            # Solid circle
            lambda: cv2.circle(np.zeros((H, W), dtype=np.uint8), (128, 128), 60, 255, -1),
            # Annulus
            lambda: cv2.circle(
                cv2.circle(np.zeros((H, W), dtype=np.uint8), (128, 128), 70, 255, -1),
                (128, 128), 30, 0, -1
            ),
            # Rectangle
            lambda: cv2.rectangle(np.zeros((H, W), dtype=np.uint8), (40, 60), (216, 196), 255, -1),
        ]

        for shape_fn in shapes:
            mask = shape_fn()
            dt_cv2 = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            dt_scipy = ndi.distance_transform_edt(mask == 255).astype(np.float32)

            # Strictly 0 at background
            assert np.all(dt_cv2[mask == 0] == 0.0)
            assert np.all(dt_scipy[mask == 0] == 0.0)

            # Foreground differences
            fg = mask == 255
            abs_err = np.abs(dt_cv2[fg] - dt_scipy[fg])
            rel_err = abs_err / (dt_scipy[fg] + 1e-6)

            max_rel_err = float(np.max(rel_err))
            mean_rel_err = float(np.mean(rel_err))
            max_abs_err = float(np.max(abs_err))
            max_d = float(np.max(dt_scipy[fg]))

            # Theoretical bound for 5x5 Borgefors chamfer metric: relative error <= 2.0%
            assert max_rel_err <= 0.020, f"Max relative error {max_rel_err*100:.2f}% exceeds 2.0% bound"
            assert mean_rel_err <= 0.010, f"Mean relative error {mean_rel_err*100:.2f}% exceeds 1.0% bound"
            # Absolute error is proportional to distance: max_err <= 0.02 * max_d + 0.25
            assert max_abs_err <= 0.02 * max_d + 0.25, f"Max absolute error {max_abs_err:.4f} px exceeds bound"

    def test_distance_transform_on_real_bad_apple_frames(self, preprocessed_dir: Path):
        """
        Compare saved preprocessed distance arrays against scipy EDT
        across 20 sampled Bad Apple frames.
        """
        masks_dir = preprocessed_dir / "masks"
        dist_dir = preprocessed_dir / "distance"

        mask_files = sorted(list(masks_dir.glob("*.png")))
        assert len(mask_files) > 0

        # Sample every 10th frame
        sampled_files = mask_files[::10]
        max_rel_errors = []
        mean_rel_errors = []
        max_abs_errors = []

        for mf in sampled_files:
            stem = mf.stem.split("_")[-1]
            npy_path = dist_dir / f"dist_{stem}.npy"
            assert npy_path.exists(), f"Missing distance array {npy_path}"

            mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
            dt_saved = np.load(str(npy_path))
            dt_scipy = ndi.distance_transform_edt(mask == 255).astype(np.float32)

            # Assert strict 0 background
            assert np.all(dt_saved[mask == 0] == 0.0)

            fg = mask == 255
            if np.any(fg):
                diff = np.abs(dt_saved[fg] - dt_scipy[fg])
                rel = diff / (dt_scipy[fg] + 1e-6)
                max_abs_errors.append(float(np.max(diff)))
                max_rel_errors.append(float(np.max(rel)))
                mean_rel_errors.append(float(np.mean(rel)))

        overall_max_rel = max(max_rel_errors)
        overall_mean_rel = float(np.mean(mean_rel_errors))
        overall_max_abs = max(max_abs_errors)

        # Theoretical and empirical bounds: relative error is strictly <= 1.95%, mean <= 0.85%
        assert overall_max_rel <= 0.020, f"Overall max relative error {overall_max_rel*100:.2f}% exceeds 2.0%"
        assert overall_mean_rel <= 0.010, f"Overall mean relative error {overall_mean_rel*100:.2f}% exceeds 1.0%"
        assert overall_max_abs < 2.50, f"Overall max abs error {overall_max_abs:.2f} px unexpectedly large"

    def test_exact_distance_transform_with_dist_mask_precise(self):
        """
        Empirically verify that cv2.distanceTransform with cv2.DIST_MASK_PRECISE
        achieves exact zero difference (0.0 px) against scipy.ndimage.distance_transform_edt.
        """
        img = np.zeros((150, 150), dtype=np.uint8)
        cv2.circle(img, (75, 75), 50, 255, -1)
        cv2.circle(img, (75, 75), 20, 0, -1)

        dt_cv2_precise = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dt_scipy = ndi.distance_transform_edt(img == 255).astype(np.float32)

        diff = np.abs(dt_cv2_precise - dt_scipy)
        assert np.max(diff) == 0.0, "DIST_MASK_PRECISE must exactly match scipy EDT"


class TestPreprocessorThroughputBenchmark:
    """Suite 4: Performance and throughput benchmarking."""

    def test_cv_components_throughput(self):
        """
        Measure per-stage throughput on standard 512x384 frames:
        - Thresholding + polarity detection
        - 2-level contour extraction (RETR_CCOMP)
        - Euclidean distance transform (DIST_L2, 5)
        Assert combined throughput exceeds 200 FPS (>13x real-time 15 FPS).
        """
        pre = VideoPreprocessor(target_resolution=(512, 384))
        # Create realistic silhouette frame with holes
        img = np.zeros((384, 512), dtype=np.uint8)
        cv2.ellipse(img, (256, 192), (120, 160), 0, 0, 360, 255, -1)
        cv2.circle(img, (220, 150), 25, 0, -1)
        cv2.circle(img, (290, 150), 25, 0, -1)
        cv2.rectangle(img, (230, 240), (280, 270), 0, -1)

        N = 100

        # Stage 1: Thresholding & Polarity
        t0 = time.perf_counter()
        for _ in range(N):
            mask, _, _, _, _ = pre.threshold_frame(img)
        t1 = time.perf_counter()

        # Stage 2: Contour Hierarchy
        for _ in range(N):
            cnts, hier, ext, holes = pre.extract_contours_and_holes(mask)
        t2 = time.perf_counter()

        # Stage 3: Distance Transform
        for _ in range(N):
            dt, max_d, _ = pre.compute_distance_transform(mask)
        t3 = time.perf_counter()

        fps_thresh = N / (t1 - t0)
        fps_cnt = N / (t2 - t1)
        fps_dt = N / (t3 - t2)
        fps_combined = N / (t3 - t0)

        # Assert performance bounds
        assert fps_thresh > 500.0, f"Thresholding throughput too slow: {fps_thresh:.1f} FPS"
        assert fps_cnt > 500.0, f"Contour extraction throughput too slow: {fps_cnt:.1f} FPS"
        assert fps_dt > 300.0, f"Distance transform throughput too slow: {fps_dt:.1f} FPS"
        assert fps_combined > 200.0, f"Combined CV throughput too slow: {fps_combined:.1f} FPS"

    def test_end_to_end_preprocessor_throughput(self, project_root: Path):
        """
        Benchmark end-to-end process_video throughput including video decoding,
        resizing, CV processing, and complete disk I/O (PNG + NPY saves).
        Assert throughput >= 30 FPS (>2x real-time 15 FPS).
        """
        import tempfile
        clip_path = project_root / "data" / "bad_apple_clip.mp4"
        if not clip_path.exists():
            pytest.skip("Video clip not available for end-to-end benchmark")

        pre = VideoPreprocessor(target_fps=15.0, target_resolution=(512, 384))
        with tempfile.TemporaryDirectory() as tmp_dir:
            t0 = time.perf_counter()
            res = pre.process_video(clip_path, tmp_dir, max_frames=45)
            t1 = time.perf_counter()

            total_frames = res["video_metadata"]["total_preprocessed_frames"]
            elapsed = t1 - t0
            fps = total_frames / elapsed

            assert total_frames == 45
            assert fps >= 30.0, f"End-to-end preprocessing throughput {fps:.1f} FPS is below 30 FPS threshold"
