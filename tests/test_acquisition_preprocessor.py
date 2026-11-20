"""
tests/test_acquisition_preprocessor.py - Test Suite for Milestone 1.

Validates:
1. Video Acquisition: clip duration (10–20s), file size, OpenCV readability, audio extraction.
2. Frame Preprocessor: downsampled frame count, strict mask bimodality in {0, 255},
   polarity normalization across scene cuts, negative space (hole) preservation via RETR_CCOMP.
3. Distance Transform: Euclidean L2 distance properties (float32, 0 at background, >0 inside silhouette).
4. Metadata Manifest: schema compliance, monotonic timestamps, bounding box bounds, path integrity.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from pipeline.acquisition import (
    VideoAcquisition,
    VideoAcquisitionConfig,
    parse_time_to_seconds,
    parse_section_range,
    acquire_bad_apple,
)
from pipeline.preprocessor import (
    VideoPreprocessor,
    preprocess_video,
)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def clip_path(project_root: Path) -> Path:
    target = project_root / "data" / "bad_apple_clip.mp4"
    if not target.exists():
        survey_clip = project_root / ".agents" / "teamwork" / "teamwork_explorer_survey_2" / "test_clip.mp4"
        if survey_clip.exists():
            return survey_clip
    return target


@pytest.fixture(scope="session")
def audio_path(project_root: Path) -> Path:
    return project_root / "data" / "bad_apple_audio.aac"


@pytest.fixture(scope="session")
def preprocessed_dir(project_root: Path, clip_path: Path, audio_path: Path) -> Path:
    target = project_root / "data" / "preprocessed"
    manifest_file = target / "metadata.json"
    if not manifest_file.exists():
        if clip_path.exists():
            preprocess_video(
                video_path=clip_path,
                output_dir=target,
                audio_path=audio_path if audio_path.exists() else None,
                target_fps=15.0
            )
    return target


@pytest.fixture
def synthetic_ring_mask() -> np.ndarray:
    """Deterministic ring fixture with outer radius 40, inner hole radius 15."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 40, 255, -1)
    cv2.circle(mask, (100, 100), 15, 0, -1)
    return mask


class TestVideoAcquisition:
    """Suite 1: Video clip acquisition, range parsing, and audio demuxing."""

    def test_parse_time_and_section_range(self):
        # Seconds format
        assert parse_time_to_seconds(30) == 30.0
        assert parse_time_to_seconds("30.5") == 30.5
        # MM:SS format
        assert parse_time_to_seconds("00:30") == 30.0
        assert parse_time_to_seconds("01:15") == 75.0
        # HH:MM:SS format
        assert parse_time_to_seconds("00:01:15") == 75.0

        # Section range parsing
        s, e = parse_section_range("00:30-00:45")
        assert s == 30.0 and e == 45.0
        assert e - s == 15.0

        s2, e2 = parse_section_range("*01:10-01:25")
        assert s2 == 70.0 and e2 == 85.0

        s3, e3 = parse_section_range((30.0, 45.0))
        assert s3 == 30.0 and e3 == 45.0

        # Duration enforcement (10.0 <= duration <= 20.0)
        with pytest.raises(ValueError, match="Milestone requirements strictly enforce"):
            parse_section_range("00:00-00:05")  # 5s < 10s

        with pytest.raises(ValueError, match="Milestone requirements strictly enforce"):
            parse_section_range("00:00-00:35")  # 35s > 20s

        with pytest.raises(ValueError, match="End time"):
            parse_section_range("00:45-00:30")  # End <= start

    def test_video_clip_duration_and_validity(self, clip_path: Path):
        assert clip_path.exists(), f"Video clip file not found: {clip_path}"
        assert clip_path.stat().st_size > 500 * 1024, "Video clip file size < 500 KB"

        cap = cv2.VideoCapture(str(clip_path))
        assert cap.isOpened(), "Could not open video file via cv2.VideoCapture"
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        assert fps > 0, "FPS must be greater than 0"
        duration = total_frames / fps
        assert 10.0 <= duration <= 20.0, f"Duration {duration:.2f}s is outside required [10, 20] range"
        assert w > 0 and h > 0, "Invalid frame dimensions"
        aspect_ratio = w / h
        assert abs(aspect_ratio - (4.0 / 3.0)) < 0.15, f"Expected ~4:3 aspect ratio, got {aspect_ratio:.2f}"

    def test_audio_file_exists_and_valid(self, audio_path: Path):
        assert audio_path.exists(), f"Audio file not found: {audio_path}"
        assert audio_path.stat().st_size > 20 * 1024, "Audio file is suspiciously small (< 20 KB)"

    def test_media_probing(self, clip_path: Path, audio_path: Path):
        acq = VideoAcquisition()
        meta = acq.probe_video(clip_path, audio_path)
        assert meta.video_path == clip_path
        assert meta.audio_path == audio_path
        assert 10.0 <= meta.duration_sec <= 20.0
        assert meta.fps >= 24.0
        assert meta.width == 512
        assert meta.height == 384
        assert meta.total_frames >= 240
        assert meta.file_size_bytes > 500 * 1024

    def test_local_cache_reuse(self, clip_path: Path):
        acq = VideoAcquisition(output_dir=clip_path.parent)
        cached_path = acq.fetch_clip(force_download=False)
        assert cached_path == clip_path
        assert cached_path.exists()


class TestMaskBimodalityAndPolarity:
    """Suite 2: Mask bimodality, perimeter polarity detection, and flash frame handling."""

    def test_synthetic_polarity_white_background(self):
        pre = VideoPreprocessor()
        # White background with black circle in center
        img = np.full((200, 200), 255, dtype=np.uint8)
        cv2.circle(img, (100, 100), 40, 0, -1)

        is_white_bg, b_med, b_mean = pre.detect_polarity(img)
        assert is_white_bg is True
        assert b_med == 255.0

        norm_mask, otsu_t, is_white, _, _ = pre.threshold_frame(img)
        assert is_white is True
        # Foreground circle must be 255, background must be 0
        assert norm_mask[100, 100] == 255
        assert norm_mask[10, 10] == 0
        assert set(np.unique(norm_mask)).issubset({0, 255})

    def test_synthetic_polarity_black_background(self):
        pre = VideoPreprocessor()
        # Black background with white circle in center
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (100, 100), 40, 255, -1)

        is_white_bg, b_med, b_mean = pre.detect_polarity(img)
        assert is_white_bg is False
        assert b_med == 0.0

        norm_mask, otsu_t, is_white, _, _ = pre.threshold_frame(img)
        assert is_white is False
        # Foreground circle must be 255, background must be 0
        assert norm_mask[100, 100] == 255
        assert norm_mask[10, 10] == 0
        assert set(np.unique(norm_mask)).issubset({0, 255})

    def test_empty_flash_frames_handled_gracefully(self):
        pre = VideoPreprocessor()
        # All-white frame
        white_img = np.full((100, 100), 255, dtype=np.uint8)
        mask_w, _, _, _, _ = pre.threshold_frame(white_img)
        assert np.all(mask_w == 0)

        # All-black frame
        black_img = np.zeros((100, 100), dtype=np.uint8)
        mask_b, _, _, _, _ = pre.threshold_frame(black_img)
        assert np.all(mask_b == 0)

        # Safety checks on empty masks
        bbox_xywh, bbox_xyxy = pre.compute_bounding_box(mask_w)
        assert bbox_xywh == (0, 0, 0, 0)
        assert bbox_xyxy == (0, 0, 0, 0)

        cx, cy = pre.compute_moments(mask_w)
        assert cx == 0.0 and cy == 0.0

        cnts, hier, n_ext, n_holes = pre.extract_contours_and_holes(mask_w)
        assert len(cnts) == 0
        assert hier is None
        assert n_ext == 0 and n_holes == 0

    def test_preprocessed_masks_strictly_bimodal(self, preprocessed_dir: Path):
        masks_dir = preprocessed_dir / "masks"
        assert masks_dir.exists(), f"Masks directory not found: {masks_dir}"
        mask_files = sorted(list(masks_dir.glob("*.png")))
        assert len(mask_files) > 0, "No mask files found in masks directory"

        # Check all frames in sequence for strict bimodality
        for mf in mask_files:
            mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
            assert mask is not None, f"Failed to read mask {mf}"
            unique_vals = set(np.unique(mask))
            assert unique_vals.issubset({0, 255}), (
                f"Mask {mf.name} contains non-binary pixel values: {unique_vals}"
            )

    def test_preprocessed_masks_border_polarity_consistency(self, preprocessed_dir: Path):
        masks_dir = preprocessed_dir / "masks"
        mask_files = sorted(list(masks_dir.glob("*.png")))
        # Across all frames, background canvas must dominate the perimeter (median == 0.0)
        for mf in mask_files:
            mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
            top = mask[0, :]
            bot = mask[-1, :]
            left = mask[:, 0]
            right = mask[:, -1]
            border = np.concatenate([top, bot, left, right])
            border_median = float(np.median(border))
            border_mean = float(np.mean(border))
            assert border_median == 0.0, (
                f"Mask {mf.name} border median is {border_median}. Foreground/background inverted!"
            )
            assert border_mean < 128.0, (
                f"Mask {mf.name} border mean is {border_mean:.2f} (>= 128.0). Canvas polarity inverted!"
            )


class TestDistanceTransformAndTopology:
    """Suite 3: Euclidean distance transform L2 properties and RETR_CCOMP hole preservation."""

    def test_synthetic_annulus_hole_preservation(self, synthetic_ring_mask: np.ndarray):
        pre = VideoPreprocessor()
        cnts, hier, num_ext, num_holes = pre.extract_contours_and_holes(synthetic_ring_mask)

        # 1. RETR_CCOMP 2-level hierarchy check
        assert len(cnts) == 2, f"Expected 2 contours (outer + hole), got {len(cnts)}"
        assert num_ext == 1
        assert num_holes == 1
        assert hier is not None
        # Verify hole contour has outer contour as parent
        assert hier[0][1][3] == 0, "Contour 1 parent must be 0 (inner hole of outer ring)"

        # 2. cv2.distanceTransform treats hole as zero-boundary
        dt, max_d, max_l = pre.compute_distance_transform(synthetic_ring_mask)
        assert dt[100, 100] == 0.0, "Distance inside hole center must be exactly 0.0"
        # Ring width = 40 - 15 = 25 px. Max DT should be 25 / 2 = 12.5 px
        assert 11.5 <= max_d <= 13.5, f"Expected peak DT ~12.5 px, got {max_d:.2f}"

    def test_distance_files_valid_and_non_empty(self, preprocessed_dir: Path):
        dist_dir = preprocessed_dir / "distance"
        masks_dir = preprocessed_dir / "masks"
        assert dist_dir.exists(), f"Distance directory not found: {dist_dir}"

        npy_files = sorted(list(dist_dir.glob("*.npy")))
        png_files = sorted(list(dist_dir.glob("*.png")))
        assert len(npy_files) > 0, "No .npy distance files found"
        assert len(npy_files) == len(png_files), "Mismatch between .npy and .png distance files"

        # Validate distance arrays
        for df in npy_files[:30]:
            dt = np.load(str(df))
            assert dt.dtype == np.float32, f"Distance array {df.name} must be float32"
            assert dt.shape == (384, 512), f"Unexpected distance array shape: {dt.shape}"
            assert np.all(dt >= 0.0), f"Distance values in {df.name} cannot be negative"

            # Check background coherence against corresponding mask
            mask_path = masks_dir / f"mask_{df.stem.split('_')[-1]}.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                assert np.all(dt[mask == 0] == 0.0), (
                    f"Distance transform in {df.name} must be 0.0 everywhere mask is background"
                )
                if np.any(mask == 255):
                    assert dt.max() > 0.0, f"Foreground exists in mask but {df.name} max distance is 0.0"

    def test_preprocessed_frames_with_holes(self, preprocessed_dir: Path):
        manifest_path = preprocessed_dir / "metadata.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        summary = manifest["summary_statistics"]
        assert summary["frames_with_holes"] > 0, "Expected sequence to contain frames with interior holes"

        # Inspect frames with detected holes
        hole_frames = [f for f in manifest["frames"] if f["topology"]["has_holes"]]
        assert len(hole_frames) > 0

        # Sample check first 5 frames with holes
        for hf in hole_frames[:5]:
            assert hf["topology"]["hole_contours"] > 0
            mask_rel = preprocessed_dir / hf["relative_paths"]["mask"]
            dist_rel = preprocessed_dir / hf["relative_paths"]["distance"]
            mask = cv2.imread(str(mask_rel), cv2.IMREAD_GRAYSCALE)
            dt = np.load(str(dist_rel))

            pre = VideoPreprocessor()
            cnts, hier, num_ext, num_holes = pre.extract_contours_and_holes(mask)
            assert num_holes == hf["topology"]["hole_contours"]

            # Extract holes and check interior center has mask == 0 and dt == 0.0
            for i, h in enumerate(hier[0]):
                if h[3] != -1:  # Hole contour
                    hole_cnt = cnts[i]
                    m = cv2.moments(hole_cnt)
                    if m["m00"] > 0:
                        hx = int(round(m["m10"] / m["m00"]))
                        hy = int(round(m["m01"] / m["m00"]))
                        if 0 <= hy < mask.shape[0] and 0 <= hx < mask.shape[1]:
                            assert mask[hy, hx] == 0, f"Hole center ({hx}, {hy}) in frame {hf['frame_idx']} must be 0"
                            assert dt[hy, hx] == 0.0, f"Hole center ({hx}, {hy}) in frame {hf['frame_idx']} must have dt 0.0"


class TestMetadataManifest:
    """Suite 4: Structured manifest metadata.json integrity and consistency."""

    def test_manifest_schema_and_path_integrity(self, preprocessed_dir: Path):
        manifest_path = preprocessed_dir / "metadata.json"
        assert manifest_path.exists(), f"metadata.json missing at {manifest_path}"

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "video_metadata" in data
        assert "summary_statistics" in data
        assert "frames" in data

        v_meta = data["video_metadata"]
        assert v_meta["source_fps"] == 30.0
        assert v_meta["target_fps"] == 15.0
        assert v_meta["target_resolution"] == [512, 384]
        assert v_meta["total_preprocessed_frames"] == len(data["frames"])

        summary = data["summary_statistics"]
        assert summary["total_polarity_flips"] >= 1
        assert summary["max_interior_thickness_overall"] > 0.0

        prev_timestamp = -1.0
        for item in data["frames"]:
            f_idx = item["frame_idx"]
            ts = item["timestamp_sec"]
            assert ts > prev_timestamp, f"Timestamp at frame {f_idx} is not strictly monotonic"
            prev_timestamp = ts

            # Check relative paths exist on disk
            rel_paths = item["relative_paths"]
            assert (preprocessed_dir / rel_paths["frame"]).exists()
            assert (preprocessed_dir / rel_paths["mask"]).exists()
            assert (preprocessed_dir / rel_paths["distance"]).exists()
            assert (preprocessed_dir / rel_paths["distance_vis"]).exists()

            # Check bounding boxes within bounds
            bbox = item["morphometry"]["character_bbox"]
            assert len(bbox) == 4
            x, y, w, h = bbox
            assert x >= 0 and y >= 0
            assert x + w <= 512 and y + h <= 384

            # Centroid within bounds
            cx, cy = item["morphometry"]["centroid"]
            assert 0.0 <= cx <= 512.0
            assert 0.0 <= cy <= 384.0
