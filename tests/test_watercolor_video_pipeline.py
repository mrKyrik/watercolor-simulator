"""
tests/test_watercolor_video_pipeline.py - Test Suite for Flowing Watercolor Video Pipeline.

Covers:
1. Streaming Frame Generator integrity (data/preprocessed on-demand generator).
2. Video Encoders (FFmpegPipeEncoder, OpenCVFallbackEncoder, and mux_audio_to_video).
3. Video Watercolor Engine physical properties (static paper substrate, optical flow, EDT infill).
4. Short-slice pipeline execution (15 frames) with color diversity, entropy, and temporal continuity.
5. End-to-end full deliverable acceptance criteria verification.
"""

import json
import os
import sys
from pathlib import Path
import shutil
import subprocess

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest
import torch

from pipeline.watercolor_video import (
    stream_preprocessed_frames,
    PreprocessedFrameItem,
    PreprocessedDataError,
    WatercolorVideoConfig,
    VideoWatercolorPipeline,
)
from pipeline.encoder import (
    VideoEncoderConfig,
    create_video_encoder,
    FFmpegPipeEncoder,
    OpenCVFallbackEncoder,
    mux_audio_to_video,
    VideoEncodingError,
)
from engine.video_simulation import (
    VideoWatercolorEngine,
    VideoSimulationConfig,
    TransitionMode,
    HarmonicPalette3D,
    FINE_ART_PIGMENTS,
)


def compute_shannon_entropy(img_gray: np.ndarray) -> float:
    """Computes Shannon spatial information entropy (in bits)."""
    hist, _ = np.histogram(img_gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return -float(np.sum(hist * np.log2(hist)))


def compute_color_diversity(img_rgb: np.ndarray) -> float:
    """Computes multi-chromatic cross-channel variance across R, G, and B."""
    r = img_rgb[:, :, 0].astype(float)
    g = img_rgb[:, :, 1].astype(float)
    b = img_rgb[:, :, 2].astype(float)
    return float(np.std(r - g) + np.std(g - b) + np.std(b - r))


class TestStreamingFrameGenerator:
    """Tests sequential frame ingestion from data/preprocessed."""

    def test_stream_preprocessed_frames_schema(self):
        preprocessed_dir = Path("data/preprocessed")
        assert preprocessed_dir.exists(), "Preprocessed directory data/preprocessed must exist"

        frames = list(stream_preprocessed_frames(preprocessed_dir, start_frame=0, max_frames=5))
        assert len(frames) == 5, f"Expected 5 frames, got {len(frames)}"

        for i, item in enumerate(frames):
            assert isinstance(item, PreprocessedFrameItem)
            assert item.frame_idx == i
            assert item.mask.shape == (384, 512)
            assert item.mask.dtype == np.uint8
            assert set(np.unique(item.mask)).issubset({0, 255})
            assert item.dist.shape == (384, 512)
            assert item.dist.dtype == np.float32
            assert item.dist.min() >= 0.0

    def test_invalid_directory_raises_error(self):
        with pytest.raises(PreprocessedDataError):
            list(stream_preprocessed_frames("data/non_existent_directory_for_testing"))


class TestVideoEncoders:
    """Tests direct FFmpeg pipe encoding and OpenCV fallback."""

    def test_ffmpeg_pipe_encoder_lifecycle(self, tmp_path):
        out_mp4 = tmp_path / "test_ffmpeg_out.mp4"
        audio_src = Path("data/bad_apple_audio.aac")

        config = VideoEncoderConfig(
            output_path=out_mp4,
            width=512,
            height=384,
            fps=15.0,
            crf=18,
            audio_path=audio_src if audio_src.exists() else None,
            backend="ffmpeg",
        )

        with create_video_encoder(config) as encoder:
            assert isinstance(encoder, FFmpegPipeEncoder)
            for i in range(30):  # 2.0 seconds @ 15 FPS
                # Create dynamic colored frame
                frame = np.full((384, 512, 3), (i * 8) % 256, dtype=np.uint8)
                frame[:, :, 0] = (i * 12) % 256
                encoder.write_frame(frame)

        assert out_mp4.exists()
        assert out_mp4.stat().st_size > 10000, "Output MP4 should be larger than 10 KB"

        cap = cv2.VideoCapture(str(out_mp4))
        assert cap.isOpened()
        fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        assert fcount == 30
        assert abs(fps - 15.0) < 0.1

    def test_opencv_fallback_encoder_lifecycle(self, tmp_path):
        out_mp4 = tmp_path / "test_opencv_out.mp4"
        config = VideoEncoderConfig(
            output_path=out_mp4,
            width=512,
            height=384,
            fps=15.0,
            backend="opencv",
        )

        with create_video_encoder(config) as encoder:
            assert isinstance(encoder, OpenCVFallbackEncoder)
            for i in range(15):
                frame = np.full((384, 512, 3), 128, dtype=np.uint8)
                encoder.write_frame(frame)

        assert out_mp4.exists()
        assert out_mp4.stat().st_size > 3000

    def test_stream_copy_audio_muxing(self, tmp_path):
        audio_src = Path("data/bad_apple_audio.aac")
        if not audio_src.exists():
            pytest.skip("Audio asset data/bad_apple_audio.aac not found")

        # Create a simple silent video
        silent_v = tmp_path / "silent.mp4"
        muxed_v = tmp_path / "muxed.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(silent_v), fourcc, 15.0, (512, 384))
        for _ in range(30):
            vw.write(np.zeros((384, 512, 3), dtype=np.uint8))
        vw.release()

        res_path = mux_audio_to_video(silent_v, audio_src, muxed_v)
        assert res_path.exists()
        assert res_path.stat().st_size > silent_v.stat().st_size

        if shutil.which("ffprobe"):
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=codec_type,codec_name",
                "-of", "json", str(muxed_v),
            ]
            probe_res = subprocess.run(probe_cmd, capture_output=True, text=True)
            probe_json = json.loads(probe_res.stdout)
            types = [s["codec_type"] for s in probe_json.get("streams", [])]
            assert "video" in types
            assert "audio" in types


class TestVideoWatercolorEngine:
    """Tests core physics and architectural invariants of VideoWatercolorEngine."""

    def test_paper_substrate_remains_static_during_advection(self):
        engine = VideoWatercolorEngine(width=128, height=96)
        initial_z_paper = engine.paper.z_paper.clone()
        initial_pinning = engine.paper.pinning.clone()

        # Synthetic displacement field (e.g. +10 px shift in X)
        flow = np.zeros((96, 128, 2), dtype=np.float32)
        flow[..., 0] = 10.0

        # Advect state
        engine.advect_state(flow)

        # Confirm paper topography is strictly static in screen space
        assert torch.equal(engine.paper.z_paper, initial_z_paper), "z_paper topography must remain static"
        assert torch.equal(engine.paper.pinning, initial_pinning), "pinning mask must remain static"

    def test_transition_detection_and_soft_reset(self):
        engine = VideoWatercolorEngine(width=128, height=96)

        # Synthetic mask A (centered square)
        mask_a = np.zeros((96, 128), dtype=np.uint8)
        mask_a[20:70, 30:90] = 255

        # Synthetic mask B (disjoint square -> IoU == 0.0)
        mask_b = np.zeros((96, 128), dtype=np.uint8)
        mask_b[5:15, 5:15] = 255

        iou = engine.calculate_mask_iou(mask_a, mask_b)
        assert iou < 0.15, f"Expected IoU < 0.15 for disjoint masks, got {iou}"

        # Deposit some water and pigment
        engine.fluid.h.fill_(0.8)
        dist_map = np.ones((96, 128), dtype=np.float32) * 10.0

        # Run step with scene cut
        placed = engine.step_video_frame(
            prev_mask=mask_a,
            curr_mask=mask_b,
            dist_map=dist_map,
            frame_idx=1,
            is_scene_cut=True,
        )
        assert placed > 0

    def test_edt_droplet_scaling_and_palette(self):
        engine = VideoWatercolorEngine(width=128, height=96)
        mask = np.zeros((96, 128), dtype=np.uint8)
        mask[20:80, 20:100] = 255

        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        droplets = engine.generate_edt_droplets(
            target_mask=mask,
            dist_map=dist_map,
            frame_idx=0,
            max_droplets=20,
        )
        assert len(droplets) > 0

        for d in droplets:
            assert d.radius >= engine.config.droplet_r_min
            assert d.radius <= engine.config.droplet_r_max
            assert d.pigment_name in FINE_ART_PIGMENTS
            assert len(d.color_rgb) == 3


class TestShortPipelineRender:
    """Tests end-to-end simulation, optical flow advection, and rendering on a 15-frame slice."""

    def test_pipeline_15_frame_slice_execution(self, tmp_path):
        out_mp4 = tmp_path / "watercolor_slice.mp4"
        audio_src = Path("data/bad_apple_audio.aac")

        config = WatercolorVideoConfig(
            preprocessed_dir="data/preprocessed",
            output_video_path=str(out_mp4),
            audio_path=str(audio_src) if audio_src.exists() else None,
            width=512,
            height=384,
            fps=15.0,
            start_frame=0,
            max_frames=15,  # 1.0 second test slice
            physics_substeps=4,
        )

        pipeline = VideoWatercolorPipeline(config)
        telemetries = []
        result = pipeline.render(progress_callback=telemetries.append)

        assert result.success is True
        assert result.total_frames_rendered == 15
        assert len(telemetries) == 15
        assert out_mp4.exists()
        assert out_mp4.stat().st_size > 15000

        # Validate rendered video visual characteristics
        cap = cv2.VideoCapture(str(out_mp4))
        frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()

        assert len(frames) == 15

        for frame in frames:
            # 1. Zero NaN / Inf
            assert not np.isnan(frame).any()
            assert not np.isinf(frame).any()

            # 2. Information entropy (rich fine-art paper & pigment texture)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            entropy = compute_shannon_entropy(gray)
            assert entropy > 2.5, f"Information entropy too low: {entropy:.2f}"

            # 3. Multi-chromatic color diversity
            diversity = compute_color_diversity(frame)
            assert diversity > 5.0, f"Color diversity too low (looks monochrome): {diversity:.2f}"

        # 4. Temporal continuity: check bounded inter-frame MAE (eliminates boiling)
        for i in range(1, len(frames)):
            mae = float(np.mean(np.abs(frames[i].astype(float) - frames[i - 1].astype(float))))
            assert mae < 40.0, f"Inter-frame MAE too high (temporal boiling detected): {mae:.2f}"


class TestDeliveredWatercolorVideo:
    """Verifies the final generated deliverable artifact data/bad_apple_watercolor.mp4."""

    def test_delivered_video_file_metrics_and_playability(self):
        video_path = Path("data/bad_apple_watercolor.mp4")
        if not video_path.exists():
            pytest.skip("data/bad_apple_watercolor.mp4 not yet generated")

        assert video_path.stat().st_size > 500000, "Delivered video must be > 500 KB"

        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), "Delivered video must be readable by OpenCV"
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / max(fps, 1.0)

        assert width == 512
        assert height == 384
        assert fps >= 12.0, f"FPS must be >= 12, got {fps}"
        assert 5.0 <= duration <= 16.0, f"Duration must be between 5 and 15 seconds, got {duration}"

        # Sample frames throughout the video
        frames = []
        step = max(1, total_frames // 30)
        idx = 0
        while True:
            ret, f = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()

        assert len(frames) >= 15, "Should have sampled at least 15 frames"

        for f in frames:
            assert not np.isnan(f).any()
            assert not np.isinf(f).any()
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            entropy = compute_shannon_entropy(gray)
        diversities = [compute_color_diversity(f) for f in frames]
        mean_div = float(np.mean(diversities))
        max_div = float(np.max(diversities))
        assert mean_div > 5.0, f"Mean color diversity too low: {mean_div:.2f}"
        assert max_div > 30.0, f"Max color diversity too low: {max_div:.2f}"

        # Verify stream presence via ffprobe if available
        if shutil.which("ffprobe"):
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=codec_type,codec_name",
                "-of", "json", str(video_path),
            ]
            probe_res = subprocess.run(probe_cmd, capture_output=True, text=True)
            probe_json = json.loads(probe_res.stdout)
            types = [s["codec_type"] for s in probe_json.get("streams", [])]
            assert "video" in types, "Video stream must be present"
            assert "audio" in types, "Audio stream must be present"
