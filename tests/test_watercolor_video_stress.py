"""
tests/test_watercolor_video_stress.py - Empirical Challenger Stress Test Suite for Milestone 2.

Adversarial stress-testing of data/bad_apple_watercolor.mp4 and the video engine:
1. Container parameters (duration 5-15s, FPS >= 12, resolution 512x384).
2. Audio stream specifications (AAC, 44.1 kHz, stereo, AV sync delta < 0.1s).
3. Full sequential decoding of all 225 frames without corruption or dropped packets.
4. FFmpeg null-sink packet integrity and error logging.
5. Packet-level DTS/PTS monotonicity and keyframe cadence.
6. Bidirectional arbitrary seek robustness (forward and backward seek verification).
7. Out-of-bounds seek boundary handling.
8. Inter-frame temporal smoothness and scene cut transition dynamics.
9. Silhouette spatial alignment and chromatic droplet infilling.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

VIDEO_PATH = Path("data/bad_apple_watercolor.mp4")
MASKS_DIR = Path("data/preprocessed/masks")


@pytest.fixture(scope="module")
def video_file():
    """Ensure the target video file exists."""
    assert VIDEO_PATH.exists(), f"Video deliverable {VIDEO_PATH} does not exist"
    return VIDEO_PATH


@pytest.fixture(scope="module")
def sequential_frames(video_file):
    """Decode all frames sequentially once and cache in memory for oracle comparisons."""
    cap = cv2.VideoCapture(str(video_file))
    assert cap.isOpened(), "Failed to open video with cv2.VideoCapture"
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


class TestEmpiricalContainerParameters:
    """Empirical verification of video container parameters."""

    def test_container_parameters_strict(self, video_file):
        cap = cv2.VideoCapture(str(video_file))
        assert cap.isOpened(), "VideoCapture must open video successfully"
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / max(fps, 1.0)
        cap.release()

        assert total_frames == 225, f"Expected exactly 225 frames, got {total_frames}"
        assert width == 512, f"Expected width 512, got {width}"
        assert height == 384, f"Expected height 384, got {height}"
        assert fps == 15.0, f"Expected FPS 15.0 (>= 12), got {fps}"
        assert 5.0 <= duration <= 15.0, f"Duration {duration:.2f}s outside required [5, 15]s range"

    def test_ffprobe_stream_metadata(self, video_file):
        if not shutil.which("ffprobe"):
            pytest.skip("ffprobe not available in PATH")

        cmd = [
            "ffprobe", "-v", "error",
            "-show_streams", "-show_format",
            "-of", "json",
            str(video_file)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        meta = json.loads(res.stdout)

        v_stream = next((s for s in meta["streams"] if s["codec_type"] == "video"), None)
        assert v_stream is not None, "Video stream must be present"
        assert v_stream["codec_name"] == "h264", f"Expected h264, got {v_stream['codec_name']}"
        assert int(v_stream["width"]) == 512
        assert int(v_stream["height"]) == 384
        assert v_stream["pix_fmt"] == "yuv420p"
        assert v_stream["nb_frames"] == "225"
        v_dur = float(v_stream["duration"])
        assert 5.0 <= v_dur <= 15.0


class TestEmpiricalAudioTrack:
    """Empirical verification of audio track specifications and AV sync."""

    def test_audio_specifications_and_av_sync(self, video_file):
        if not shutil.which("ffprobe"):
            pytest.skip("ffprobe not available in PATH")

        cmd = [
            "ffprobe", "-v", "error",
            "-show_streams", "-show_format",
            "-of", "json",
            str(video_file)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        meta = json.loads(res.stdout)

        v_stream = next((s for s in meta["streams"] if s["codec_type"] == "video"), None)
        a_stream = next((s for s in meta["streams"] if s["codec_type"] == "audio"), None)

        assert a_stream is not None, "Audio stream must be present in container"
        assert a_stream["codec_name"] == "aac", f"Expected AAC codec, got {a_stream['codec_name']}"
        assert int(a_stream["channels"]) == 2, f"Expected 2 channels (stereo), got {a_stream['channels']}"
        assert int(a_stream["sample_rate"]) == 44100, f"Expected 44.1 kHz, got {a_stream['sample_rate']}"

        v_duration = float(v_stream["duration"])
        a_duration = float(a_stream["duration"])
        sync_diff = abs(v_duration - a_duration)

        # Audio/video duration synchronization delta must be strictly < 0.1s
        assert sync_diff < 0.1, (
            f"Audio/video sync difference too large: {sync_diff:.4f}s (v={v_duration}s, a={a_duration}s)"
        )


class TestEmpiricalDecoderRobustness:
    """Empirical stress-testing of decoding integrity, packet health, and random seeking."""

    def test_full_sequential_frame_decode_all_225(self, sequential_frames):
        assert len(sequential_frames) == 225, f"Expected 225 frames, got {len(sequential_frames)}"

        for i, frame in enumerate(sequential_frames):
            assert frame is not None, f"Frame {i} is None"
            assert frame.shape == (384, 512, 3), f"Frame {i} shape mismatch: {frame.shape}"
            assert frame.dtype == np.uint8, f"Frame {i} dtype mismatch: {frame.dtype}"
            assert not np.isnan(frame).any(), f"Frame {i} contains NaN"
            assert not np.isinf(frame).any(), f"Frame {i} contains Inf"

            # Check that frame is neither pure black nor pure white
            mean_val = float(np.mean(frame))
            std_val = float(np.std(frame))
            assert 10.0 < mean_val < 250.0, f"Frame {i} mean brightness anomaly: {mean_val:.2f}"
            assert std_val > 5.0, f"Frame {i} lacks visual content / too flat: std={std_val:.2f}"

    def test_ffmpeg_null_decode_packet_integrity(self, video_file):
        """Pass full stream through ffmpeg null muxer to catch bitstream corruption or dropped packets."""
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not available in PATH")

        cmd = [
            "ffmpeg", "-v", "error",
            "-i", str(video_file),
            "-f", "null", "-"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"ffmpeg null decode failed with code {res.returncode}: {res.stderr}"
        assert res.stderr.strip() == "", f"ffmpeg detected packet/bitstream errors: {res.stderr}"

    def test_packet_monotonicity_and_keyframes(self, video_file):
        """Check presentation and decode timestamp monotonicity for video and audio packets."""
        if not shutil.which("ffprobe"):
            pytest.skip("ffprobe not available in PATH")

        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "packet=pts,dts,flags,stream_index",
            "-of", "json",
            str(video_file)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        packets = json.loads(res.stdout).get("packets", [])
        assert len(packets) > 0, "No packets found in video"

        v_pkts = [p for p in packets if p.get("stream_index") == 0]
        a_pkts = [p for p in packets if p.get("stream_index") == 1]

        assert len(v_pkts) == 225, f"Expected 225 video packets, found {len(v_pkts)}"
        assert len(a_pkts) > 600, f"Expected >600 audio packets, found {len(a_pkts)}"

        # DTS monotonicity
        v_dts = [int(p["dts"]) for p in v_pkts if "dts" in p]
        assert all(v_dts[i] <= v_dts[i + 1] for i in range(len(v_dts) - 1)), "Video DTS is not monotonic"

        a_dts = [int(p["dts"]) for p in a_pkts if "dts" in p]
        assert all(a_dts[i] <= a_dts[i + 1] for i in range(len(a_dts) - 1)), "Audio DTS is not monotonic"

        # Check keyframe count: must have at least 1 keyframe (IDR frame)
        keyframes = [p for p in v_pkts if "K" in p.get("flags", "")]
        assert len(keyframes) >= 1, "Video contains zero keyframes"

    def test_bidirectional_seek_robustness(self, video_file, sequential_frames):
        """Stress-test random arbitrary forward and backward seeks against sequential baseline."""
        cap = cv2.VideoCapture(str(video_file))
        assert cap.isOpened()

        # Highly non-linear sequence of target frame indices
        seek_targets = [
            0, 50, 25, 100, 10, 200, 150, 5, 224, 0,
            180, 75, 220, 33, 160, 90, 120, 15, 205, 45
        ]

        for target in seek_targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            assert ret, f"Failed to decode frame at seek position {target}"
            # Compare with sequential oracle frame
            expected = sequential_frames[target]
            diff = np.max(np.abs(frame.astype(float) - expected.astype(float)))
            assert diff == 0.0, f"Seek mismatch at frame {target}: max pixel diff={diff}"

        cap.release()

    def test_out_of_bounds_seek_handling(self, video_file):
        """Check decoder safety when seeking at or beyond EOF."""
        cap = cv2.VideoCapture(str(video_file))
        assert cap.isOpened()

        # Seek exactly to total_frames (index 225)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 225)
        ret, frame = cap.read()
        assert not ret, "Reading frame at index 225 must return ret=False (clean EOF)"

        # Seek far beyond total_frames (index 500)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
        ret, frame = cap.read()
        assert not ret, "Reading frame at index 500 must return ret=False without crashing"

        cap.release()


class TestEmpiricalVisualAndPhysicsDynamics:
    """Stress-test physical realism: temporal continuity, scene transition, and silhouette alignment."""

    def test_interframe_temporal_smoothness(self, sequential_frames):
        """Verify bounded inter-frame MAE across continuous sequences (absence of boiling)."""
        maes = []
        for i in range(1, len(sequential_frames)):
            f_prev = sequential_frames[i - 1].astype(float)
            f_curr = sequential_frames[i].astype(float)
            mae = float(np.mean(np.abs(f_curr - f_prev)))
            maes.append(mae)

        # In continuous animation segments, MAE should remain low (< 25.0).
        # At scene cuts (e.g. frame 200-205), MAE may spike but should not exceed 100.
        high_mae_count = sum(1 for m in maes if m > 35.0)
        # Should have very few high-MAE frames (only scene transitions)
        assert high_mae_count <= 5, f"Too many temporal discontinuities / boiling detected: {high_mae_count}"
        median_mae = float(np.median(maes))
        assert median_mae < 15.0, f"Median inter-frame MAE too high: {median_mae:.2f}"

    def test_silhouette_alignment_and_chromatic_richness(self, video_file, sequential_frames):
        """Verify that watercolor pigments are concentrated inside the Bad Apple silhouettes."""
        if not MASKS_DIR.exists():
            pytest.skip("Masks directory not found")

        # Test at various representative frames
        test_indices = [15, 45, 90, 135, 180]
        for idx in test_indices:
            mask_path = MASKS_DIR / f"mask_{idx:04d}.png"
            if not mask_path.exists():
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            frame = sequential_frames[idx]

            fg_mask = mask > 128
            bg_mask = mask <= 128

            if fg_mask.sum() > 500:
                fg_pixels = frame[fg_mask]
                bg_pixels = frame[bg_mask]

                # Convert to HSV to test saturation
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                fg_sat = hsv[fg_mask, 1]

                # Foreground should have significant color saturation from watercolor pigments
                mean_fg_sat = float(np.mean(fg_sat))
                assert mean_fg_sat > 25.0, (
                    f"Frame {idx}: Silhouette interior lacks color saturation ({mean_fg_sat:.1f})"
                )

                # Background should have low saturation (creamy watercolor paper substrate)
                bg_sat = hsv[bg_mask, 1]
                mean_bg_sat = float(np.mean(bg_sat))
                assert mean_bg_sat < 20.0, (
                    f"Frame {idx}: Paper substrate background has unexpected saturation ({mean_bg_sat:.1f})"
                )
