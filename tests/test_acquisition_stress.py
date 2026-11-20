"""
tests/test_acquisition_stress.py - Empirical Adversarial Stress-Test Suite for Video Acquisition.

Focus areas:
1. Video and audio asset authenticity and integrity (H.264, AAC, Bad Apple signature).
2. Time parsing and section range boundary stress-testing and edge cases.
3. Cache-hit vs cache-miss behavior (0-byte, corrupt, truncated, wrong duration).
4. Invalid URL handling, retry/fallback mechanisms, and error propagation.
5. Audio extraction throughput, alternative codecs, and silent video handling.
6. Disk safety, missing files, and path handling.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Generator
from unittest.mock import patch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from pipeline.acquisition import (
    VideoAcquisition,
    VideoAcquisitionConfig,
    VideoMetadata,
    AcquisitionError,
    parse_time_to_seconds,
    parse_section_range,
    format_seconds_to_time,
)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def real_clip_path(project_root: Path) -> Path:
    target = project_root / "data" / "bad_apple_clip.mp4"
    if not target.exists():
        survey_clip = project_root / ".agents" / "teamwork" / "teamwork_explorer_survey_2" / "test_clip.mp4"
        if survey_clip.exists():
            return survey_clip
    return target


@pytest.fixture(scope="session")
def real_audio_path(project_root: Path) -> Path:
    return project_root / "data" / "bad_apple_audio.aac"


# ==============================================================================
# Suite 1: Media Authenticity & Integrity Verification
# ==============================================================================

class TestBadAppleAssetAuthenticity:
    """Verifies that the downloaded media is authentic Bad Apple footage."""

    def test_video_container_and_streams_ffprobe(self, real_clip_path: Path):
        assert real_clip_path.exists(), f"Video file missing: {real_clip_path}"
        assert real_clip_path.stat().st_size > 500 * 1024, "File size suspiciously small (< 500 KB)"

        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(real_clip_path)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert res.returncode == 0, f"ffprobe failed: {res.stderr}"

        probe_data = json.loads(res.stdout)
        streams = probe_data.get("streams", [])
        assert len(streams) >= 2, f"Expected at least 2 streams (video + audio), got {len(streams)}"

        # Video stream check
        v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        assert v_stream is not None, "No video stream found"
        assert v_stream.get("codec_name") == "h264", f"Expected H.264 codec, got {v_stream.get('codec_name')}"
        assert v_stream.get("pix_fmt") == "yuv420p"
        assert int(v_stream.get("width", 0)) == 512
        assert int(v_stream.get("height", 0)) == 384
        assert int(v_stream.get("nb_frames", 0)) == 450
        assert float(v_stream.get("duration", 0.0)) == 15.0

        # Audio stream check
        a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        assert a_stream is not None, "No audio stream found"
        assert a_stream.get("codec_name") == "aac", f"Expected AAC codec, got {a_stream.get('codec_name')}"
        assert int(a_stream.get("sample_rate", 0)) == 44100
        assert int(a_stream.get("channels", 0)) == 2

        # Metadata tags check
        fmt_tags = probe_data.get("format", {}).get("tags", {})
        title = fmt_tags.get("title", "")
        comment = fmt_tags.get("comment", "")
        # Either title contains Bad Apple or comment points to sm8628149
        is_bad_apple = "Bad Apple" in title or "sm8628149" in comment or "あにら" in fmt_tags.get("artist", "")
        assert is_bad_apple, f"File metadata tags do not indicate Bad Apple PV: {fmt_tags}"

    def test_video_frame_count_and_timing(self, real_clip_path: Path):
        cap = cv2.VideoCapture(str(real_clip_path))
        assert cap.isOpened(), "cv2.VideoCapture could not open video clip"

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps

        assert abs(fps - 30.0) < 0.1, f"Expected 30 FPS, got {fps}"
        assert frame_count == 450, f"Expected 450 frames, got {frame_count}"
        assert w == 512 and h == 384
        assert abs(duration - 15.0) < 0.05, f"Expected 15.0s duration, got {duration}"

        # Decode every frame sequentially to test for decode errors or corrupt frames
        decoded_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            decoded_frames += 1
            assert frame.shape == (384, 512, 3)

        cap.release()
        assert decoded_frames == 450, f"Only decoded {decoded_frames}/450 frames successfully"

    def test_video_content_authenticity_and_contrast(self, real_clip_path: Path):
        """Bad Apple is iconic shadow art: over 95% of pixels are binary black or white."""
        cap = cv2.VideoCapture(str(real_clip_path))
        means = []
        aa_fractions = []

        step = 15  # Sample every 15 frames (2 samples per second)
        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                means.append(float(np.mean(gray)))
                # Count pixels between 30 and 225 (anti-aliasing boundary)
                mid_pixels = np.count_nonzero((gray > 30) & (gray < 225))
                aa_fractions.append(mid_pixels / gray.size)
            current_frame += 1
        cap.release()

        # Check that video has dynamic range (both black silhouettes and white backgrounds)
        assert min(means) < 50.0, f"Min luminance {min(means)} is too high (lacks dark frames)"
        assert max(means) > 200.0, f"Max luminance {max(means)} is too low (lacks bright frames)"

        # Check that shadow-art sharpness holds: average intermediate pixels < 5%
        avg_aa = float(np.mean(aa_fractions))
        assert avg_aa < 0.05, f"Average anti-aliasing pixel fraction is {avg_aa*100:.2f}% (> 5%). Not sharp shadow art!"

    def test_audio_stream_pcm_and_loudness(self, real_audio_path: Path):
        assert real_audio_path.exists(), f"Audio file missing: {real_audio_path}"
        assert real_audio_path.stat().st_size > 50 * 1024

        cmd = ["ffmpeg", "-v", "quiet", "-i", str(real_audio_path), "-f", "f32le", "-ac", "2", "-ar", "44100", "-"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.returncode == 0, "ffmpeg failed to decode audio"

        pcm = np.frombuffer(proc.stdout, dtype=np.float32)
        total_samples = len(pcm) // 2
        duration = total_samples / 44100.0

        assert 14.9 <= duration <= 15.6, f"Audio duration {duration:.3f}s outside expected 15.0s window"
        rms = float(np.sqrt(np.mean(pcm ** 2)))
        peak = float(np.max(np.abs(pcm)))

        assert rms > 0.05, f"Audio RMS {rms:.4f} is too quiet (possible silence)"
        assert peak > 0.5, f"Audio peak {peak:.4f} indicates weak or missing audio signal"


# ==============================================================================
# Suite 2: Time Parsing & Section Range Boundary Stress Tests
# ==============================================================================

class TestTimeParsingAdversarial:
    """Stress tests time parsing functions with boundary values, invalid types, and malformed inputs."""

    def test_parse_time_valid_variants(self):
        assert parse_time_to_seconds(0) == 0.0
        assert parse_time_to_seconds(15) == 15.0
        assert parse_time_to_seconds(15.75) == 15.75
        assert parse_time_to_seconds("0") == 0.0
        assert parse_time_to_seconds("45.2") == 45.2
        assert parse_time_to_seconds(" 30 ") == 30.0
        assert parse_time_to_seconds("*00:30") == 30.0
        assert parse_time_to_seconds("00:00") == 0.0
        assert parse_time_to_seconds("01:30") == 90.0
        assert parse_time_to_seconds("00:01:30") == 90.0
        assert parse_time_to_seconds("01:00:00") == 3600.0
        assert parse_time_to_seconds("02:15:30.5") == 2 * 3600 + 15 * 60 + 30.5

    def test_parse_time_malformed_inputs(self):
        malformed = [
            "",
            "   ",
            "abc",
            "00:xx",
            "12:34:56:78",
            "None",
            "::",
            "00:00:",
        ]
        for m in malformed:
            with pytest.raises(ValueError):
                parse_time_to_seconds(m)

    def test_nan_and_inf_handling_finding(self):
        """
        Adversarial finding:
        IEEE 754 'inf' and 'nan' strings convert to float('inf') and float('nan') without raising ValueError.
        Specifically, ('nan', 'nan') completely bypasses duration checks in parse_section_range because
        NaN comparisons evaluate to False.
        """
        inf_val = parse_time_to_seconds("inf")
        assert np.isinf(inf_val)

        nan_val = parse_time_to_seconds("nan")
        assert np.isnan(nan_val)

        # parse_section_range with ('nan', 'nan') bypasses end_sec <= start_sec and duration limits
        s, e = parse_section_range(("nan", "nan"), enforce_duration_limits=True)
        assert np.isnan(s) and np.isnan(e)

    def test_parse_section_range_exact_boundaries(self):
        # Exactly 10.0 seconds (minimum allowed)
        s, e = parse_section_range("00:10-00:20", enforce_duration_limits=True)
        assert s == 10.0 and e == 20.0
        assert e - s == 10.0

        # Exactly 20.0 seconds (maximum allowed)
        s, e = parse_section_range("00:00-00:20", enforce_duration_limits=True)
        assert s == 0.0 and e == 20.0
        assert e - s == 20.0

        # Tuple format exact bounds
        s, e = parse_section_range((10.0, 20.0), enforce_duration_limits=True)
        assert s == 10.0 and e == 20.0

    def test_parse_section_range_boundary_violations(self):
        # 9.99 seconds (just below minimum 10.0s)
        with pytest.raises(ValueError, match="Milestone requirements strictly enforce"):
            parse_section_range((0.0, 9.99), enforce_duration_limits=True)

        # 20.01 seconds (just above maximum 20.0s)
        with pytest.raises(ValueError, match="Milestone requirements strictly enforce"):
            parse_section_range((0.0, 20.01), enforce_duration_limits=True)

        # Zero duration
        with pytest.raises(ValueError, match="End time"):
            parse_section_range("00:30-00:30")

        # Inverted range
        with pytest.raises(ValueError, match="End time"):
            parse_section_range("00:45-00:30")

    def test_parse_section_range_malformed_formats(self):
        invalid_formats = [
            "00:30",  # Missing second time
            "00:30-00:40-00:50",  # Extra hyphen
            "-",
            "00:30-",
            "-00:30",
            "invalid-range",
        ]
        for inv in invalid_formats:
            with pytest.raises(ValueError):
                parse_section_range(inv)

        # Unsupported types
        with pytest.raises(TypeError):
            parse_section_range(12345)  # type: ignore

        with pytest.raises(ValueError):
            parse_section_range((10.0,))  # 1 element

        with pytest.raises(ValueError):
            parse_section_range((10.0, 20.0, 30.0))  # 3 elements

    def test_negative_timestamp_finding(self):
        """
        Adversarial test: tests how parse_section_range and format_seconds_to_time handle negative timestamps.
        Documents the empirical behavior.
        """
        # String format '-00:10-00:05' splits into 3 parts due to leading minus, raising ValueError
        with pytest.raises(ValueError):
            parse_section_range("-00:10-00:05")

        # Tuple format (-10.0, 5.0): start_sec is negative.
        # Duration is 5.0 - (-10.0) = 15.0s, which is within [10, 20].
        # parse_section_range currently accepts this without checking start_sec >= 0.
        s, e = parse_section_range((-10.0, 5.0), enforce_duration_limits=True)
        assert s == -10.0 and e == 5.0

        # format_seconds_to_time with negative time wraps around:
        wrapped_time = format_seconds_to_time(-10.0)
        assert wrapped_time == "59:50.00", f"Unexpected wrapped time: {wrapped_time}"


# ==============================================================================
# Suite 3: Cache Hit vs Cache Miss Behavior
# ==============================================================================

class TestCacheBehaviorAndValidation:
    """Stress tests local cache hit, miss, truncation, and corruption scenarios."""

    def test_cache_hit_bypasses_network(self, real_clip_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            cached_clip = td / "bad_apple_clip.mp4"
            shutil.copy2(str(real_clip_path), str(cached_clip))

            acq = VideoAcquisition(output_dir=td)
            # Use an unresolvable URL to prove no network call is made
            result = acq.fetch_clip(
                source_url="http://completely-unresolvable-domain-123456.org/video.mp4",
                section="00:30-00:45",
                force_download=False
            )
            assert result == cached_clip
            assert result.exists()

    def test_cache_miss_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            empty_file = td / "bad_apple_clip.mp4"
            empty_file.write_bytes(b"")

            acq = VideoAcquisition(output_dir=td)
            assert not acq.is_valid_clip(empty_file)

    def test_cache_miss_truncated_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            truncated = td / "bad_apple_clip.mp4"
            truncated.write_bytes(b"\x00" * 50000)  # 50 KB < 100 KB threshold

            acq = VideoAcquisition(output_dir=td)
            assert not acq.is_valid_clip(truncated)

    def test_cache_miss_corrupted_garbage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            corrupt = td / "bad_apple_clip.mp4"
            # 200 KB of random bytes (larger than threshold, but invalid MP4 container)
            corrupt.write_bytes(b"CORRUPT_NOT_A_VALID_MP4_HEADER" * 7000)

            acq = VideoAcquisition(output_dir=td)
            assert not acq.is_valid_clip(corrupt)

    def test_cache_miss_wrong_duration(self, real_clip_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            # Create a 2-second clip using ffmpeg
            short_clip = td / "short.mp4"
            cmd = [
                "ffmpeg", "-y", "-i", str(real_clip_path),
                "-t", "2.0",
                "-c", "copy",
                str(short_clip)
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            acq = VideoAcquisition(output_dir=td)
            # is_valid_clip default expects [10.0, 20.0] duration
            assert not acq.is_valid_clip(short_clip, min_duration=10.0, max_duration=20.0)

    def test_force_download_bypasses_cache(self, real_clip_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            cached_clip = td / "bad_apple_clip.mp4"
            shutil.copy2(str(real_clip_path), str(cached_clip))

            acq = VideoAcquisition(output_dir=td)
            # Mock download_segment to simulate download execution when force_download=True
            with patch.object(acq, "download_segment") as mock_dl:
                mock_dl.return_value = cached_clip
                result = acq.fetch_clip(
                    source_url="https://archive.org/details/bad-apple-pv_202307",
                    force_download=True
                )
                assert mock_dl.called, "download_segment should have been called when force_download=True"


# ==============================================================================
# Suite 4: Network Failure, Invalid URLs, and Fallback Handling
# ==============================================================================

class TestNetworkAndFallbackErrorHandling:
    """Stress tests network exceptions and fallback chains."""

    def test_invalid_url_without_fallback_raises_acquisition_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            acq = VideoAcquisition(output_dir=td)

            # Mock download_segment to simulate network failure immediately without DNS timeouts
            def mock_fail(url, start_sec, end_sec, target_file):
                raise AcquisitionError(f"Simulated network unreachable for {url}")

            # Prevent offline fallback from finding test_clip.mp4
            orig_exists = Path.exists

            def mock_exists_fn(self, *args, **kwargs):
                if "teamwork_explorer_survey_2" in str(self):
                    return False
                return orig_exists(self, *args, **kwargs)

            with patch.object(acq, "download_segment", side_effect=mock_fail):
                with patch.object(Path, "exists", mock_exists_fn):
                    with pytest.raises(AcquisitionError, match="All download sources failed"):
                        acq.fetch_clip(
                            source_url="http://completely-unresolvable-domain-99999.invalid/fake.mp4",
                            fallback_url="http://completely-unresolvable-fallback-99999.invalid/fake.mp4",
                            force_download=True
                        )

    def test_secondary_fallback_attempted_on_primary_failure(self, real_clip_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            acq = VideoAcquisition(output_dir=td)

            calls = []

            def mock_download(url, start_sec, end_sec, target_file):
                calls.append(url)
                if "primary-fail" in url:
                    raise AcquisitionError(f"Simulated network error for {url}")
                # Secondary succeeds
                shutil.copy2(str(real_clip_path), str(target_file))
                return target_file

            with patch.object(acq, "download_segment", side_effect=mock_download):
                result = acq.fetch_clip(
                    source_url="http://primary-fail.example.com/video.mp4",
                    fallback_url="http://secondary-success.example.com/video.mp4",
                    force_download=True
                )
                assert "http://primary-fail.example.com/video.mp4" in calls
                assert "http://secondary-success.example.com/video.mp4" in calls
                assert result.exists()

    def test_offline_fallback_recovery(self, project_root: Path):
        survey_clip = project_root / ".agents" / "teamwork" / "teamwork_explorer_survey_2" / "test_clip.mp4"
        if not survey_clip.exists():
            pytest.skip("Survey reference clip not present in workspace")

        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            acq = VideoAcquisition(output_dir=td)

            def mock_fail(url, start, end, target):
                raise AcquisitionError("Network unreachable")

            with patch.object(acq, "download_segment", side_effect=mock_fail):
                result = acq.fetch_clip(
                    source_url="http://unreachable-network.org/bad_apple.mp4",
                    fallback_url="http://unreachable-fallback.org/bad_apple.mp4",
                    force_download=True
                )
                assert result.exists()
                assert result.stat().st_size > 100 * 1024


# ==============================================================================
# Suite 5: Audio Extraction & Throughput Performance
# ==============================================================================

class TestAudioExtractionAndThroughput:
    """Measures execution throughput and edge cases in audio extraction."""

    def test_extract_audio_stream_copy_throughput(self, real_clip_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            acq = VideoAcquisition(output_dir=td)
            out_audio = td / "bench_audio.aac"

            t0 = time.perf_counter()
            res = acq.extract_audio(real_clip_path, output_filename=out_audio)
            elapsed = time.perf_counter() - t0

            assert res.exists()
            assert res.stat().st_size > 100 * 1024
            # Stream copy should execute in less than 500 ms
            assert elapsed < 0.5, f"Audio stream copy took too long: {elapsed:.3f}s (budget: 0.5s)"

    def test_extract_audio_transcode_mp3(self, real_clip_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            acq = VideoAcquisition(output_dir=td)
            out_mp3 = td / "bench_audio.mp3"

            t0 = time.perf_counter()
            res = acq.extract_audio(real_clip_path, output_filename=out_mp3)
            elapsed = time.perf_counter() - t0

            assert res.exists()
            assert res.stat().st_size > 100 * 1024
            # Transcoding to MP3 should complete in under 2.0s for 15s clip
            assert elapsed < 2.0, f"MP3 transcode took too long: {elapsed:.3f}s (budget: 2.0s)"

    def test_extract_audio_silent_video_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            silent_mp4 = td / "silent.mp4"
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "testsrc=duration=1:size=320x240:rate=30",
                "-c:v", "libx264", "-an",
                str(silent_mp4)
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            acq = VideoAcquisition(output_dir=td)
            with pytest.raises(AcquisitionError, match="FFmpeg audio extraction failed"):
                acq.extract_audio(silent_mp4, output_filename=td / "silent.aac")

    def test_probe_video_throughput(self, real_clip_path: Path):
        acq = VideoAcquisition()
        t0 = time.perf_counter()
        meta = acq.probe_video(real_clip_path)
        elapsed = time.perf_counter() - t0

        assert meta.total_frames == 450
        # Media probe should take under 50 ms
        assert elapsed < 0.05, f"Video probing took too long: {elapsed*1000:.2f} ms (budget: 50 ms)"


# ==============================================================================
# Suite 6: Disk Safety & File Error Propagation
# ==============================================================================

class TestDiskSafetyAndErrors:
    """Verifies filesystem error handling and directory creation safety."""

    def test_missing_video_probe_raises_filenotfound(self):
        acq = VideoAcquisition()
        with pytest.raises(FileNotFoundError):
            acq.probe_video("non_existent_file_path_12345.mp4")

    def test_zero_byte_video_probe_raises_acquisition_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            empty_file = td / "empty.mp4"
            empty_file.write_bytes(b"")

            acq = VideoAcquisition(output_dir=td)
            with pytest.raises(AcquisitionError, match="Video file is empty"):
                acq.probe_video(empty_file)

    def test_missing_video_extract_audio_raises_filenotfound(self):
        acq = VideoAcquisition()
        with pytest.raises(FileNotFoundError):
            acq.extract_audio("missing_video.mp4")

    def test_output_dir_auto_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_dir = Path(tmpdir) / "level1" / "level2" / "deep_data"
            assert not deep_dir.exists()

            acq = VideoAcquisition(output_dir=deep_dir)
            assert deep_dir.exists(), "VideoAcquisition failed to create parent directories"

    def test_acquire_duration_validation_rejection(self, real_clip_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            short_clip = td / "short.mp4"
            # Create a 5s clip
            cmd = ["ffmpeg", "-y", "-i", str(real_clip_path), "-t", "5.0", "-c", "copy", str(short_clip)]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            acq = VideoAcquisition(output_dir=td)
            # If acquire somehow produces a clip outside [10, 20]s, it must raise AcquisitionError
            with patch.object(acq, "fetch_clip", return_value=short_clip):
                with pytest.raises(AcquisitionError, match="outside required \\[10, 20\\] second window"):
                    acq.acquire(section="00:00-00:15")
