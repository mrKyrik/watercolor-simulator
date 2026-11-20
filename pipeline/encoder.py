"""
pipeline/encoder.py - Video Stream Encoding and Audio Multiplexing Module.

Supports:
- Direct FFmpeg stdin pipe encoding (H.264 libx264, crf 18, yuv420p, preset medium).
- Deadlock-free pipe streaming with stderr isolation (-loglevel error).
- On-the-fly audio multiplexing via FFmpeg stdin + audio input.
- High-speed standalone stream copy audio remuxing (-c:v copy -c:a aac -shortest).
- OpenCV cv2.VideoWriter fallback when FFmpeg is not available on PATH.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Any
import logging
import os
import shutil
import subprocess

import cv2
import numpy as np

logger = logging.getLogger("pipeline.encoder")


class VideoEncodingError(RuntimeError):
    """Raised when video stream encoding fails."""
    pass


class AudioMuxingError(RuntimeError):
    """Raised when audio multiplexing fails."""
    pass


@dataclass
class VideoEncoderConfig:
    """Configuration for video encoding and audio multiplexing."""
    output_path: Union[str, Path]
    width: int = 512
    height: int = 384
    fps: float = 15.0
    codec: str = "libx264"
    pix_fmt: str = "yuv420p"
    crf: int = 18
    preset: str = "medium"
    audio_path: Optional[Union[str, Path]] = None
    backend: str = "auto"           # "auto", "ffmpeg", or "opencv"
    input_pix_fmt: str = "rgb24"    # "rgb24" or "bgr24"
    loglevel: str = "error"


class BaseVideoEncoder(ABC):
    """Abstract base class for video encoders."""

    @abstractmethod
    def write_frame(self, frame: np.ndarray) -> None:
        """Writes a single RGB uint8 frame to the video stream."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Finalizes the video stream and flushes output."""
        pass

    def __enter__(self) -> "BaseVideoEncoder":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class FFmpegPipeEncoder(BaseVideoEncoder):
    """
    Streams raw video frames directly into an FFmpeg subprocess via standard input.
    Encodes visually lossless H.264 (CRF 18) and muxes audio simultaneously.
    Designed to prevent pipe deadlocks on Windows via log isolation and explicit stream closing.
    """

    def __init__(self, config: VideoEncoderConfig) -> None:
        self.config = config
        self.output_path = Path(config.output_path).resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if config.width % 2 != 0 or config.height % 2 != 0:
            raise ValueError(f"Dimensions must be even for yuv420p chroma subsampling: {config.width}x{config.height}")

        self.expected_frame_bytes = config.width * config.height * 3
        self.frames_written = 0
        self._is_closed = False

        cmd = [
            "ffmpeg", "-y",
            "-loglevel", config.loglevel,
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{config.width}x{config.height}",
            "-pix_fmt", config.input_pix_fmt,
            "-r", str(config.fps),
            "-i", "-",  # Video from stdin pipe
        ]

        self.has_audio = False
        if config.audio_path:
            audio_p = Path(config.audio_path).resolve()
            if audio_p.exists() and audio_p.stat().st_size > 0:
                cmd.extend(["-i", str(audio_p)])
                self.has_audio = True
            else:
                logger.warning(f"Audio file '{audio_p}' not found or empty. Encoding video only.")

        cmd.extend([
            "-c:v", config.codec,
            "-pix_fmt", config.pix_fmt,
            "-preset", config.preset,
            "-crf", str(config.crf),
        ])

        if self.has_audio:
            cmd.extend([
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
            ])

        cmd.append(str(self.output_path))
        self.cmd = cmd

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise VideoEncodingError("FFmpeg executable not found on system PATH.") from e
        except Exception as e:
            raise VideoEncodingError(f"Failed to spawn FFmpeg process: {e}") from e

    def write_frame(self, frame: np.ndarray) -> None:
        if self._is_closed:
            raise VideoEncodingError("Attempted to write frame to closed FFmpegPipeEncoder.")

        if self.proc.poll() is not None:
            _, err_bytes = self.proc.communicate()
            err_msg = err_bytes.decode("utf-8", errors="ignore").strip()
            raise VideoEncodingError(
                f"FFmpeg process terminated prematurely (exit code {self.proc.returncode}): {err_msg}"
            )

        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Frame must be a numpy ndarray, got {type(frame)}")

        h, w = frame.shape[:2]
        if w != self.config.width or h != self.config.height:
            raise ValueError(
                f"Frame dimension mismatch: expected ({self.config.height}, {self.config.width}), got ({h}, {w})"
            )

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)

        frame_bytes = frame.tobytes()
        if len(frame_bytes) != self.expected_frame_bytes:
            raise ValueError(f"Expected {self.expected_frame_bytes} bytes, got {len(frame_bytes)}")

        try:
            self.proc.stdin.write(frame_bytes)
            self.frames_written += 1
        except (BrokenPipeError, OSError) as e:
            _, err_bytes = self.proc.communicate()
            err_msg = err_bytes.decode("utf-8", errors="ignore").strip()
            raise VideoEncodingError(
                f"FFmpeg pipe broken on frame {self.frames_written}: {err_msg or e}"
            ) from e

    def close(self) -> None:
        if self._is_closed:
            return
        self._is_closed = True

        if hasattr(self, "proc") and self.proc:
            if self.proc.stdin and not self.proc.stdin.closed:
                try:
                    self.proc.stdin.close()
                except OSError:
                    pass

            try:
                _, stderr_data = self.proc.communicate(timeout=60.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                _, stderr_data = self.proc.communicate()
                raise VideoEncodingError("FFmpeg process timed out during stream finalization.")

            if self.proc.returncode != 0:
                err_msg = stderr_data.decode("utf-8", errors="ignore").strip()
                raise VideoEncodingError(
                    f"FFmpeg encoding failed with exit code {self.proc.returncode}: {err_msg}"
                )

        if not self.output_path.exists() or self.output_path.stat().st_size == 0:
            raise VideoEncodingError(f"Output video was not created or is empty: {self.output_path}")

        logger.info(
            f"FFmpeg successfully encoded {self.frames_written} frames to {self.output_path} "
            f"({self.output_path.stat().st_size} bytes)"
        )


class OpenCVFallbackEncoder(BaseVideoEncoder):
    """
    Fallback video encoder using OpenCV cv2.VideoWriter ('mp4v').
    Writes silent video first, then optionally muxes audio via FFmpeg if available.
    """

    def __init__(self, config: VideoEncoderConfig) -> None:
        self.config = config
        self.output_path = Path(config.output_path).resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_video_path = self.output_path.with_suffix(".temp_silent.mp4")
        self.frames_written = 0
        self._is_closed = False

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.temp_video_path),
            fourcc,
            float(config.fps),
            (config.width, config.height),
            isColor=True,
        )

        if not self.writer.isOpened():
            raise VideoEncodingError(f"Failed to open OpenCV VideoWriter with 'mp4v' at {self.temp_video_path}")

    def write_frame(self, frame: np.ndarray) -> None:
        if self._is_closed:
            raise VideoEncodingError("Attempted to write to closed OpenCVFallbackEncoder.")

        h, w = frame.shape[:2]
        if w != self.config.width or h != self.config.height:
            raise ValueError(
                f"Frame dimension mismatch: expected ({self.config.height}, {self.config.width}), got ({h}, {w})"
            )

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        if self.config.input_pix_fmt == "rgb24":
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            bgr = frame

        self.writer.write(bgr)
        self.frames_written += 1

    def close(self) -> None:
        if self._is_closed:
            return
        self._is_closed = True

        if hasattr(self, "writer") and self.writer:
            self.writer.release()

        if not self.temp_video_path.exists() or self.temp_video_path.stat().st_size == 0:
            raise VideoEncodingError(f"OpenCV writer failed to produce valid video file at {self.temp_video_path}")

        # Post-mux audio if requested and audio file exists
        if self.config.audio_path and Path(self.config.audio_path).exists():
            if shutil.which("ffmpeg"):
                try:
                    mux_audio_to_video(
                        video_path=self.temp_video_path,
                        audio_path=self.config.audio_path,
                        output_path=self.output_path,
                        overwrite=True,
                    )
                    if self.temp_video_path.exists():
                        self.temp_video_path.unlink()
                    return
                except Exception as e:
                    logger.warning(f"Post-muxing audio failed: {e}. Preserving video-only file.")
            else:
                logger.warning("FFmpeg not installed. Cannot multiplex audio with OpenCV fallback. Output will be silent.")

        # If no audio or remux failed, move temp to final output
        if self.output_path.exists():
            self.output_path.unlink()
        shutil.move(str(self.temp_video_path), str(self.output_path))


def create_video_encoder(config: VideoEncoderConfig) -> BaseVideoEncoder:
    """
    Factory creating an appropriate video encoder.
    Prefers FFmpegPipeEncoder; falls back to OpenCVFallbackEncoder.
    """
    backend = config.backend.lower()
    ffmpeg_available = shutil.which("ffmpeg") is not None

    if backend == "opencv":
        logger.info("Using OpenCVFallbackEncoder as explicitly requested.")
        return OpenCVFallbackEncoder(config)
    elif backend in ("auto", "ffmpeg"):
        if ffmpeg_available:
            try:
                return FFmpegPipeEncoder(config)
            except Exception as e:
                if backend == "ffmpeg":
                    raise
                logger.warning(f"FFmpeg pipe encoder failed to initialize ({e}). Falling back to OpenCV.")
                return OpenCVFallbackEncoder(config)
        else:
            if backend == "ffmpeg":
                raise VideoEncodingError("FFmpeg backend requested but FFmpeg is not installed on system PATH.")
            logger.info("FFmpeg not found on PATH. Falling back to OpenCVFallbackEncoder.")
            return OpenCVFallbackEncoder(config)
    else:
        raise ValueError(f"Unknown video encoder backend: '{config.backend}'. Must be 'auto', 'ffmpeg', or 'opencv'.")


class VideoEncoder:
    """
    Convenience wrapper unifying FFmpeg pipe encoding and OpenCV fallback.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        width: int = 512,
        height: int = 384,
        fps: float = 15.0,
        audio_path: Optional[Union[str, Path]] = None,
        crf: int = 18,
        preset: str = "medium",
        backend: str = "auto",
    ) -> None:
        self.config = VideoEncoderConfig(
            output_path=output_path,
            width=width,
            height=height,
            fps=fps,
            audio_path=audio_path,
            crf=crf,
            preset=preset,
            backend=backend,
        )
        self._encoder = create_video_encoder(self.config)

    def write_frame(self, frame: np.ndarray) -> None:
        self._encoder.write_frame(frame)

    def close(self) -> None:
        self._encoder.close()

    def __enter__(self) -> "VideoEncoder":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def mux_audio_to_video(
    video_path: Union[str, Path],
    audio_path: Union[str, Path],
    output_path: Union[str, Path],
    shortest: bool = True,
    overwrite: bool = True,
) -> Path:
    """
    Merges an existing video file with an AAC audio file using stream copy.
    Command: ffmpeg -y -loglevel error -i video -i audio -c:v copy -c:a aac -shortest out
    
    Safe for in-place remuxing (video_path == output_path) via atomic temporary rename.
    """
    v_path = Path(video_path).resolve()
    a_path = Path(audio_path).resolve()
    out_path = Path(output_path).resolve()

    if not v_path.exists():
        raise FileNotFoundError(f"Video file not found: {v_path}")
    if not a_path.exists():
        raise FileNotFoundError(f"Audio file not found: {a_path}")

    in_place = (v_path == out_path)
    target_out = out_path.with_suffix(".tmp_mux.mp4") if in_place else out_path
    target_out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-loglevel", "error",
        "-i", str(v_path),
        "-i", str(a_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
    ]
    if shortest:
        cmd.append("-shortest")
    cmd.append(str(target_out))

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        if target_out.exists():
            target_out.unlink()
        raise AudioMuxingError(f"Audio muxing failed (exit {res.returncode}): {res.stderr.strip()}")

    if in_place:
        v_path.unlink()
        shutil.move(str(target_out), str(out_path))

    return out_path
