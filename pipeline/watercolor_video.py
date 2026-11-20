"""
pipeline/watercolor_video.py - Bad Apple Flowing Watercolor Video Pipeline.

Orchestrates:
1. Streaming preprocessed silhouette frames and distance transforms from data/preprocessed.
2. Coupling OpenCV DIS Optical Flow with PyTorch Semi-Lagrangian GPU state advection.
3. EDT-weighted droplet infilling into newly uncovered silhouette areas.
4. Multi-chromatic Kubelka-Munk palette rendering over static paper substrate.
5. Direct FFmpeg pipe encoding and AAC audio multiplexing.
6. Real-time telemetry reporting (instantaneous/rolling FPS, elapsed time, ETA, CUDA VRAM).
7. Final deliverable generation: data/bad_apple_watercolor.mp4.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable, Union, Generator, List
import argparse
import json
import logging
import time
import os

import cv2
import numpy as np
import torch

from engine.video_simulation import (
    VideoWatercolorEngine,
    VideoSimulationConfig,
    TransitionMode,
    HarmonicPalette3D,
)
from pipeline.encoder import (
    VideoEncoderConfig,
    create_video_encoder,
    BaseVideoEncoder,
    VideoEncodingError,
    mux_audio_to_video,
)

logger = logging.getLogger("pipeline.watercolor_video")


class PreprocessedDataError(RuntimeError):
    """Raised when preprocessed assets cannot be discovered, parsed, or read."""
    pass


class WatercolorPipelineError(RuntimeError):
    """Raised when pipeline rendering fails or is aborted."""
    pass


@dataclass
class PreprocessedFrameItem:
    """Encapsulates precomputed assets for a single video frame."""
    frame_idx: int                                              # Preprocessed sequential index (0, 1, 2, ...)
    src_frame_idx: int                                          # Original source video frame index (0, 2, 4, ...)
    timestamp_sec: float                                        # Presentation timestamp in seconds (0.0, 0.0667, ...)
    mask: np.ndarray                                            # (H, W) uint8 binary mask, strictly in {0, 255}
    dist: np.ndarray                                            # (H, W) float32 Euclidean Distance Transform (L2)
    raw_frame: Optional[np.ndarray] = None                      # (H, W, 3) uint8 RGB source frame (optional)
    is_white_background: bool = True                            # Polarity detected by preprocessor
    is_polarity_flip: bool = False                              # True if transition frame between white/black background
    character_bbox: Optional[Tuple[int, int, int, int]] = None  # [x, y, w, h] bounding box
    max_distance: float = 0.0                                   # Precomputed maximum interior thickness (px)


def stream_preprocessed_frames(
    preprocessed_dir: Union[str, Path],
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    load_raw_frames: bool = False,
) -> Generator[PreprocessedFrameItem, None, None]:
    """
    Streams preprocessed Bad Apple frame assets sequentially from disk on demand.
    Maintains minimal memory overhead (< 5 MB RAM) regardless of sequence length.
    
    Reads data/preprocessed/metadata.json to retrieve exact frame metadata and paths.
    If metadata.json is absent, discovers assets via lexical sorting of masks and distance dirs.
    """
    base_dir = Path(preprocessed_dir).resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        raise PreprocessedDataError(f"Preprocessed directory not found: {base_dir}")

    metadata_path = base_dir / "metadata.json"

    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            frame_records = meta.get("frames", [])
        except Exception as e:
            raise PreprocessedDataError(f"Failed to parse metadata.json in {base_dir}: {e}") from e

        total_available = len(frame_records)
        end_idx = total_available if max_frames is None else min(start_frame + max_frames, total_available)

        for idx in range(start_frame, end_idx):
            rec = frame_records[idx]
            rel = rec["relative_paths"]

            mask_path = base_dir / rel["mask"]
            dist_path = base_dir / rel["distance"]

            if not mask_path.exists():
                raise PreprocessedDataError(f"Missing mask file: {mask_path}")
            if not dist_path.exists():
                raise PreprocessedDataError(f"Missing distance file: {dist_path}")

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise PreprocessedDataError(f"Failed to decode mask: {mask_path}")

            try:
                dist = np.load(str(dist_path))
            except Exception as e:
                raise PreprocessedDataError(f"Failed to load distance numpy array: {dist_path}") from e

            raw_rgb = None
            if load_raw_frames and "frame" in rel:
                frame_path = base_dir / rel["frame"]
                if frame_path.exists():
                    raw_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                    if raw_bgr is not None:
                        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

            polarity = rec.get("polarity", {})
            morph = rec.get("morphometry", {})

            yield PreprocessedFrameItem(
                frame_idx=rec.get("frame_idx", idx),
                src_frame_idx=rec.get("src_frame_idx", idx * 2),
                timestamp_sec=rec.get("timestamp_sec", idx / 15.0),
                mask=mask,
                dist=dist,
                raw_frame=raw_rgb,
                is_white_background=polarity.get("is_white_background", True),
                is_polarity_flip=polarity.get("polarity_flip_transition", False),
                character_bbox=tuple(morph.get("character_bbox", (0, 0, 0, 0))) if morph.get("character_bbox") else None,
                max_distance=float(morph.get("max_distance", dist.max())),
            )
    else:
        # Fallback directory globbing
        masks_dir = base_dir / "masks"
        dist_dir = base_dir / "distance"
        if not masks_dir.exists() or not dist_dir.exists():
            raise PreprocessedDataError(f"Required subdirectories 'masks' and 'distance' missing in {base_dir}")

        mask_files = sorted(list(masks_dir.glob("mask_*.png")))
        if not mask_files:
            raise PreprocessedDataError(f"No mask files found in {masks_dir}")

        total_available = len(mask_files)
        end_idx = total_available if max_frames is None else min(start_frame + max_frames, total_available)

        for idx in range(start_frame, end_idx):
            mask_path = mask_files[idx]
            dist_path = dist_dir / f"dist_{idx:04d}.npy"
            if not dist_path.exists():
                raise PreprocessedDataError(f"Distance array missing for frame {idx}: {dist_path}")

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            dist = np.load(str(dist_path))

            yield PreprocessedFrameItem(
                frame_idx=idx,
                src_frame_idx=idx * 2,
                timestamp_sec=idx / 15.0,
                mask=mask,
                dist=dist,
                max_distance=float(dist.max()),
            )


@dataclass
class RenderTelemetry:
    """Telemetry data emitted after rendering each frame."""
    frame_idx: int                 # 0-based index of frame just rendered
    total_frames: int              # Total frame count in sequence
    fps_instantaneous: float       # Single frame rendering throughput (FPS)
    fps_rolling: float             # Exponential moving average FPS
    elapsed_sec: float             # Wall clock seconds elapsed since render started
    eta_sec: float                 # Estimated seconds remaining
    vram_allocated_mb: float       # PyTorch CUDA memory allocated in MB
    vram_reserved_mb: float        # PyTorch CUDA memory reserved in MB
    vram_max_mb: float             # Peak CUDA memory allocated in MB
    droplets_placed: int           # Number of EDT droplets injected this frame
    status: str                    # Status flag: "init", "advect_infill", "scene_cut", "normal"

    def format_log(self) -> str:
        """Formats telemetry as a clean, human-readable console line."""
        pct = ((self.frame_idx + 1) / max(self.total_frames, 1)) * 100.0
        return (
            f"[{self.frame_idx + 1:04d}/{self.total_frames:04d} ({pct:5.1f}%)] "
            f"FPS: {self.fps_rolling:4.1f} (inst: {self.fps_instantaneous:4.1f}) | "
            f"Elapsed: {self.elapsed_sec:5.1f}s | ETA: {self.eta_sec:5.1f}s | "
            f"VRAM: {self.vram_allocated_mb:4.1f}MB | Drops: {self.droplets_placed:2d} ({self.status})"
        )


@dataclass
class VideoRenderResult:
    """Summary statistics returned upon render completion."""
    output_path: Path
    total_frames_rendered: int
    duration_sec: float
    target_fps: float
    average_fps: float
    peak_vram_mb: float
    has_audio: bool
    success: bool
    telemetry_history: Optional[List[RenderTelemetry]] = None


@dataclass
class WatercolorVideoConfig:
    """Configuration for end-to-end watercolor video rendering pipeline."""
    preprocessed_dir: Union[str, Path] = "data/preprocessed"
    output_video_path: Union[str, Path] = "data/bad_apple_watercolor.mp4"
    audio_path: Optional[Union[str, Path]] = "data/bad_apple_audio.aac"
    width: int = 512
    height: int = 384
    fps: float = 15.0
    start_frame: int = 0
    max_frames: Optional[int] = None
    physics_substeps: int = 6
    physics_dt: float = 0.040
    edt_scale_cr: float = 0.75
    droplet_r_min: float = 4.0
    droplet_r_max: float = 50.0
    scene_cut_iou_threshold: float = 0.15
    enable_wake_dissipation: bool = True
    wake_decay_rate: float = 0.40
    encoder_crf: int = 18
    encoder_preset: str = "medium"
    encoder_backend: str = "auto"
    device: Optional[str] = None
    log_interval: int = 1


# Alias for backward-compatibility with explorer specifications
VideoPipelineConfig = WatercolorVideoConfig


class VideoWatercolorPipeline:
    """
    Transforms preprocessed Bad Apple silhouette frame sequences into a flowing watercolor animation.
    """

    def __init__(self, config: Optional[WatercolorVideoConfig] = None) -> None:
        self.config = config or WatercolorVideoConfig()

        if self.config.device:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            f"VideoWatercolorPipeline initialized on device: {self.device} "
            f"({self.config.width}x{self.config.height} @ {self.config.fps} FPS)"
        )

    def compute_iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """Computes Intersection-over-Union between two binary silhouette masks."""
        intersection = np.logical_and(mask_a > 127, mask_b > 127).sum()
        union = np.logical_or(mask_a > 127, mask_b > 127).sum()
        if union == 0:
            return 1.0
        return float(intersection) / float(union)

    def render(
        self,
        progress_callback: Optional[Callable[[RenderTelemetry], None]] = None,
    ) -> VideoRenderResult:
        """
        Executes end-to-end rendering loop across preprocessed video frames.
        
        Args:
            progress_callback: Optional callback invoked after each frame with RenderTelemetry.
            
        Returns:
            VideoRenderResult with final performance metrics and output path.
        """
        start_time = time.perf_counter()

        # 1. Initialize GPU Simulation Engine
        sim_config = VideoSimulationConfig(
            flow_preset=cv2.DISOPTICAL_FLOW_PRESET_FAST,
            iou_cut_threshold=self.config.scene_cut_iou_threshold,
            transition_mode=TransitionMode.SOFT_RESET,
            sim_substeps=self.config.physics_substeps,
            sim_dt=self.config.physics_dt,
            edt_scale_cr=self.config.edt_scale_cr,
            droplet_r_min=self.config.droplet_r_min,
            droplet_r_max=self.config.droplet_r_max,
            wake_decay_rate=self.config.wake_decay_rate if self.config.enable_wake_dissipation else 0.0,
        )

        engine = VideoWatercolorEngine(
            width=self.config.width,
            height=self.config.height,
            config=sim_config,
            device=self.device,
        )

        # 2. Count total available frames from preprocessed sequence
        meta_file = Path(self.config.preprocessed_dir) / "metadata.json"
        total_frames = 225
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    total_frames = len(meta.get("frames", []))
            except Exception:
                pass
        if self.config.max_frames:
            total_frames = min(total_frames, self.config.max_frames)

        frame_items_generator = stream_preprocessed_frames(
            preprocessed_dir=self.config.preprocessed_dir,
            start_frame=self.config.start_frame,
            max_frames=self.config.max_frames,
        )

        # 3. Initialize Video Encoder
        enc_config = VideoEncoderConfig(
            output_path=self.config.output_video_path,
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
            crf=self.config.encoder_crf,
            preset=self.config.encoder_preset,
            audio_path=self.config.audio_path,
            backend=self.config.encoder_backend,
            input_pix_fmt="rgb24",
        )
        encoder = create_video_encoder(enc_config)

        # 4. State tracking variables
        prev_mask: Optional[np.ndarray] = None
        rolling_fps = self.config.fps
        rendered_count = 0
        peak_vram = 0.0
        telemetries: List[RenderTelemetry] = []

        logger.info(
            f"Beginning render of {total_frames} frames to '{self.config.output_video_path}' "
            f"using {type(encoder).__name__} on {self.device}..."
        )

        try:
            with encoder:
                for item in frame_items_generator:
                    frame_start_t = time.perf_counter()
                    curr_mask = item.mask
                    curr_dist = item.dist

                    if prev_mask is None:
                        # Frame 0 Initialization: Full silhouette infill
                        status = "init"
                        droplets_placed = engine.initialize_full_silhouette(
                            curr_mask=curr_mask,
                            dist_map=curr_dist,
                            frame_idx=item.frame_idx,
                        )
                    else:
                        # Transition Check (IoU drop or metadata polarity flip)
                        iou = self.compute_iou(prev_mask, curr_mask)
                        is_scene_cut = (iou < self.config.scene_cut_iou_threshold) or item.is_polarity_flip

                        if is_scene_cut:
                            status = "scene_cut"
                            engine.soft_scene_reset()
                            droplets_placed = engine.initialize_full_silhouette(
                                curr_mask=curr_mask,
                                dist_map=curr_dist,
                                frame_idx=item.frame_idx,
                            )
                        else:
                            status = "advect_infill"
                            droplets_placed = engine.step_video_frame(
                                prev_mask=prev_mask,
                                curr_mask=curr_mask,
                                dist_map=curr_dist,
                                frame_idx=item.frame_idx,
                                substeps=self.config.physics_substeps,
                                dt=self.config.physics_dt,
                                is_scene_cut=False,
                            )

                    # Render composited RGB frame via Kubelka-Munk optics over static paper
                    rendered_rgb = engine.render_rgb()

                    # Stream to video encoder pipe
                    encoder.write_frame(rendered_rgb)

                    # Update history
                    prev_mask = curr_mask
                    rendered_count += 1

                    # Telemetry calculation
                    frame_elapsed = time.perf_counter() - frame_start_t
                    instantaneous_fps = 1.0 / max(frame_elapsed, 1e-4)
                    rolling_fps = 0.20 * instantaneous_fps + 0.80 * rolling_fps
                    total_elapsed = time.perf_counter() - start_time
                    remaining_frames = total_frames - rendered_count
                    eta = remaining_frames / max(rolling_fps, 1e-2)

                    vram_alloc = 0.0
                    vram_res = 0.0
                    if torch.cuda.is_available() and self.device.type == "cuda":
                        vram_alloc = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
                        vram_res = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
                        peak_vram = max(peak_vram, vram_alloc)

                    telemetry = RenderTelemetry(
                        frame_idx=item.frame_idx,
                        total_frames=total_frames,
                        fps_instantaneous=instantaneous_fps,
                        fps_rolling=rolling_fps,
                        elapsed_sec=total_elapsed,
                        eta_sec=eta,
                        vram_allocated_mb=vram_alloc,
                        vram_reserved_mb=vram_res,
                        vram_max_mb=peak_vram,
                        droplets_placed=droplets_placed,
                        status=status,
                    )
                    telemetries.append(telemetry)

                    if progress_callback:
                        progress_callback(telemetry)

                    if (item.frame_idx + 1) % self.config.log_interval == 0:
                        print(telemetry.format_log(), flush=True)

        except Exception as e:
            logger.error(f"Render failed at frame {rendered_count}: {e}")
            raise WatercolorPipelineError(f"Video watercolor rendering aborted: {e}") from e

        total_wall_time = time.perf_counter() - start_time
        avg_fps = rendered_count / max(total_wall_time, 1e-4)
        out_p = Path(self.config.output_video_path).resolve()

        print(
            f"\n[DONE] Render complete: {rendered_count} frames in {total_wall_time:.2f}s "
            f"({avg_fps:.1f} FPS) -> '{out_p}' ({out_p.stat().st_size if out_p.exists() else 0} bytes)",
            flush=True,
        )

        return VideoRenderResult(
            output_path=out_p,
            total_frames_rendered=rendered_count,
            duration_sec=rendered_count / self.config.fps,
            target_fps=self.config.fps,
            average_fps=avg_fps,
            peak_vram_mb=peak_vram,
            has_audio=bool(self.config.audio_path and Path(self.config.audio_path).exists()),
            success=True,
            telemetry_history=telemetries,
        )


def main() -> None:
    """Command-line interface to execute the video watercolor pipeline."""
    parser = argparse.ArgumentParser(description="Render Bad Apple watercolor animation.")
    parser.add_argument("--preprocessed-dir", default="data/preprocessed", help="Path to preprocessed assets")
    parser.add_argument("--output", default="data/bad_apple_watercolor.mp4", help="Output MP4 file path")
    parser.add_argument("--audio", default="data/bad_apple_audio.aac", help="Path to input AAC audio")
    parser.add_argument("--fps", type=float, default=15.0, help="Output frame rate")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to render (None for all)")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--substeps", type=int, default=6, help="Navier-Stokes substeps per frame")
    parser.add_argument("--backend", default="auto", choices=["auto", "ffmpeg", "opencv"], help="Encoder backend")
    args = parser.parse_args()

    config = WatercolorVideoConfig(
        preprocessed_dir=args.preprocessed_dir,
        output_video_path=args.output,
        audio_path=args.audio if Path(args.audio).exists() else None,
        fps=args.fps,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        physics_substeps=args.substeps,
        encoder_backend=args.backend,
    )

    pipeline = VideoWatercolorPipeline(config)
    pipeline.render()


if __name__ == "__main__":
    main()
