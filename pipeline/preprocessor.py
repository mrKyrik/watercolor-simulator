"""
pipeline/preprocessor.py - Video Frame Preprocessor for Bad Apple Shadow-Art Footage.

Implements sequential downsampling (15 FPS), Otsu bimodal thresholding, order-statistic
border perimeter median polarity sampling, 2-level contour hierarchy extraction (cv2.RETR_CCOMP)
with hole preservation, float32 Euclidean Distance Transform (L2), and structured metadata generation.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import json
import logging
import os

import cv2
import numpy as np

logger = logging.getLogger("pipeline.preprocessor")


class PreprocessingError(RuntimeError):
    """Raised when computer vision frame preprocessing fails."""
    pass


class NumpyJSONEncoder(json.JSONEncoder):
    """Encodes NumPy scalar and array types into standard Python primitives for JSON."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class FrameMetadata:
    """Metadata and morphology statistics for a single preprocessed frame."""
    frame_idx: int
    src_frame_idx: int
    timestamp_sec: float
    relative_paths: Dict[str, str]
    polarity: Dict[str, Any]
    morphometry: Dict[str, Any]
    topology: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreprocessingSummary:
    """Sequence-level summary statistics of preprocessed video assets."""
    video_metadata: Dict[str, Any]
    summary_statistics: Dict[str, Any]
    manifest_path: str
    total_frames: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VideoPreprocessor:
    """
    Transforms raw video frames into normalized binary silhouette masks,
    Euclidean distance transform arrays, contour hierarchies, and structured metadata.
    """

    def __init__(
        self,
        target_fps: float = 15.0,
        target_resolution: Optional[Tuple[int, int]] = (512, 384),
        border_strip_px: int = 2,
        morph_cleanup: bool = False,
        min_contour_area: float = 4.0,
        save_dist_vis: bool = True
    ) -> None:
        """
        Args:
            target_fps: Frame rate for animation (default: 15.0 for animation 'on twos').
            target_resolution: (width, height) to resize frames, or None for native.
            border_strip_px: Thickness of outer boundary perimeter band for polarity sampling.
            morph_cleanup: If True, applies 3x3 morphological opening for isolated noise.
            min_contour_area: Minimum pixel area to consider a contour.
            save_dist_vis: If True, saves normalized uint8 preview image of distance transform.
        """
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        self.border_strip_px = border_strip_px
        self.morph_cleanup = morph_cleanup
        self.min_contour_area = min_contour_area
        self.save_dist_vis = save_dist_vis

    def detect_polarity(self, image: np.ndarray) -> Tuple[bool, float, float]:
        """
        Samples the outer perimeter band of width border_strip_px to determine background color.
        Returns:
            (is_white_background, border_median, border_mean)
        """
        H, W = image.shape[:2]
        K = min(self.border_strip_px, H // 2, W // 2)
        if K <= 0:
            K = 1

        top = image[0:K, :]
        bottom = image[H - K:H, :]
        left = image[:, 0:K]
        right = image[:, W - K:W]

        border_pixels = np.concatenate([
            top.ravel(),
            bottom.ravel(),
            left.ravel(),
            right.ravel()
        ])

        border_median = float(np.median(border_pixels))
        border_mean = float(np.mean(border_pixels))
        is_white_bg = bool(border_median > 128.0)
        return is_white_bg, border_median, border_mean

    def threshold_frame(
        self,
        gray: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, float, float]:
        """
        Performs Otsu's bimodal thresholding and normalizes polarity.
        Ensures that character silhouette foreground is ALWAYS 255 (white)
        and canvas/paper background is ALWAYS 0 (black).
        Returns:
            (normalized_mask, otsu_threshold, is_white_bg, border_median, border_mean)
        """
        otsu_val, raw_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        is_white_bg, b_median, b_mean = self.detect_polarity(raw_thresh)

        if is_white_bg:
            # Dark character on white background -> invert
            norm_mask = cv2.bitwise_not(raw_thresh)
        else:
            # Light character on dark background -> keep
            norm_mask = raw_thresh.copy()

        if self.morph_cleanup:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            norm_mask = cv2.morphologyEx(norm_mask, cv2.MORPH_OPEN, kernel)

        return norm_mask, float(otsu_val), is_white_bg, b_median, b_mean

    def extract_contours_and_holes(
        self,
        mask: np.ndarray
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray], int, int]:
        """
        Extracts 2-level contour hierarchy using cv2.RETR_CCOMP to preserve
        exterior boundaries and interior negative spaces (holes).
        Returns:
            (contours, hierarchy, external_contour_count, hole_contour_count)
        """
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or len(contours) == 0:
            return [], None, 0, 0

        hier_data = hierarchy[0]
        # In OpenCV RETR_CCOMP:
        # h[3] == -1 indicates an external outer contour.
        # h[3] != -1 indicates a hole boundary whose parent is the enclosing outer contour.
        ext_count = int(sum(1 for h in hier_data if h[3] == -1))
        hole_count = int(sum(1 for h in hier_data if h[3] != -1))
        return contours, hierarchy, ext_count, hole_count

    def compute_distance_transform(
        self,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Computes sub-pixel L2 Euclidean Distance Transform on foreground silhouette.
        Returns:
            (dist_map_float32, max_distance, (max_x, max_y))
        """
        # cv2.distanceTransform calculates distance to zero pixels
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        min_val, max_dist, min_loc, max_loc = cv2.minMaxLoc(dist_map)
        return dist_map, float(max_dist), (int(max_loc[0]), int(max_loc[1]))

    def compute_bounding_box(
        self,
        mask: np.ndarray
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        Computes bounding box encompassing all foreground pixels.
        Returns ((x, y, w, h), (xmin, ymin, xmax, ymax)).
        Safely handles empty masks.
        """
        pts = cv2.findNonZero(mask)
        if pts is None or len(pts) == 0:
            return (0, 0, 0, 0), (0, 0, 0, 0)
        x, y, w, h = cv2.boundingRect(pts)
        return (int(x), int(y), int(w), int(h)), (int(x), int(y), int(x + w), int(y + h))

    def compute_moments(self, mask: np.ndarray) -> Tuple[float, float]:
        """Computes center of mass (cx, cy) of silhouette. Returns (0.0, 0.0) if empty."""
        m = cv2.moments(mask)
        if m["m00"] > 0:
            return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
        return 0.0, 0.0

    def process_video(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path] = "data/preprocessed",
        audio_path: Optional[Union[str, Path]] = "data/bad_apple_audio.aac",
        max_frames: Optional[int] = None,
        save_frames: bool = True,
        save_masks: bool = True,
        save_distance_npy: bool = True,
        save_distance_png: bool = True,
        save_contours_png: bool = True
    ) -> Dict[str, Any]:
        """
        Executes end-to-end extraction and preprocessing.
        Populates output_dir with:
          - frames/frame_%04d.png
          - masks/mask_%04d.png
          - distance/dist_%04d.npy
          - distance/dist_%04d.png (visualization)
          - contours/contour_%04d.png
          - metadata.json
        """
        v_path = Path(video_path)
        if not v_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {v_path}")

        out_root = Path(output_dir)
        frames_dir = out_root / "frames"
        masks_dir = out_root / "masks"
        dist_dir = out_root / "distance"
        contours_dir = out_root / "contours"

        if save_frames:
            frames_dir.mkdir(parents=True, exist_ok=True)
        if save_masks:
            masks_dir.mkdir(parents=True, exist_ok=True)
        if save_distance_npy or save_distance_png:
            dist_dir.mkdir(parents=True, exist_ok=True)
        if save_contours_png:
            contours_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            raise PreprocessingError(f"OpenCV failed to open video file: {v_path}")

        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        src_duration = total_src_frames / src_fps if src_fps > 0 else 0.0

        target_w, target_h = self.target_resolution if self.target_resolution else (src_w, src_h)
        frame_interval = max(1, int(round(src_fps / self.target_fps)))

        logger.info(
            "Preprocessing: source %dx%d @ %.1ffps (%d frames) -> target %dx%d @ %.1ffps (step %d)",
            src_w, src_h, src_fps, total_src_frames, target_w, target_h, self.target_fps, frame_interval
        )

        frames_manifest: List[Dict[str, Any]] = []
        out_frame_idx = 0
        src_frame_idx = 0
        total_polarity_flips = 0
        prev_is_white_bg: Optional[bool] = None
        inverted_source_frames_count = 0
        frames_with_holes_count = 0
        max_thickness_overall = 0.0
        area_ratios: List[float] = []

        while src_frame_idx < total_src_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame stride selection
            if src_frame_idx % frame_interval == 0:
                frame_stem = f"frame_{out_frame_idx:04d}"
                mask_stem = f"mask_{out_frame_idx:04d}"
                dist_stem = f"dist_{out_frame_idx:04d}"
                contour_stem = f"contour_{out_frame_idx:04d}"
                timestamp_sec = round(src_frame_idx / src_fps, 4)

                # Resize if target resolution differs from source
                if (src_w, src_h) != (target_w, target_h):
                    interp = cv2.INTER_AREA if (target_w < src_w and target_h < src_h) else cv2.INTER_LINEAR
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=interp)

                # 1. Grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 2. Otsu thresholding + Polarity Normalization
                norm_mask, otsu_t, is_white_bg, b_median, b_mean = self.threshold_frame(gray)

                # Track polarity flips
                is_flip_transition = False
                if prev_is_white_bg is not None and is_white_bg != prev_is_white_bg:
                    total_polarity_flips += 1
                    is_flip_transition = True
                    logger.info("Polarity flip detected at frame %d (t=%.3fs): %s -> %s",
                                out_frame_idx, timestamp_sec, prev_is_white_bg, is_white_bg)
                prev_is_white_bg = is_white_bg

                if is_white_bg:
                    inverted_source_frames_count += 1

                # 3. Contour hierarchy & hole preservation
                contours, hierarchy, num_ext, num_holes = self.extract_contours_and_holes(norm_mask)
                has_holes = bool(num_holes > 0)
                if has_holes:
                    frames_with_holes_count += 1

                # 4. Euclidean Distance Transform
                dist_map, max_dist, max_loc = self.compute_distance_transform(norm_mask)
                if max_dist > max_thickness_overall:
                    max_thickness_overall = max_dist

                # 5. Geometry: Bounding Box and Centroid
                bbox_xywh, bbox_xyxy = self.compute_bounding_box(norm_mask)
                cx, cy = self.compute_moments(norm_mask)

                # 6. Silhouette Area Metrics
                fg_pixels = int(np.count_nonzero(norm_mask))
                area_ratio = float(fg_pixels / (target_w * target_h))
                area_ratios.append(area_ratio)

                # Relative paths dictionary
                rel_paths = {
                    "frame": f"frames/{frame_stem}.png",
                    "mask": f"masks/{mask_stem}.png",
                    "distance": f"distance/{dist_stem}.npy",
                    "distance_vis": f"distance/{dist_stem}.png",
                    "contour": f"contours/{contour_stem}.png"
                }

                # Save assets to disk
                if save_frames:
                    cv2.imwrite(str(frames_dir / f"{frame_stem}.png"), frame)

                if save_masks:
                    cv2.imwrite(str(masks_dir / f"{mask_stem}.png"), norm_mask)

                if save_distance_npy:
                    np.save(str(dist_dir / f"{dist_stem}.npy"), dist_map.astype(np.float32))

                if save_distance_png and self.save_dist_vis:
                    dist_vis = (dist_map / max(1.0, max_dist) * 255.0).astype(np.uint8)
                    cv2.imwrite(str(dist_dir / f"{dist_stem}.png"), dist_vis)

                if save_contours_png:
                    contour_img = np.zeros((target_h, target_w), dtype=np.uint8)
                    if len(contours) > 0 and hierarchy is not None:
                        cv2.drawContours(contour_img, contours, -1, 255, thickness=1, hierarchy=hierarchy, maxLevel=1)
                    cv2.imwrite(str(contours_dir / f"{contour_stem}.png"), contour_img)

                # Record frame metadata
                frame_record = {
                    "frame_idx": out_frame_idx,
                    "src_frame_idx": src_frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "relative_paths": rel_paths,
                    "polarity": {
                        "detected_bg": "white" if is_white_bg else "black",
                        "is_white_background": is_white_bg,
                        "is_inverted_in_source": is_white_bg,
                        "polarity_flip_transition": is_flip_transition,
                        "border_median": b_median,
                        "border_mean": round(b_mean, 2)
                    },
                    "morphometry": {
                        "otsu_threshold": round(otsu_t, 2),
                        "foreground_pixels": fg_pixels,
                        "area_ratio": round(area_ratio, 4),
                        "centroid": [round(cx, 1), round(cy, 1)],
                        "character_bbox": list(bbox_xywh),
                        "character_bbox_xyxy": list(bbox_xyxy),
                        "max_distance": round(max_dist, 2),
                        "max_distance_loc": list(max_loc)
                    },
                    "topology": {
                        "total_contours": len(contours),
                        "external_contours": num_ext,
                        "hole_contours": num_holes,
                        "has_holes": has_holes
                    }
                }
                frames_manifest.append(frame_record)

                out_frame_idx += 1
                if max_frames is not None and out_frame_idx >= max_frames:
                    break

            src_frame_idx += 1

        cap.release()

        # Build top-level manifest
        manifest_data = {
            "video_metadata": {
                "source_file": v_path.as_posix(),
                "audio_file": Path(audio_path).as_posix() if audio_path else None,
                "source_fps": round(src_fps, 2),
                "source_frame_count": total_src_frames,
                "source_resolution": [src_w, src_h],
                "source_duration_sec": round(src_duration, 3),
                "target_fps": self.target_fps,
                "target_resolution": [target_w, target_h],
                "frame_interval": frame_interval,
                "total_preprocessed_frames": out_frame_idx
            },
            "summary_statistics": {
                "total_polarity_flips": total_polarity_flips,
                "inverted_source_frames_count": inverted_source_frames_count,
                "frames_with_holes": frames_with_holes_count,
                "max_interior_thickness_overall": round(float(max_thickness_overall), 2),
                "mean_silhouette_area_ratio": round(float(np.mean(area_ratios)), 4) if area_ratios else 0.0
            },
            "frames": frames_manifest
        }

        manifest_path = out_root / "metadata.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, cls=NumpyJSONEncoder)

        logger.info(
            "Preprocessing complete. Extracted %d frames to %s (Manifest: %s)",
            out_frame_idx, out_root, manifest_path
        )

        return manifest_data


def preprocess_video(
    video_path: Union[str, Path] = "data/bad_apple_clip.mp4",
    output_dir: Union[str, Path] = "data/preprocessed",
    audio_path: Optional[Union[str, Path]] = "data/bad_apple_audio.aac",
    target_fps: float = 15.0,
    target_resolution: Optional[Tuple[int, int]] = (512, 384)
) -> Dict[str, Any]:
    """Convenience functional wrapper for VideoPreprocessor."""
    pre = VideoPreprocessor(target_fps=target_fps, target_resolution=target_resolution)
    return pre.process_video(video_path=video_path, output_dir=output_dir, audio_path=audio_path)
