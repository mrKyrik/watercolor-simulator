# Project: Experimental Watercolor Video Maker (Bad Apple)

## Architecture
The system transforms video footage into a flowing watercolor animation through an integrated 4-stage pipeline:
1. **Video Ingestion & Acquisition (`pipeline/acquisition.py`)**: Fetches a 10–20 second segment of the iconic Bad Apple shadow-art video using `yt-dlp` from verified unthrottled sources (Internet Archive / Nico Nico Douga), isolating audio and video streams.
2. **Computer Vision Silhouette Preprocessing (`pipeline/preprocessor.py`)**: Processes video frames using Otsu's bimodal thresholding, automated border polarity normalization, 2-level contour hierarchy extraction with interior hole preservation (`cv2.RETR_CCOMP`), and Euclidean Distance Transforms ($L_2$).
3. **Fluid Simulation & Watercolor Video Engine (`pipeline/watercolor_video.py`, `engine/video_simulation.py`)**: Extends existing PyTorch CUDA physics modules (`engine/fluid.py`, `engine/pigment.py`, `engine/optics.py`, `engine/paper.py`):
   - **Hybrid Optical Flow State Advection**: Employs OpenCV `DISOpticalFlow` coupled with PyTorch Semi-Lagrangian grid sampling (`F.grid_sample`) to advect persistent fluid height and pigment tensors between frames without boiling.
   - **EDT-Weighted Incremental Droplet Infilling**: Distributes randomized droplets scaled by distance transform depth ($r \propto DT(x,y)$) into newly uncovered silhouette regions ($\Delta M_{\text{new}} = M_t \setminus M_{t-1}$).
   - **Harmonic Multi-Chromatic Color Field**: Generates multi-pigment triads (Prussian Blue, Burnt Sienna, Quinacridone Rose, Gamboge Yellow, Viridian Green) mapped through 3D Perlin noise and mixed via Kubelka-Munk Duncan subtractive optics.
   - **Stationary Paper Grounding**: Maintains paper substrate topography static in screen space, allowing pigment to precipitate naturally into paper valleys as silhouettes move.
4. **Video Assembly & Audio Muxing (`pipeline/encoder.py`)**: Encodes simulated frames via an FFmpeg H.264 pipe, multiplexing original audio into a playable, smooth MP4 video (5–15 seconds, >= 12 FPS).
5. **Collaborative Documentation (`progress_report.md`)**: Documents technical architecture, multi-agent discussion notes and trade-offs, and rendering performance/results.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Automated yt-dlp Video Acquisition | Fetch 10–20s Bad Apple video segment from unauthenticated working archive source with section slicing | M1 | Survey 1, Survey 2 |
| 2 | Audio Extraction | Demux original AAC/MP3 audio track for later video remuxing | M1 | Survey 2, Survey 3 |
| 3 | Polarity-Aware Bimodal Thresholding | Otsu thresholding with border-sampling heuristic to normalize silhouette foreground to 255 | M1 | Survey 2 |
| 4 | Contour Hierarchy & Hole Preservation | Extract outer contours and inner negative spaces using `cv2.RETR_CCOMP` | M1 | Survey 2 |
| 5 | Euclidean Distance Transform Mapping | Compute $L_2$ distance transform on silhouette masks for depth-aware droplet scaling | M1 | Survey 2, Survey 3 |
| 6 | DIS Optical Flow Computation | Calculate dense motion vectors ($u, v$) between consecutive silhouette frames using `cv2.DISOpticalFlow` | M2 | Survey 3 |
| 7 | Semi-Lagrangian GPU State Advection | Advect fluid height $h$ and pigment $c, d$ tensors using `F.grid_sample` to eliminate boiling | M2 | Survey 3 |
| 8 | EDT-Weighted Incremental Droplet Infilling | Procedurally place droplets scaled to local clearance exclusively in newly uncovered silhouette areas | M2 | Survey 3 |
| 9 | 3D Perlin Multi-Chromatic Palette Mixing | Sample fine-art watercolor triads via Perlin space, mixed via Kubelka-Munk Duncan subtractive optics | M2 | Survey 1, Survey 3 |
| 10 | Screen-Space Static Paper Grounding | Fix paper texture topography in screen space to anchor physical pigment sedimentation | M2 | Survey 3 |
| 11 | Scene-Cut & Wipe Transition Detection | Detect silhouette IoU drop (<0.15) and perform soft state decay/reset on scene changes | M2 | Survey 2, Survey 3 |
| 12 | Video Encoding & Audio Muxing | Stream frames to FFmpeg H.264 / OpenCV VideoWriter and mux audio into final MP4 | M2 | Survey 1, Survey 3 |
| 13 | Multi-Agent Discussion & Trade-offs Log | Document collaborative technical discussions, architectural choices, and design alternatives | M3 | R3, Survey 1-3 |
| 14 | Progress Report & Performance Synthesis | Compile comprehensive `progress_report.md` covering architecture, benchmarks, and results | M3 | R3, Acceptance |
| 15 | End-to-End Test Suite & Verification | Automated verification of video duration (5–15s), FPS (>=12), playable MP4, droplet dynamics, and docs | M4 | Acceptance |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Video Acquisition & Preprocessing | Features 1, 2, 3, 4, 5: `pipeline/acquisition.py`, `pipeline/preprocessor.py` | none | DONE |
| M2 | Flowing Watercolor Video Engine | Features 6, 7, 8, 9, 10, 11, 12: `pipeline/watercolor_video.py`, `engine/video_simulation.py` | M1 | IN_PROGRESS |
| M3 | Collaborative Documentation | Features 13, 14: `progress_report.md` in workspace root | M1, M2 | PLANNED |
| M4 | E2E Testing & Acceptance | Feature 15: `tests/test_e2e_video.py` and acceptance criteria verification | M1, M2, M3 | PLANNED |

## Interface Contracts
### `pipeline.acquisition` ↔ `pipeline.preprocessor`
- Input: Target URL / identifier (default: Internet Archive `bad-apple-pv_202307` or `nicovideo-sm8628149`), time range string (e.g. `00:30-00:45`), output directory.
- Output: `data/bad_apple_clip.mp4` (video), `data/bad_apple_audio.aac` (audio), metadata dictionary (duration, fps, resolution, total_frames).
- Error handling: Fallback to secondary mirror if primary network stream fails; validate non-empty file > 500 KB.

### `pipeline.preprocessor` ↔ `pipeline.watercolor_video`
- Input: `data/bad_apple_clip.mp4`
- Output directory: `data/preprocessed/` containing:
  - `frames/frame_%04d.png`: Raw RGB frames.
  - `masks/mask_%04d.png`: Binary uint8 silhouette mask (foreground = 255, background = 0).
  - `distance/dist_%04d.npy` or `dist_%04d.png`: $L_2$ Euclidean distance transform.
  - `metadata.json`: Video metadata (FPS, count, resolution, detected polarity flip frames).
- Signatures: `class VideoPreprocessor.process_video(clip_path, out_dir, target_fps=15, target_res=(512, 384))`

### `pipeline.watercolor_video` ↔ `pipeline.encoder`
- Input: Sequence of preprocessed masks and raw frames from `data/preprocessed/`.
- Method: `VideoWatercolorPipeline.render(preprocessed_dir, output_mp4, audio_path=None)`
- Output: Validated MP4 video file (duration 5–15 seconds, FPS >= 12, H.264 codec, stereo audio).

## Code Layout
- `engine/`: Existing GPU watercolor physics modules (`fluid.py`, `pigment.py`, `optics.py`, `paper.py`, `noise.py`, `simulation.py`).
- `pipeline/`:
  - `__init__.py`: Package initialization.
  - `acquisition.py`: Video download and audio extraction using `yt-dlp` and `ffmpeg`.
  - `preprocessor.py`: Frame extraction, Otsu bimodal thresholding, polarity detection, contour hierarchy, distance transform.
  - `watercolor_video.py`: End-to-end video pipeline combining optical flow state advection, incremental droplet infilling, and rendering.
- `tests/`:
  - `test_physics.py`, etc.: Existing 14 unit test suites (remain untouched and passing).
  - `test_e2e_video.py`: End-to-end integration and acceptance tests.
- `progress_report.md`: User-facing progress report in workspace root.
- `data/`: Working directory for downloaded video clips, extracted frames, and output animations.
