"""
pipeline - Video Acquisition and Watercolor Preprocessing Pipeline.
"""

from pipeline.acquisition import (
    VideoAcquisition,
    VideoAcquisitionConfig,
    VideoMetadata,
    VideoProbeResult,
    AcquisitionError,
    acquire_bad_apple,
    parse_time_to_seconds,
    parse_section_range,
)

from pipeline.preprocessor import (
    VideoPreprocessor,
    FrameMetadata,
    PreprocessingSummary,
    PreprocessingError,
    preprocess_video,
    NumpyJSONEncoder,
)

from pipeline.encoder import (
    VideoEncoderConfig,
    create_video_encoder,
    FFmpegPipeEncoder,
    OpenCVFallbackEncoder,
    mux_audio_to_video,
    VideoEncoder,
    VideoEncodingError,
    AudioMuxingError,
)

from pipeline.watercolor_video import (
    VideoWatercolorPipeline,
    WatercolorVideoConfig,
    VideoPipelineConfig,
    stream_preprocessed_frames,
    PreprocessedFrameItem,
    RenderTelemetry,
    VideoRenderResult,
    WatercolorPipelineError,
)

__all__ = [
    "VideoAcquisition",
    "VideoAcquisitionConfig",
    "VideoMetadata",
    "VideoProbeResult",
    "AcquisitionError",
    "acquire_bad_apple",
    "parse_time_to_seconds",
    "parse_section_range",
    "VideoPreprocessor",
    "FrameMetadata",
    "PreprocessingSummary",
    "PreprocessingError",
    "preprocess_video",
    "NumpyJSONEncoder",
    "VideoEncoderConfig",
    "create_video_encoder",
    "FFmpegPipeEncoder",
    "OpenCVFallbackEncoder",
    "mux_audio_to_video",
    "VideoEncoder",
    "VideoEncodingError",
    "AudioMuxingError",
    "VideoWatercolorPipeline",
    "WatercolorVideoConfig",
    "VideoPipelineConfig",
    "stream_preprocessed_frames",
    "PreprocessedFrameItem",
    "RenderTelemetry",
    "VideoRenderResult",
    "WatercolorPipelineError",
]
