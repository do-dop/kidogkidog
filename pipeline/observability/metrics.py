"""영상 처리 단계의 시간과 실패 횟수를 Prometheus에 기록한다."""

from contextlib import contextmanager
import os

from prometheus_client import Counter, Histogram, start_http_server


STAGE_SECONDS = Histogram(
    "kidog_pipeline_stage_seconds",
    "Time spent in each pipeline stage, in seconds.",
    ["stage"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 180, 600),
)
STAGE_ERRORS = Counter(
    "kidog_pipeline_stage_errors_total",
    "Errors in each pipeline stage.",
    ["stage"],
)
CHUNKS = Counter(
    "kidog_pipeline_chunks_total",
    "Processed video chunks by outcome.",
    ["outcome"],
)
FRAMES = Counter(
    "kidog_pipeline_frames_total",
    "Frames indexed by the worker.",
)
QUEUE_WAIT_SECONDS = Histogram(
    "kidog_pipeline_queue_wait_seconds",
    "Time from API enqueue to worker start, in seconds.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 180, 600),
)
END_TO_END_SECONDS = Histogram(
    "kidog_pipeline_upload_to_index_seconds",
    "Time from simulator upload completion to durable index completion, in seconds.",
    buckets=(1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
)
VIDEO_SECONDS = Counter(
    "kidog_pipeline_video_seconds_total",
    "Duration of successfully indexed video chunks, in seconds.",
)
RTF = Histogram(
    "kidog_pipeline_real_time_factor",
    "Chunk processing wall time divided by source video duration.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)


@contextmanager
def measure_stage(stage):
    """허용된 단계 이름만 호출부에서 전달해 시계열 수를 제한한다."""
    try:
        with STAGE_SECONDS.labels(stage=stage).time():
            yield
    except Exception:
        STAGE_ERRORS.labels(stage=stage).inc()
        raise


def start_worker_metrics(*_args, **_kwargs):
    """Celery solo worker에서 Prometheus 수집용 HTTP 서버를 연다."""
    if os.getenv("METRICS_ENABLED", "false").lower() not in {"1", "true", "yes"}:
        return
    port = int(os.getenv("METRICS_PORT", "8002"))
    start_http_server(port, addr="0.0.0.0")
