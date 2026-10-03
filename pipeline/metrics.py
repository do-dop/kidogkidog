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
