import os
from typing import Any

import httpx


DEFAULT_JEV_BASE_URL = "https://api.typesafe.ai/v1"
DEFAULT_JEV_MODEL = "jev-1.13.0"


class JevClientError(RuntimeError):
    """Jev 호출 또는 응답 검증에 실패했을 때 발생한다."""


class JevClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = (
            os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY")
            if api_key is None
            else api_key
        )
        self.model = model or os.getenv(
            "TYPESAFE_DEFAULT_MODEL",
            os.getenv("JEV_MODEL", DEFAULT_JEV_MODEL),
        )
        self.base_url = (
            base_url
            or os.getenv("TYPESAFE_BASE_URL")
            or os.getenv("JEV_BASE_URL", DEFAULT_JEV_BASE_URL)
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds or float(os.getenv("JEV_TIMEOUT_SECONDS", "2.0"))
        self.transport = transport

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def classify_question(self, query: str) -> dict[str, Any]:
        if not self.api_key:
            raise JevClientError("JEV_API_KEY가 설정되어 있지 않습니다.")

        payload = {
            "model": self.model,
            "state": {"user_query": query},
            "questions": {
                "temporal_scope": {
                    "type": "choice",
                    "instructions": "질문에 답하기 위해 필요한 최소 영상 증거 범위를 분류하세요.",
                    "criteria": {
                        "static": "한 시점의 프레임만 보면 답할 수 있다.",
                        "event": "하나의 연속된 행동 구간을 봐야 한다.",
                        "sequence": "둘 이상의 행동이나 상태의 시간적 순서를 확인해야 한다.",
                        "aggregate": "전체 기간의 횟수, 빈도 또는 누적 시간을 계산해야 한다.",
                    },
                },
                "requires_multiple_frames": {
                    "type": "noul",
                    "instructions": "이 질문은 여러 프레임이나 행동 구간을 확인해야 합니까?",
                },
                "requires_event_order": {
                    "type": "noul",
                    "instructions": "이 질문은 둘 이상의 행동이나 상태가 발생한 순서를 비교해야 합니까?",
                },
                "requires_aggregation": {
                    "type": "noul",
                    "instructions": "이 질문은 횟수, 빈도, 총시간 또는 기간별 비교가 필요합니까?",
                },
                "requires_time_filter": {
                    "type": "noul",
                    "instructions": "이 질문은 특정 날짜나 시간대에 결과를 제한해야 합니까?",
                },
            },
        }

        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.post(
                    "/systemone",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise JevClientError(f"Jev 질문 분류 요청에 실패했습니다: {exc}") from exc

        if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
            raise JevClientError("Jev 응답에 answers 객체가 없습니다.")

        return body
