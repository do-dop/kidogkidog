import json
import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import httpx

from pipeline.search.jev_client import JevClient
from pipeline.search.question_router import (
    QuestionType,
    classify_question,
    classify_with_rules,
    run_shadow_classification,
)


def jev_response(scope="event", confidence=0.9, **signals):
    probabilities = {"static": 0.02, "event": 0.02, "sequence": 0.02, "aggregate": 0.02}
    probabilities[scope] = confidence
    defaults = {
        "requires_multiple_frames": 0.9,
        "requires_event_order": 0.1,
        "requires_aggregation": 0.1,
        "requires_time_filter": 0.1,
    }
    defaults.update(signals)
    # 실제 Jev /systemone 응답 형식 (jev-1.13.0 기준)
    return {
        "model": "jev-test",
        "answers": {
            "temporal_scope": {
                "type": "choice",
                "choice": scope,
                "confidence": confidence,
                "probabilities": probabilities,
            },
            **{key: {"type": "noul", "noul": value} for key, value in defaults.items()},
        },
        "usage": {"input_tokens": 563, "output_tokens": 122},
    }


class QuestionRouterRuleTests(unittest.TestCase):
    def test_static_question(self):
        result = classify_with_rules("강아지가 소파 위에 있나요?")
        self.assertEqual(result.question_type, QuestionType.STATIC)
        self.assertLess(result.requires_multiple_frames, 0.5)

    def test_event_question(self):
        result = classify_with_rules("강아지가 물을 마셨나요?")
        self.assertEqual(result.question_type, QuestionType.EVENT)
        self.assertGreater(result.requires_multiple_frames, 0.7)

    def test_sequence_takes_precedence_over_event(self):
        result = classify_with_rules("강아지가 밥을 먹은 뒤 물을 마셨나요?")
        self.assertEqual(result.question_type, QuestionType.SEQUENCE)
        self.assertEqual(result.requires_event_order, 1.0)

    def test_aggregate_takes_precedence_and_detects_time_filter(self):
        result = classify_with_rules("오늘 오후에 물을 몇 번 마셨나요?")
        self.assertEqual(result.question_type, QuestionType.AGGREGATE)
        self.assertEqual(result.requires_aggregation, 1.0)
        self.assertEqual(result.requires_time_filter, 1.0)


class JevClientTests(unittest.TestCase):
    def test_request_and_response_contract(self):
        def handler(request):
            self.assertEqual(request.url.path, "/api/v1/systemone")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            body = json.loads(request.content)
            self.assertEqual(body["state"]["user_query"], "물을 마셨나요?")
            self.assertEqual(body["questions"]["temporal_scope"]["type"], "choice")
            return httpx.Response(200, json=jev_response())

        client = JevClient(
            api_key="test-key",
            base_url="https://example.test/api/v1",
            transport=httpx.MockTransport(handler),
        )
        result = classify_question("물을 마셨나요?", client=client)
        self.assertEqual(result.question_type, QuestionType.EVENT)
        self.assertEqual(result.source, "jev")
        self.assertEqual(result.model_version, "jev-test")

    def test_low_confidence_uses_hybrid(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(
            200, json=jev_response(scope="event", confidence=0.45)
        ))
        client = JevClient(api_key="test-key", base_url="https://example.test", transport=transport)
        result = classify_question("무슨 일이 있었나요?", client=client)
        self.assertEqual(result.question_type, QuestionType.HYBRID)
        self.assertEqual(result.source, "low_confidence_fallback")

    def test_confidence_between_fallback_and_high_threshold_keeps_jev_type(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(
            200, json=jev_response(scope="event", confidence=0.55)
        ))
        client = JevClient(api_key="test-key", base_url="https://example.test", transport=transport)
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("QUESTION_ROUTER_FALLBACK_THRESHOLD", None)
            result = classify_question("강아지가 놀랐나요?", client=client)
        self.assertEqual(result.question_type, QuestionType.EVENT)
        self.assertEqual(result.source, "jev_with_fallback")

    def test_api_failure_falls_back_to_rules(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(503))
        client = JevClient(api_key="test-key", base_url="https://example.test", transport=transport)
        result = classify_question("오늘 몇 번 물을 마셨나요?", client=client)
        self.assertEqual(result.question_type, QuestionType.AGGREGATE)
        self.assertEqual(result.source, "rule_jev_fallback")
        self.assertIsNotNone(result.error)

    def test_missing_key_uses_rules_without_network(self):
        client = JevClient(api_key="")
        result = classify_question("밥을 먹은 뒤 잠들었나요?", client=client)
        self.assertEqual(result.question_type, QuestionType.SEQUENCE)
        self.assertEqual(result.source, "rule_no_jev_key")


class ShadowModeTests(unittest.TestCase):
    def test_off_mode_does_nothing(self):
        with patch.dict("os.environ", {"QUESTION_ROUTER_MODE": "off"}):
            self.assertIsNone(run_shadow_classification("강아지가 있나요?"))

    def test_shadow_log_does_not_contain_raw_query(self):
        query = "강아지가 소파 위에 있나요?"
        output = StringIO()
        with patch.dict("os.environ", {"QUESTION_ROUTER_MODE": "shadow", "JEV_API_KEY": ""}), \
             redirect_stdout(output):
            result = run_shadow_classification(query, client=JevClient(api_key=""))

        log = output.getvalue()
        self.assertEqual(result.question_type, QuestionType.STATIC)
        self.assertNotIn(query, log)
        self.assertIn("question_route_shadow", log)


if __name__ == "__main__":
    unittest.main()
