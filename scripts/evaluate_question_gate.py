"""질문 적합성 판단(question_gate)을 실제 LLM으로 평가한다. OPENAI_API_KEY가 필요하다(없으면 종료).

막으면 안 되는 질문이 막힌 것(오탐)과 막아야 하는 질문이 통과한 것(미탐)을 질문과 이유까지 보여준다.
프롬프트나 도메인 단어를 바꿀 때마다 돌려서 오탐이 늘지 않았는지 확인한다.

사용: OPENAI_API_KEY=... python -m scripts.evaluate_question_gate
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline.search import question_gate as gate

CASES = Path(__file__).resolve().parents[1] / "evaluation/data/question_gate_cases.json"


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 없어 실행하지 않습니다.")
        return 2
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    with ThreadPoolExecutor(4) as pool:
        decisions = list(pool.map(lambda case: gate.check_question_fit(case["query"]), cases))

    false_blocks = [(c, d) for c, d in zip(cases, decisions) if c["fit"] and not d.fit]
    misses = [(c, d) for c, d in zip(cases, decisions) if not c["fit"] and d.fit]
    fit_total = sum(c["fit"] for c in cases)
    unfit_total = len(cases) - fit_total
    llm_calls = sum(d.source == "llm" for d in decisions)
    print(f"막으면 안 되는 질문 {fit_total}개 중 막힘(오탐) {len(false_blocks)}개")
    print(f"막아야 하는 질문 {unfit_total}개 중 통과(미탐) {len(misses)}개  | LLM 호출 {llm_calls}회")
    for label, rows in (("오탐", false_blocks), ("미탐", misses)):
        for case, decision in rows:
            print(f"  [{label}] [{case['category']}] {case['query']} -> {decision.reason} ({decision.source})")
    return 1 if false_blocks else 0


if __name__ == "__main__":
    sys.exit(main())
