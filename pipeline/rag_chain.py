import os
from typing import Any

from pipeline.vector_store import search
from pipeline.query_analyzer import build_prompt_hint


DEFAULT_FALLBACK_MESSAGE = (
    "현재 OPENAI_API_KEY가 설정되어 있지 않아 LLM 답변은 생성하지 못했습니다. "
    "대신 검색된 프레임의 시간, 감지 객체, 유사도 점수를 기준으로 결과를 확인해주세요."
)


def run_rag_query(
    query: str,
    video_id: str | None = None,
    top_k: int = 5,
    user_id: str | None = None,
) -> dict:
    """
    사용자 질문을 받아 ChromaDB에서 관련 프레임을 검색하고,
    검색 결과 metadata를 바탕으로 LLM 답변을 생성한다.

    현재 단계에서는 이미지를 직접 LLM에 넣지 않고,
    검색된 프레임의 timestamp, object_labels, score, video_id를 근거로 답변한다.
    """
    retrieved_frames = search(
        query=query,
        top_k=top_k,
        video_id=video_id,
    )

    if not retrieved_frames:
        return {
            "query": query,
            "video_id": video_id,
            "answer": "관련 프레임을 찾지 못했습니다. 검색어를 조금 다르게 입력해보세요.",
            "results": [],
            "used_llm": False,
        }

    context_text = _build_retrieved_context(retrieved_frames)
    prompt_hint = build_prompt_hint(user_id=user_id, video_id=video_id)

    answer = _generate_answer_with_langchain(
        query=query,
        retrieved_context=context_text,
        prompt_hint=prompt_hint,
    )

    used_llm = answer != DEFAULT_FALLBACK_MESSAGE

    if not used_llm:
        answer = _build_fallback_answer(
            query=query,
            retrieved_frames=retrieved_frames,
        )

    return {
        "query": query,
        "video_id": video_id,
        "answer": answer,
        "results": retrieved_frames,
        "used_llm": used_llm,
    }


def _build_retrieved_context(retrieved_frames: list[dict[str, Any]]) -> str:
    """
    검색된 프레임 metadata를 LLM에게 전달하기 좋은 텍스트로 변환한다.
    """
    lines = []

    for idx, frame in enumerate(retrieved_frames, start=1):
        timestamp = frame.get("timestamp")
        timestamp_text = f"{float(timestamp):.2f}초" if timestamp is not None else "알 수 없음"

        object_labels = frame.get("object_labels") or "감지 객체 없음"
        score = frame.get("score")
        score_text = f"{float(score):.4f}" if score is not None else "알 수 없음"

        lines.append(
            f"{idx}. video_id: {frame.get('video_id', 'unknown')}\n"
            f"   frame_id: {frame.get('frame_id', 'unknown')}\n"
            f"   timestamp: {timestamp_text}\n"
            f"   object_labels: {object_labels}\n"
            f"   similarity_score: {score_text}\n"
            f"   s3_key: {frame.get('s3_key') or '없음'}"
        )

    return "\n".join(lines)


def _generate_answer_with_langchain(
    query: str,
    retrieved_context: str,
    prompt_hint: str,
    model: str = "gpt-4o-mini",
) -> str:
    """
    LangChain을 사용해서 검색 결과 기반 답변을 생성한다.
    OPENAI_API_KEY가 없거나 호출 실패 시 fallback 메시지를 반환한다.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return DEFAULT_FALLBACK_MESSAGE

    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
너는 반려동물 영상 검색 서비스의 답변 도우미다.

규칙:
- 반드시 제공된 검색 결과 metadata만 근거로 답변한다.
- 영상을 직접 본 것처럼 단정하지 않는다.
- object_labels는 객체 탐지 모델의 결과이므로 오탐 가능성이 있음을 고려한다.
- "확실히 ~했다"가 아니라 "~로 보이는 후보가 있습니다", "~초 부근을 확인해볼 수 있습니다"처럼 조심스럽게 말한다.
- 사용자가 행동을 물어보더라도, 검색 결과만으로 행동을 확정할 수 없으면 확정하지 않는다.
- 답변에는 관련 시간대와 감지 객체를 포함한다.
- 한국어로 답변한다.
""".strip(),
            ),
            (
                "human",
                """
사용자 질문:
{query}

사용자/영상 맥락:
{prompt_hint}

검색된 프레임 근거:
{retrieved_context}

위 근거만 사용해서 답변해줘.
""".strip(),
            ),
        ])

        llm = ChatOpenAI(
            model=model,
            temperature=0,
        )

        chain = prompt | llm | StrOutputParser()

        return chain.invoke({
            "query": query,
            "prompt_hint": prompt_hint,
            "retrieved_context": retrieved_context,
        }).strip()

    except Exception as exc:
        print(f"LangChain RAG 답변 생성 실패: {exc}", flush=True)
        return DEFAULT_FALLBACK_MESSAGE


def _build_fallback_answer(
    query: str,
    retrieved_frames: list[dict[str, Any]],
) -> str:
    """
    LLM을 못 쓸 때도 검색 결과 기반으로 간단한 답변을 생성한다.
    """
    top_items = []

    for frame in retrieved_frames[:3]:
        timestamp = frame.get("timestamp")
        timestamp_text = f"{float(timestamp):.2f}초" if timestamp is not None else "알 수 없는 시간"
        labels = frame.get("object_labels") or "감지 객체 없음"
        score = frame.get("score")
        score_text = f"{float(score):.4f}" if score is not None else "알 수 없음"

        top_items.append(
            f"- {timestamp_text}: 감지 객체 `{labels}`, 유사도 {score_text}"
        )

    joined = "\n".join(top_items)

    return (
        f"질문 '{query}'에 대해 검색된 프레임 기준으로는 아래 시간대를 확인해볼 수 있습니다.\n\n"
        f"{joined}\n\n"
        "단, 현재 답변은 검색 metadata 기반이며, 실제 행동을 확정하는 것은 아닙니다."
    )