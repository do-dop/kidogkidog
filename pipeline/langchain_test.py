import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def main() -> None:
    # 프로젝트 루트의 .env 파일 로드
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a pet behavior assistant. "
            "Answer only based on the provided retrieved frame results. "
            "If the evidence is weak, say it is uncertain. "
            "Do not guess unseen events."
        ),
        (
            "user",
            """User query: {query}

Retrieved frame results:
{results}

Please explain:
1. Which frame is the best match
2. Why it seems to be the best match
3. Mention uncertainty if the similarity scores are low
"""
        ),
    ])

    chain = prompt | llm

    result = chain.invoke({
        "query": "Find the scene where the dog is eating food",
        "results": """
1. frame_4_40s.jpg, score=0.2812
2. frame_6_41s.jpg, score=0.2635
3. frame_2_40s.jpg, score=0.2411
""",
    })

    print("\n=== LangChain Response ===")
    print(result.content)


if __name__ == "__main__":
    main()