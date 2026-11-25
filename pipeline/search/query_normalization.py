"""Normalize bounded colloquial variants before choosing a retrieval route."""

import re


# A furniture-only noun phrase can omit both the subject particle and spaces.
# Do not rewrite an animal crossing furniture ("강아지가 소파를 넘어간").
_FURNITURE_TIPPING = re.compile(
    r"^(?P<object>소파|의자|탁자|책상|선반|식탁)(?:가|이)?\s*"
    r"넘어\s*(?:간|갔던|가는)\s*(?P<scene>장면|구간)(?=$|[\s?!.,])"
)


def normalize_search_query(query: str) -> str:
    query = query.strip().replace("쇼파", "소파")
    return _FURNITURE_TIPPING.sub(
        lambda match: (
            f"{match['object']}{'이' if (ord(match['object'][-1]) - 0xAC00) % 28 else '가'} "
            f"넘어진 {match['scene']}"
        ), query,
    )
